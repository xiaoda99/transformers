import sys
from collections import OrderedDict, defaultdict
# from typing import Iterable
from collections.abc import Iterable
import numpy as np
import math
from dataclasses import dataclass, field, fields
from functools import reduce, partial
from itertools import chain, product, combinations, cycle
import types
from tqdm import tqdm
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.manifold import TSNE, MDS
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F 

import einops
from einops import rearrange

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention, GPTNeoBlock
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJModel
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXModel, GPTNeoXForCausalLM
from transformers.models.xglm.modeling_xglm import XGLMForCausalLM, XGLMAttention, XGLMDecoderLayer

sys.path.insert(0, '/nas/xd/projects/pptree')
from pptree import Node, print_tree

from common_utils import numpy, einsum, my_isinstance, convert_ids_to_tokens, show_topk, topk_md, topi_md, \
    equal, join_lists, iterable, pad, Timer, maybe_map, reduce_objects, mr, maybe_mr, list_get, fn2str

from child_utils import make_data_tuple, get_answer_index
from weight_analysis import get_head_weights

@dataclass
class Outputs:
    inputs_embeds: torch.FloatTensor = None
    position_embeds: torch.FloatTensor = None
    attn_outputs: tuple = ()
    head_inputs: tuple = ()
    head_outputs: tuple = ()
    intermediates: tuple = ()
    mlp_outputs: tuple = ()
    hidden_states: tuple = ()
    attentions: tuple = ()
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None
    loss: torch.FloatTensor = None
    attn_attr: dict = field(default_factory=dict)

@dataclass
class Attributions:
    embed: torch.FloatTensor = 0.
    attn: torch.FloatTensor = 0.
    head: torch.FloatTensor = 0.
    neuron: torch.FloatTensor = 0.
    mlp: torch.FloatTensor = 0.

@dataclass
class AttrData:
    step: int = None
    topi: int = None
    layer: int = None
    head: int = None
    label_type: str = None
    attn_pattern: str = None
    attribute_k: bool = False
    attr: Attributions = None
    scores: dict = None
    top_score: list = None
    dummy: bool = False

@dataclass
class Result:
    task: tuple = None
    trans_args: dict = None
    args: dict = None
    all_examples: list = None
    texts: list = None
    data_tuples: list = None
    node: Node = None
    mean_loss: float = None
    mean_acc: float = None

@dataclass
class Eigovs:
    ov: torch.FloatTensor = None

@dataclass
class Heads:
    ov: torch.FloatTensor = None

def fill_list(e, length, i, default_e=None): # fill e to ith position of a list of default_es
    if isinstance(e, (list, tuple)):
        assert len(e) == length, f'{len(e)} != {length}'; return e
    l = [default_e for _ in range(length)]
    if i is not None: l[i] = e
    return l

def default_get_hqkv(h): return h, h, h  # h is used for query, key and value
def get_hqkv_k(h, h0): return h0, h, h0  # h is only used for key


def unify(model):
    if my_isinstance(model, XGLMForCausalLM):
        model.transformer = model.model
        model.model.h = model.model.layers
        model.model.ln_f = model.model.layer_norm
    elif my_isinstance(model, GPTNeoXForCausalLM):
        model.transformer = model.gpt_neox
        model.lm_head = model.embed_out
        model.transformer.wte = model.transformer.embed_in
        model.transformer.h = model.transformer.layers
        model.transformer.ln_f = model.transformer.final_layer_norm
        model.transformer.drop = nn.Dropout(0)

    for i, block in enumerate(model.transformer.h):
        if my_isinstance(block, XGLMDecoderLayer):
            block.attn = block.self_attn
            block.ln_1 = block.self_attn_layer_norm
            block.ln_2 = block.final_layer_norm
        elif my_isinstance(block, GPTNeoBlock):
            block.attn = block.attn.attention
        elif my_isinstance(block, GPTJBlock):
            block.ln_2 = block.ln_1
        elif my_isinstance(block, GPTNeoXLayer):
            block.attn = block.attention
            block.ln_1 = block.input_layernorm
            block.ln_2 = block.post_attention_layernorm
        self = block.attn
        if my_isinstance(self, GPT2Attention):
            embed_dim = self.c_proj.weight.size(0)
            for proj_name, w, b in zip(['q_proj', 'k_proj', 'v_proj'], 
                                        self.c_attn.weight.chunk(3, dim=1), 
                                        self.c_attn.bias.chunk(3)):
                proj = nn.Linear(embed_dim, embed_dim)
                proj.weight, proj.bias = nn.Parameter(w.T), nn.Parameter(b)
                setattr(self, proj_name, proj)
            # transformed from Conv1D, which has transposed weight compared to Linear
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj.weight = nn.Parameter(self.c_proj.weight.T)
            self.out_proj.bias = self.c_proj.bias
        elif my_isinstance(self, GPTNeoSelfAttention):
            self.attention_type = 'global' if i % 2 == 0 else 'local'
        elif my_isinstance(self, XGLMAttention):
            self.attn_dropout = nn.Dropout(0)
            self.resid_dropout = nn.Dropout(0)
        elif my_isinstance(self, GPTJAttention):
            self.num_heads = self.num_attention_heads
        elif my_isinstance(self, GPTNeoXAttention):
            for old_name, new_name in [('num_attention_heads', 'num_heads'), ('hidden_size', 'embed_dim'),
                        ('head_size', 'head_dim'), ('rotary_ndims', 'rotary_dim'), ('dense', 'out_proj')]:
                setattr(self, new_name, getattr(self, old_name))
            self.attn_dropout = nn.Dropout(0)
            self.resid_dropout = nn.Dropout(0)
            from bitsandbytes.nn import Linear8bitLt
            if my_isinstance(self.query_key_value, Linear8bitLt):
                self.qkv_proj = self.query_key_value
            elif not hasattr(self, 'q_proj'):
                weight_chunks = self.query_key_value.weight.view(
                    self.num_heads, 3 * self.head_dim, self.embed_dim).chunk(3, dim=1)
                bias_chunks = self.query_key_value.bias.chunk(3)
                for proj_name, w, b in zip(['q_proj', 'k_proj', 'v_proj'], weight_chunks, bias_chunks):
                    proj = nn.Linear(self.embed_dim, self.embed_dim)
                    proj.weight, proj.bias = nn.Parameter(rearrange(w, 'n d e -> (n d) e')), nn.Parameter(b)
                    setattr(self, proj_name, proj)

# def get_data_as(param, tensor):
#     if param.data.device == tensor.device: return param.data
#     new_name = 'data_' + str(tensor.device).replace(':', '_')  # 'cuda:0' -> 'cuda_0'
#     if not hasattr(param, new_name):
#         setattr(param, new_name, param.data.to(tensor.device).type_as(tensor))
#     return getattr(param, new_name)

def scaled_ln(ln, x, scaled=True):
    if not scaled: return ln(x)
    self = ln
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    std = std[-1:].detach()  # use the std of the last original (unscaled) example in the batch
    # weight, bias = get_data_as(self.weight, x), get_data_as(self.bias, x)
    y = (x - mean) * self.weight / std + self.bias
    return y

def scaled_ln_wrapper(ln): return lambda x: scaled_ln(ln, x)

def custom_ln(ln, x, mean=None, std=None, bias=True):
    self = ln
    if mean is None:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
    y = (x - mean) / std * self.weight + self.bias if bias else x / std * self.weight
    return y, mean.detach(), std.detach()

def ln2statefulfn(ln):
    def fn(x):
        self = ln
        if getattr(fn, 'mean', None) is None:
            fn.mean = x.mean(dim=-1, keepdim=True); print('set state')
        if getattr(fn, 'std', None) is None:
            fn.std = (((x - fn.mean) ** 2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        y = (x - fn.mean) / fn.std * self.weight + self.bias
        return y 
    return fn

def embed_forward(transformer, inputs, output_embeds=True):
    self = transformer
    input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs.input_ids
    hidden_states = inputs_embeds = self.wte(input_ids)
    position_embeds = None
    if not my_isinstance(transformer, (GPTJModel, GPTNeoXModel)):
        input_shape = input_ids.size()
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeds = self.wpe(position_ids) 
        hidden_states = hidden_states + position_embeds
    hidden_states = self.drop(hidden_states)
    return (hidden_states, inputs_embeds, position_embeds) if output_embeds else hidden_states

def _split_heads(tensor, num_heads, attn_head_size, rotary=False):
    '''b i (n d) -> b n i d'''
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    if rotary: return tensor  # bind
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

def _merge_heads(tensor, num_heads, attn_head_size):
    '''b n i d -> b i (n d)'''
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)

class RotaryEmbedding(torch.nn.Module):  # from gpt-neox
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        # self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)

def fixed_pos_embedding(x, seq_len=None, gpt_neox_style=False):
    self = fixed_pos_embedding; device = x.device
    self.sin, self.cos = {}, {}
    if device not in self.sin: #not hasattr(self, 'sin'):
        seq_len = 2048
        if gpt_neox_style:
            rotary_emb = RotaryEmbedding(x.shape[-1], 2048, base=10000)
            self.cos[device], self.sin[device] = rotary_emb(x, seq_len=seq_len)
        else:
            dim = x.shape[-1]
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
            sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(device).float()
            self.sin[device], self.cos[device] = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
    return self.sin[device], self.cos[device]

def rotate_every_two(x):  # gpt-j
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def rotate_half(x):  # gpt-neox
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_every_two(x, sincos, offset=0): # x: bnir
    # i(r/2) -> 11i(r/2) -> 11ir
    sin, cos = map(lambda t: t[None, None, offset : x.shape[2] + offset, :].repeat_interleave(2, 3), sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[2]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)  # binr,1i1r->binr

def apply_rotary_pos_emb_half(x, sincos, offset=0): # x: bnir
    fn = (lambda t: t[None, None, offset : x.shape[2] + offset, :].repeat(1, 1, 1, 2) # gpt_j_style: i(r/2)->11i(r/2)->11ir
        if t.ndim == 2 else t[:, :, offset : x.shape[2] + offset, :]) # gpt_neox_style
    sin, cos = map(fn, sincos)
    return (x * cos) + (rotate_half(x) * sin)  # bnir,11ir->bnir

def apply_rotary_pos_emb(query_rot, key_rot, seq_len=2048, offset=0, is_gpt_neox=False):
    sincos = fixed_pos_embedding(key_rot, seq_len=seq_len, gpt_neox_style=False)
    apply_rotary_pos_emb_fn = apply_rotary_pos_emb_half if is_gpt_neox else apply_rotary_pos_emb_every_two
    key_rot = apply_rotary_pos_emb_fn(key_rot, sincos, offset=offset)
    query_rot = apply_rotary_pos_emb_fn(query_rot, sincos, offset=offset)
    return query_rot, key_rot

def _attn(self, query, key, value, attention_mask=None):
    if isinstance(self, GPTNeoSelfAttention) or isinstance(self, GPTJAttention):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query, key = query.to(torch.float32), key.to(torch.float32)
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) # bnid,bnjd->bnij

    # turns out gptneo fold_scaling_into_initializer, and uses float32_logits. 
    # see https://crfm.stanford.edu/2021/08/26/mistral.html (Diagnosing Numerical Instability, Eureka!)
    # and https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/attention.py#L517&66
    if not isinstance(self, GPTNeoSelfAttention):
        # attn_weights = attn_weights / (value.size(-1) ** 0.5) # vale may be None
        attn_weights = attn_weights / (query.size(-1) ** 0.5)

    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
    attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
    if attention_mask is not None: attn_weights = attn_weights + attention_mask
    if value is None: return None, attn_weights

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value) # bnij,bnjd->bnid
    return attn_output, attn_weights

def backup_heads(self):
    self.backup_names = ['num_heads', 'q_proj', 'k_proj', 'v_proj', 'out_proj']
    assert self.num_heads == 16, f'{self.num_heads} != 16'  # for gpt-j
    for name in self.backup_names: setattr(self, name + '0', getattr(self, name))

def restore_heads(self):
    assert self.num_heads0 == 16, f'{self.num_heads0} != 16'  # for gpt-j
    for name in self.backup_names:
        setattr(self, name , getattr(self, name + '0'))
        # delattr(self, name + '0')

def trim_heads(self, kept_heads, proj_names=None, new_weight=False):
    if proj_names is None: proj_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    backup_heads(self)
    for proj_name in proj_names:
        assert getattr(self, proj_name).bias is None #lxy，报错
        weight = getattr(self, proj_name).weight
        if proj_name == 'out_proj':
            patterns = ['e (n d) -> n d e', 'n d e -> e (n d)']
            size = (self.head_dim * len(kept_heads), self.embed_dim)
        else:
            patterns = ['(n d) e -> n d e', 'n d e -> (n d) e']
            size = (self.embed_dim, self.head_dim * len(kept_heads))
        weight = rearrange(weight, patterns[0], n=self.num_heads)[kept_heads] #[kept_heads,head_dim,embed_dim]
        weight = rearrange(weight, patterns[1])
        if new_weight: weight = weight.clone().etach()
        proj = nn.Linear(*size, bias=False)
        proj.weight = nn.Parameter(weight)
        setattr(self, proj_name, proj)
    self.num_heads = len(kept_heads)

def compose_heads(model, attn_mod, composed_heads):
    k = len(composed_heads)
    wqs, wks = zip(*[get_head_weights(model, *qk_head, transpose=False)[:2] if type(qk_head) != str else (None, None)
                     for qk_head, _ in composed_heads])
    wvs, wos = zip(*[get_head_weights(model, *ov_head, transpose=False)[2:] for _, ov_head in composed_heads])
    for proj_name, ws in zip(['q', 'k', 'v', 'out'], [wqs, wks, wvs, wos]):
        assert all(w is None for w in ws) or not any(w is None for w in ws)  # wqs or wks
        if all(w is None for w in ws):
            proj = None  # use attn_pattern as qk
        else:
            size, cat_dim = (((ws[0].size(1), k * ws[0].size(0)), 0)  # qkv: e(kd)
                if proj_name != 'out' else ((k * ws[0].size(1), ws[0].size(0)), 1))  # out: (kd)e
            proj = nn.Linear(*size, bias=False)
            proj.weight = nn.Parameter(torch.cat(ws, dim=cat_dim))  # qkv: k*[de]->(kd)e, out: k*[ed]->e(kd)
        setattr(attn_mod, f'composed_{proj_name}_proj', proj)

def remove_composed_heads(attn_mod):
    try_delattr(attn_mod, 'composed_heads')
    for proj_name in ['q', 'k', 'v', 'out']:
        try_delattr(attn_mod, f'composed_{proj_name}_proj')

def get_all_composed_heads(model):
    r = []
    for i, block in enumerate(model.transformer.h):
        composed_heads = getattr(block.attn, 'composed_heads', None)
        if composed_heads: r.append((i, composed_heads))
    return r

def composed_heads2str(model):
    all_composed_heads = join_lists([composed_heads for l, composed_heads in get_all_composed_heads(model)])
    d = defaultdict(list)
    for qk_head, ov_head in all_composed_heads: d[qk_head].append(ov_head)
    
    def head2str(head): return f'{head[0]}-{head[1]}' if type(head) != str else head
    s = '_'.join(head2str(qk_head) + ':' + ','.join(head2str(h) for h in ov_heads) for qk_head, ov_heads in d.items())
    if s != '': s = '_' + s
    return s

def use_composed_heads(attn_mod):
    backup_heads(attn_mod)
    for proj_name in ['q', 'k', 'v', 'out']:
        setattr(attn_mod, f'{proj_name}_proj', getattr(attn_mod, f'composed_{proj_name}_proj'))
    attn_mod.num_heads = len(attn_mod.composed_heads)

def scale_to(tensor, target):
    assert tensor.size() == target.size(), f'{tensor.size()} != {target.size()}'
    tensor = tensor * (target.norm(dim=-1) / tensor.norm(dim=-1)).unsqueeze(-1)
    assert torch.allclose(tensor.norm(dim=-1), target.norm(dim=-1))  # debug
    return tensor

def make_dropout_head_mask(dropout_heads, ranges_i, ranges, hidden_states_size, L, H, p=0.5):
    bsz, qlen, _ = hidden_states_size  # bie
    head_mask = torch.ones(bsz, H, qlen)  # bni
    head_mask = []
    return head_mask

def maybe_composed_attn_forward(model, block, hq, hk, hv, by_head=True, ranges=None, **kwargs):
    self = block.attn
    composed_heads, has_pred_heads = getattr(self, 'composed_heads', None), getattr(self, 'has_pred_heads', False)
    if composed_heads: by_head = list(set((by_head or []) +  ['head_output']))
    attn_output, aw, head_input, head_output = attn_forward(block, hq, hk, hv, by_head=by_head, **kwargs)
    
    if composed_heads:
        try:
            use_composed_heads(self)
            attn_size = aw.size()[-2:]
            aw2 = (torch.stack([attn_pattern2labels(ranges, qk_head, attn_size)  # ij
                    for qk_head, (l, h) in self.composed_heads]).unsqueeze(0)  # m*[ij]->mij->1mij
                    if self.q_proj is None else None)
            _, aw2, _, head_output2 = attn_forward(block, hq, hk, hv, by_head=['head_output'], attn_weights=aw2)
            for i, ((qk_head, (l, h)), range_i) in enumerate(zip(composed_heads, self.ranges_i)):
                assert range_i.endswith('->*')
                if type(qk_head) == str: assert qk_head.split('->')[0] == range_i.split('->')[0], f'{qk_head}.q != {range_i}.q'
                range_i = attn_pattern2labels(ranges, range_i, attn_size)[:, 0] != 0  # ij-i
                aw[:, h, range_i] = aw2[:, i, range_i]  # bnij->brj = bmij->brj
                head_output[:, h, range_i] = head_output2[:, i, range_i]  # bnie->bre = bmie->bre
            attn_output = head_output.sum(1)  # bnie->bie  # assume no out_proj.b
            if False:  # predicting heads
                old = head_output[:, [h for _, (l, h) in composed_heads]]  # bmie
                attn_output = attn_output + (scale_to(head_output2, old) - old).sum(1) # bmie->bie
                # old_logits, new_logits = [], []  # debug
                # for i, (_, (layer, head)) in enumerate(composed_heads):  # ov part
                #     assert layer == block.layer
                #     old, new = head_output[:, head], head_output2[:, i]  # bnie->bie or bmie->bie
                #     attn_output = attn_output - old + scale_to(new, old) # magic happens here
                #     # debug
                #     old_logits.append(model.lm_head(model.transformer.ln_f(old)))
                #     new_logits.append(model.lm_head(model.transformer.ln_f(new)))
            # head_output = torch.cat([head_output, head_output2], dim=1)  # bnie,bmie->b(n+m)ie
            # aw = torch.cat([aw, aw2], dim=1)  # bnij,bmij->b(n+m)ij
        finally: restore_heads(self)
    return attn_output, aw, head_input, head_output

def attn_forward(block, hq, hk, hv, by_head=None,
                attention_mask=None, head_mask=None, attn_weights=None):
    self = block.attn  # block.attn.attention already renamed
    query, key, value = None, None, None
    if hasattr(self, 'qkv_proj'):  # gpt-neox
        qkv = self.qkv_proj(hq)  # hq == hk == hv
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_dim)
        query, key, value = [rearrange(a, 'b i n d -> b n i d') # [bind] * 3 -> [bnid] * 3
            for a in qkv.view(*new_qkv_shape).chunk(3, dim=-1)] # bin(3d) -> [bind] * 3
        if hv is None: value = None
    else:
        if hq is not None and hk is not None and attn_weights is None:
            key = self.k_proj(hk)
            key = _split_heads(key, self.num_heads, self.head_dim)#, rotary=rotary)  # bind
            if hq.ndim == 3:  # bie
                query = self.q_proj(hq)
                query = _split_heads(query, self.num_heads, self.head_dim)#, rotary=rotary) # bind
            else:
                assert hq.ndim == 4  # bnid, computed in cat_attn_forward
                # if rotary: hq = rearrange(hq, 'b n i d -> b i n d')
                assert hq.size()[1:] == key.size()[1:], f'{hq.size()} != {key.size()}'
                query = hq
        if hv is not None: value = _split_heads(self.v_proj(hv), self.num_heads, self.head_dim)
    rotary = my_isinstance(self, (GPTJAttention, GPTNeoXAttention))
    if rotary and query is not None and key is not None:
        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]
        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]

        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, is_gpt_neox=my_isinstance(self, GPTNeoXAttention))

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
    attn_output, attn_weights = _attn(self, query, key, value, attention_mask) \
        if attn_weights is None else (attn_weights.to(value.device) @ value, attn_weights)  # XD
    if value is None: return None, attn_weights, None, None

    if head_mask is not None: attn_output = einsum('bnid,bni->bnid', attn_output, head_mask)

    head_input, head_output = None, None
    if by_head:
        w_o = self.out_proj.weight.data
        do_bmm = True # not (any('0' in s for s in by_head) or w_o.dtype == torch.int8)
        if w_o.dtype == torch.int8: w_o = self.out_proj.weight.data0  # float16 weight on cpu
        if do_bmm: w_o = rearrange(w_o, 'e (n d) -> n d e', n=self.num_heads)
        else: f = partial(get_head_io, num_heads=self.num_heads)
        # for smaller models (e.g. gpt-j) on gpu, @ then to-cpu may be faster than to-cpu then @
        if 'head_output' in by_head: #any('head_output' in s for s in by_head):
            head_output = attn_output.to('cpu').float() @ w_o.to('cpu').float() if do_bmm else (f, attn_output.to('cpu').float(), w_o)  # bnid,nde->bnie
        if 'head_input' in by_head: #any('head_input' in s for s in by_head):
            head_input = value.to('cpu').float() @ w_o.to('cpu').float() if do_bmm else (f, value.to('cpu').float(), w_o.float())  # bnid,nde->bnie

    attn_output = _merge_heads(attn_output, self.num_heads, self.head_dim) # bnid->bi(nd)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    return attn_output, attn_weights, head_input, head_output

def head_forward(model, hidden_states, layer, head, labels=None, loss_reduction=None,
            attn_weights=None, attn_mask=None, attn_labels=None, hidden_states0=None, attribute_k=False, trim=True, scaled=True):
    assert not isinstance(layer, Iterable)  # should use mixed_forward, which is clearer and simpler
    block = model.transformer.h[layer]
    # only hq and hv can be scaled, not hk
    hk = block.ln_1(hidden_states0 if hidden_states0 is not None else hidden_states)
    h = scaled_ln(block.ln_1, hidden_states, scaled=scaled) #if scaled else block.ln_1(hidden_states)
    by_head = None
    if attn_weights is not None:
        hq, hv = None, h
        by_head = ['head_output']
    elif attn_labels is not None:
        hq, hv = h, None # return attn_logits instead of attn_weights by passing None hv 
        if attribute_k: hq, hk = hk, hq

    if trim:
        self = block.attn
        try:
            trim_heads(self, [head])
            if attn_weights is not None: attn_weights = attn_weights[:, [head]] # bnij->b1ij
            _, attn_logits, _, head_output = attn_forward(block, hq, hk, hv, attn_weights=attn_weights, by_head=by_head)
            if head_output is not None: assert head_output.size(1) == 1, str(head_output.size())
            head = 0  # used to index head_output and attn_logits
        finally: restore_heads(self)
    else:
        _, attn_logits, _, head_output = attn_forward(block, hq, hk, hv, attn_weights=attn_weights, by_head=by_head)
    if head_output is not None: head_output = head_output[:, head]
    logits, loss = None, None
    if labels is not None:
        logits, loss = lm_head_forward(model, head_output, labels=labels, loss_reduction=loss_reduction, scaled=scaled)
    elif attn_labels is not None:
        # may do some attn_logits masking here
        logits = attn_logits[:, head]  # bnij->bij
        if not attribute_k: logits = (logits + (attn_mask * -1e9 if attn_mask is not None else 0)).log_softmax(-1)
        loss = -torch.einsum('bij->b', logits * attn_labels) # per_example_sum. per_example_mean is hard to define when using unormalized attn attr  # bug fix
    return Outputs(hidden_states=(head_output,) if head_output is not None else (), logits=logits, loss=loss)

def mlp_forward(block, hidden_states, layer=None, output_intermediate=False, 
                labels=None, loss_reduction=None, scaled=False):
    if layer is not None:
        model = block
        block = model.transformer.h[layer]
    hidden_states = scaled_ln(block.ln_2, hidden_states, scaled=scaled)
    if layer is None: return block.mlp(hidden_states, output_intermediate=output_intermediate)
    hidden_states = block.mlp(hidden_states, output_intermediate=False)
    logits, loss = lm_head_forward(model, hidden_states, labels=labels, loss_reduction=loss_reduction, scaled=scaled) \
        if labels is not None else (None, None)
    return Outputs(hidden_states=(hidden_states,), logits=logits, loss=loss)

def get_attn_pattern_from_label_type(label_type):
    if ':' not in label_type: return None, None
    attn_pattern, k_shot = ('bos->ans0]', 3) if label_type.endswith(':') else \
        (label_type.split(':')[1].split(',')[0], (int(label_type[-1]) if label_type[-1].isdigit() else None))
    return attn_pattern, k_shot

def mixed_forward(model, hidden_states, layer, head, labels=None, label_type=None,
    outputs=None, attn_attr=None, hidden_states0=None, ranges=None, attribute_k=False, **kwargs):
    if not isinstance(layer, Iterable): layer, head = [layer], [head]
    H = outputs.attentions[0].size(1)
    if any(h == H for h in head): assert label_type is None or 'attn_labels' not in label_type
    logits_after_sum = label_type is not None and 'attn_labels' not in label_type
    if not isinstance(hidden_states, (list, tuple)): hidden_states = [hidden_states] * len(layer)
    else: assert len(layer) == len(head) == len(hidden_states), f'{len(layer)}, {len(head)}, {len(hidden_states)}'

    def get_kwargs(layer, head):
        if label_type and 'attn_labels' in label_type:
            assert head < H
            attn_size = outputs.attentions[0].size()[-2:]
            if label_type.startswith('argmax'):
                attn_labels = get_argmax_attn_labels(outputs, layer, head, labels=labels)
            elif ':' in label_type:  # e.g. 'attn_lables:bos->ans0],3'
                attn_pattern, k_shot = get_attn_pattern_from_label_type(label_type)
                attn_labels = attn_pattern2labels(ranges, attn_pattern, attn_size, k_shot=k_shot)
            else:
                attn_labels = attn_attr[layer, head]
            kwargs = {'attn_labels': attn_labels, 'hidden_states0': hidden_states0 
                    if hidden_states0 is not None else outputs.hidden_states[layer]}
            if '/' in label_type:
                mask_type = label_type.split('/')[1]
                kwargs['attn_mask'] = attn_pattern2labels(ranges, 'bos->'+mask_type, attn_size)
            return kwargs
        kwargs = {'attn_weights': outputs.attentions[layer]} if head < H else {}
        if label_type in ['labels', 'argmax_labels']:
            kwargs['labels'] = get_argmax_labels(model, outputs.head_outputs[layer][:, head], labels) \
                if label_type == 'argmax_labels' else labels
        return kwargs

    if len(layer) > 1: assert label_type in [None, 'labels'], f'{len(layer)}, {label_type}'
    fwd_outputs = [head_forward(model, hs, l, h, attribute_k=attribute_k, **get_kwargs(l, h), **kwargs) 
        if h < H else mlp_forward(model, hs, layer=l, **get_kwargs(l, h), **kwargs)
        for l, h, hs in zip(layer, head, hidden_states)]
    if len(fwd_outputs) == 1: return fwd_outputs[0]

    def try_sum(l):
        none_flags = [i is None for i in l]
        assert all(none_flags) or not any(none_flags), str([i.size() if isinstance(i, torch.Tensor) else i for i in l])
        return sum(l) if not any(none_flags) else None
    assert all(isinstance(o.hidden_states, tuple) and len(o.hidden_states) in [0, 1] for o in fwd_outputs), \
        str([(type(o.hidden_states), len(o.hidden_states)) for o in fwd_outputs])
    all_hidden_states, all_logits, all_loss = zip(*[
        (o.hidden_states[0] if len(o.hidden_states) == 1 else None, o.logits, o.loss) for o in fwd_outputs])
    hidden_states, logits, loss = [try_sum(l) for l in [all_hidden_states, all_logits, all_loss]]
    if logits_after_sum: logits, loss = lm_head_forward(model, hidden_states, labels=labels, **kwargs)
    hidden_states = (hidden_states,) if hidden_states is not None else ()
    return Outputs(hidden_states=hidden_states, logits=logits, loss=loss)

def lm_head_forward(model, hidden_states, labels=None, loss_reduction=None, compact=False, scaled=False):
    if compact and labels is not None:
        if labels.size(0) != hidden_states.size(0): labels = einops.repeat(labels, '1 i -> b i', b=hidden_states.size(0))
        valid_flags = labels != -100
        n_valid = torch.einsum('bi->b', valid_flags)[0].item()
        hidden_states = rearrange(hidden_states[valid_flags], '(b i) e -> b i e', b=hidden_states.size(0), i=n_valid)
        labels = rearrange(labels[valid_flags], '(b i) -> b i', b=labels.size(0), i=n_valid)
    logits = model.lm_head(scaled_ln(model.transformer.ln_f, hidden_states, scaled=scaled))
    loss = compute_loss(logits, labels, reduction=loss_reduction) if labels is not None else None
    if compact:
        logits0 = logits.new(logits.size(0), valid_flags.size(1), logits.size(2)).zero_()
        logits0[valid_flags] = rearrange(logits, 'b i v -> (b i) v')
        logits = logits0
    return logits, loss

def compute_loss(logits, labels, reduction=None):
    if reduction is None: reduction = 'per_example_mean'
    # print('in compute_loss, labels =', labels)
    if reduction == 'argmax':
        labels = labels.clone()
        labels[labels != -100] = logits[-1:].argmax(-1)[labels != -100]
        reduction = 'per_example_mean'
    if labels.size(0) < logits.size(0): # logits has been scaled
        labels = einops.repeat(labels, '1 i -> b i', b=logits.size(0))
    loss_fct = nn.CrossEntropyLoss(reduction='none' if reduction == 'per_example_mean' else reduction)
    # print(f'logits.size = {logits.size()}, labels.size = {labels.size()}')
    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    if reduction != 'mean':
        loss = loss.view(labels.size(0), -1) #4,16
        if reduction == 'per_example_mean':
            # loss = einops.reduce(loss, 'b i -> b', 'sum') / einops.reduce(labels != -100, 'b i -> b', 'sum')
            # print('in compute_loss, before loss =', loss)
            loss = torch.einsum('bi->b', loss) / torch.einsum('bi->b', labels != -100)
            # print('in compute_loss, after loss =', loss)
    return loss

def block2gpu(block, mlp_to_gpu=False):
    names = ['ln_1', 'attn']
    if mlp_to_gpu: names += ['ln_2', 'mlp']
    for name in names: setattr(block, name, getattr(block, name + '_gpu'))

def block2cpu(block, mlp_to_gpu=False):
    names = ['ln_1', 'attn']
    if mlp_to_gpu: names += ['ln_2', 'mlp']
    for name in names: setattr(block, name, getattr(block, name + '_cpu'))

def cat_attn_forward0(block, cat_hidden_states, sum_hidden_states, mask=None, attn_labels=None, scaled=True):
    # cat_hidden_states = special_head_outputs  sie
    cat_hidden_states = torch.cat([sum_hidden_states - cat_hidden_states.sum(0), cat_hidden_states]) # cat(1ie - sie->ie, sie) -> (1+s)ie
    hidden_states = torch.einsum('bnpi,pie->bnie', mask, cat_hidden_states) # p=1+s

    # hidden_states = torch.einsum('bnloi,loie->bnie', mask, cat_hidden_states) # o=n+2
    # print(equal(hidden_states[-1:, 0], sum_hidden_states))  # bnie->1ie
    hq, hk, hv = scaled_ln(block.ln_1, hidden_states, scaled=scaled), block.ln_1(sum_hidden_states), None
    self = block.attn
    assert self.q_proj.bias is None
    wq = rearrange(block.attn.q_proj.weight, '(n d) e -> n e d', n=self.num_heads)
    query = hq @ wq  # bnie,ned->bnid, the most important line
    attn_logits = attn_forward(block, query, hk, hv)[1]
    loss = None
    if attn_labels is not None:
        loss = -torch.einsum('bnij->b', attn_logits.log_softmax(-1) * attn_labels)
    return Outputs(hidden_states=(hidden_states), logits=attn_logits, loss=loss)

def get_multiplier(block, hidden_states, attentions, head_input, special_head_output, labels, layer,
        attr_threshold=1, base_multiplier=1.5, verbose=False):
    H = attentions.size(1)
    # mask = torch.ones(1, H, L, H + 2, outputs.cat_hidden_states.size(-2)); mask[:, :, layer:] = 0 # bnloi
    # x = {'mask': mask}
    
    # special_head_output: sie
    x = {'mask': torch.ones(1, H, special_head_output.size(0) + 1, special_head_output.size(1))}  # bnpi

    attn_labels = torch.einsum('bnij,bnj->bnij', attentions, head_input.norm(dim=-1)) # bnje->bnj
    # attn_labels = attn_labels / attn_labels.sum(-1, keepdim=True)  # bnij->bni1 
    fwd_fn = partial(cat_attn_forward0, cat_hidden_states=special_head_output, #outputs.cat_hidden_states, 
                sum_hidden_states=hidden_states, attn_labels=attn_labels)
    attr, ys, logits = _attribute(fwd_fn, block, x, num_points=3)

    a = attr['mask']  # nloi, nsi
    # a = torch.add(a[:, 8, 1, :]*0, a[:, 12, 10, :]) # ni
    a = a[:, -1, :]  # ni

    labels_mask = (labels != -100).squeeze(0)#.fill_(1)
    multiplier_mask = (a >= attr_threshold) * labels_mask
    if verbose and multiplier_mask.sum() > 0:
        for head in range(H):
            n_multiplied = multiplier_mask[head].sum()
            if n_multiplied > 0:
                print(f'{layer}-{head}', n_multiplied, (a * labels_mask)[head].topk(min(a.size(1), n_multiplied + 1)))
    return base_multiplier * multiplier_mask + 1 * ~multiplier_mask

def cat_attn_forward(block, hidden_states0, cat_hidden_states, mask=None, attn_labels=None, scaled=True):
    # cat_hidden_states = torch.cat([cat_hidden_states, hidden_states0 - cat_hidden_states.sum(0)]) # cat(sie, 1ie - sie->ie) -> (s+1)ie
    # hidden_states = torch.einsum('bnpi,pie->bnie', mask, cat_hidden_states) # p=1+s
    # hidden_states is replicated n times to keep track of grad for each head
    hidden_states = torch.einsum('bnmri,mrie->bnie', mask, cat_hidden_states) # m=m+1
    # hidden_states = torch.einsum('bnloi,loie->bnie', mask, cat_hidden_states) # o=n+2

    hq, hk, hv = scaled_ln(block.ln_1, hidden_states, scaled=scaled), block.ln_1(hidden_states0), None
    self = block.attn
    assert self.q_proj.bias is None
    wq = rearrange(self.q_proj.weight, '(n d) e -> n e d', n=self.num_heads)
    query = hq @ wq  # bnie,ned->bnid, the most important line
    attn_logits = attn_forward(block, query, hk, hv)[1]
    loss = -torch.einsum('bnij->b', attn_logits.log_softmax(-1) * attn_labels) \
        if attn_labels is not None else None
    return Outputs(hidden_states=(hidden_states), logits=attn_logits, loss=loss)

def get_self_attr(block, hidden_states0, cat_hidden_states, attn_labels, device='cpu'):
    H = block.attn.num_heads
    # cat_hidden_states: (m+1)(r+1)ie
    x = {'mask': torch.ones(1, H, *cat_hidden_states.size()[:-1]).to(device)}  # bn(m+1)(r+1)i

    fwd_fn = partial(cat_attn_forward, hidden_states0=hidden_states0,
                cat_hidden_states=cat_hidden_states, attn_labels=attn_labels)
    attr, _, _ = _attribute(fwd_fn, block, x, num_points=3)

    a = attr['mask']  # nloi, nsi, n(m+1)(r+1)i
    return a

def to(a, device):
    if isinstance(a, torch.Tensor):
        return a.to(device).float() if device == 'cpu' else a.to(device)
    return a  # may be None

def forward0(model, inputs, labels=None, loss_reduction=None, by_head=None, ranges=None, attribute_layer=None, 
            head_mask=None, mlp_mask=None, attn_weights=None, hidden_states=None, detach_layer=None,
            special_head=(None, None), special_head_multiplier=1,
            multiplied_layers=[], attr_threshold=1., base_multiplier=1.5,
            output_hidden_states=True, output_device='cpu'):
    L = len(model.transformer.h)
    head_mask, mlp_mask, attn_weights = [fill_list(mask, L, attribute_layer)
        for mask in [head_mask, mlp_mask, attn_weights]]
    from_layer = attribute_layer if hidden_states is not None else None

    self = model.transformer
    (hidden_states, inputs_embeds, position_embeds) = embed_forward(self, inputs) \
        if from_layer is None else (hidden_states, None, None)
    inputs_embeds = inputs_embeds.to(output_device)
    all_hidden_states, intermediates, attn_outputs, mlp_outputs = (), (), (), ()
    attn_fwd_outputs = []
    for i, b in enumerate(self.h):
        if from_layer is not None and i < from_layer: continue
        if i == detach_layer: hidden_states = hidden_states.detach()
        if output_hidden_states: all_hidden_states += (to(hidden_states, output_device),)
        h = b.ln_1(hidden_states)
        attn_output, *attn_fwd_output = maybe_composed_attn_forward(model, b, h, h, h, by_head=by_head,
                            head_mask=head_mask[i], attn_weights=attn_weights[i], ranges=ranges)
        attn_fwd_output = [to(o, output_device) for o in attn_fwd_output]
        attn_fwd_outputs.append(attn_fwd_output)
        
        if i == special_head[0]:
            head_output = list(rearrange(head_output, 'b n i e -> n b i e'))  # nbie->n*bie
            head_output[special_head[1]] = head_output[special_head[1]] * special_head_multiplier
            special_head_outputs = torch.cat([special_head_outputs, head_output[special_head[1]]]) \
                if special_head_outputs is not None else head_output[special_head[1]] # sie, s in [1, 2] for 8-1, 12-10
            attn_output = sum(head_output)
        if i in multiplied_layers:
            multiplier = get_multiplier(b, hidden_states, aw, head_input, special_head_outputs, labels, i,
                attr_threshold=attr_threshold, base_multiplier=base_multiplier)
            attn_output = torch.einsum('bnie,ni->bie', head_output, multiplier)
        parallel_attn_mlp = my_isinstance(b, (GPTJBlock, GPTNeoXLayer))
        if not parallel_attn_mlp: hidden_states = hidden_states + attn_output
        mlp_output, _ = mlp_forward(b, hidden_states, output_intermediate=True)
        if mlp_mask[i] is not None: mlp_output = einsum('bie,bi->bie', mlp_output, mlp_mask[i])
        # intermediates += (to(intermediate, output_device),)
        hidden_states = (hidden_states + mlp_output) if not parallel_attn_mlp else \
            (attn_output + mlp_output + hidden_states)  # order matters!
        # attn_output = to(attn_output, output_device); attn_outputs += (attn_output,)
        mlp_output = to(mlp_output, output_device); mlp_outputs += (mlp_output,)
    all_attentions, head_inputs, head_outputs = zip(*attn_fwd_outputs)
    if output_hidden_states: all_hidden_states += (to(hidden_states, output_device),)
    hidden_states = self.ln_f(hidden_states)
    if output_hidden_states: all_hidden_states += (to(hidden_states, output_device),)

    logits = model.lm_head(hidden_states)
    loss = compute_loss(logits, labels, reduction=loss_reduction) if labels is not None else None
    logits, loss = to(logits, output_device), to(loss, output_device)

    return Outputs(
        inputs_embeds=inputs_embeds, position_embeds=position_embeds,
        attn_outputs=attn_outputs, head_inputs=head_inputs, head_outputs=head_outputs, 
        intermediates=intermediates, mlp_outputs=mlp_outputs,
        hidden_states=all_hidden_states, attentions=all_attentions, 
        logits=logits, loss=loss,
    )

def forward(model, inputs, labels=None, loss_reduction=None, by_head=False, 
            head_mask=None, mlp_mask=None, attn_weights=None,
            attribute_layer=None, hidden_states=None, detach_layer=None,
            relating_heads=[], intermediary_heads=[], predicting_heads=[],
            intm_head_multipliers=[], hq_multiplier=0., multipliers=[0, 0],
            self_attr_threshold=5., device='cpu', to_gpu_layer=None, mlp_to_gpu=False, exit_layer=None
            ):
    if device != 'cpu' and to_gpu_layer is None: to_gpu_layer = intermediary_heads[0][0]
    blocks = model.transformer.h; L = len(blocks); H = blocks[0].attn.num_heads
    if head_mask is None: head_mask = fill_list(head_mask, L, attribute_layer)
    mlp_mask = fill_list(mlp_mask, L, attribute_layer)
    attn_weights = fill_list(attn_weights, L, attribute_layer)
    from_layer = attribute_layer if hidden_states is not None else None
    if intm_head_multipliers is None: intm_head_multipliers = [1.] * len(intermediary_heads)

    self = model.transformer
    (hidden_states, inputs_embeds, position_embeds) = embed_forward(self, inputs) \
        if from_layer is None else (hidden_states, None, None)
    all_hidden_states, intermediates, mlp_outputs = (), (), ()
    attn_fwd_outputs = []; all_attentions = []
    

    label_mask = (labels != -100) if labels is not None else \
        ((inputs != -100) if isinstance(inputs, torch.Tensor) else (inputs.input_ids != -100))
    all_attentions2, cat_hqs = {}, {}
    self_attrs = torch.zeros(L, H, len(intermediary_heads) + 1, 
        len(relating_heads) + 2 if len(relating_heads) > 0 else 1, label_mask.sum()) # lnmri
    relating_head_outputs, intm_head_outputs = None, None

    if True or len(relating_heads) > 0:
        hidden_states_attn = torch.zeros_like(hidden_states)
        hidden_states_mlp = hidden_states  # init hidden_states_mlp with embed_output
    for i, b in enumerate(self.h):
        if from_layer is not None and i < from_layer: continue
        if i == detach_layer: hidden_states = hidden_states.detach()
        all_hidden_states += (hidden_states,)

        rel_heads = [head[1] for head in relating_heads if head[0] == i]
        intm_heads = [(head, mul) for head, mul in zip(intermediary_heads, intm_head_multipliers) if head[0] == i]
        # pred_heads = [head[1] for head in predicting_heads if head[0] == i]
        pred_head_mask = torch.LongTensor([1 if (i, h) in predicting_heads else 0 for h in range(H)])#.to(device)
        _by_head = by_head or len(rel_heads) > 0 or len(intm_heads) > 0 or pred_head_mask.sum() > 0
        multiply = pred_head_mask.sum() > 0 #and intm_head_outputs is not None

        _device = 'cpu'
        to_gpu = to_gpu_layer is not None and i >= to_gpu_layer and device != 'cpu'
        if to_gpu: block2gpu(b, mlp_to_gpu); _device = device
        hidden_states0 = hidden_states.to(_device); hidden_states_attn = hidden_states_attn.to(_device)
        if mlp_to_gpu: hidden_states = hidden_states0; hidden_states_mlp = hidden_states_mlp.to(_device)
        (hs, ln_mean, ln_std) = custom_ln(b.ln_1, hidden_states0) \
            if len(intm_heads) > 0 or multiply else (b.ln_1(hidden_states), None, None)
        hq = hk = hv = hs # b.ln_1(hidden_states0)

        attn_fwd_output = attn_forward(b, hq, hk, hv, by_head=_by_head, compute_head_input=True,
            head_mask=head_mask[i], attn_weights=attn_weights[i])
        if device == 'cpu': attn_fwd_outputs.append(attn_fwd_output)
        attn_output, aw, head_input, head_output = attn_fwd_output
        all_attentions.append(aw.to('cpu'))
        if multiply:
            cat_hs = intm_head_outputs # m(r+2)ie or m1ie, m <= n_intm_heads
            intm_hs = cat_hs.sum(dim=(0, 1))  # ie
            cat_hs = pad(cat_hs, dim=0, to_len=len(intermediary_heads) + 1)
            if len(relating_heads) == 0: #assert cat_hs.size(1) == 1, str(cat_hs.size(1))
                cat_hs[-1, -1] = hidden_states0[0] - intm_hs
            else:
                cat_hs[-1, -2] = hidden_states_attn[0] - intm_hs
                cat_hs[-1, -1] = hidden_states_mlp[0]
            # print(i, 'cat_hs', equal(hidden_states0[0], cat_hs.sum((0, 1))))
            attn_labels = torch.einsum('bnij,bnj->bnij', aw, head_input.norm(dim=-1)) # bnje->bnj
            a0 = a = get_self_attr(b, hidden_states0, cat_hs, attn_labels, device=_device)  # ni  n(m+1)(r+1)i
            self_attrs[i] = a[..., label_mask.to(_device)[0]].to('cpu')
            cat_hq = custom_ln(b.ln_1, cat_hs, ln_mean, ln_std, bias=False)[0]
            cat_hq[-1, -1:] += (- ln_mean / ln_std * b.ln_1.weight + b.ln_1.bias) # add ln bias back
            # print(i, 'cat_hq', equal(b.ln_1(hidden_states0[0]), cat_hq.sum((0, 1))))
            cat_hqs[i] = cat_hq.to('cpu')

            a = a[:, :-1, :max(1, a.size(2) - 2), :].amax((-3, -2))  # n(m+1)(r+1)i->ni or n(m+1)1i->ni
            a = ((a > self_attr_threshold) * label_mask.to(_device) * pred_head_mask.unsqueeze(1).to(_device)).float()
            if a.sum() > 0:
                if multipliers[1] != 0:
                    hq = b.ln_1(intm_hs * hq_multiplier + hidden_states0)
                    # a0[:, -1] = 0
                    # hq = torch.einsum('nmri,mrie->nie', (a0 > self_attr_threshold).float(), cat_hs).unsqueeze(0) # bnie
                    # hq = b.ln_1(hq * hq_multiplier + hidden_states0)
                    # wq = rearrange(b.attn.q_proj.weight, '(n d) e -> n e d', n=b.attn.num_heads)
                    # hq = hq @ wq  # bnie,ned->bnid, same as in cat_attn_forward
                    _, aw2, _, head_output2 = attn_forward(b, hq, hk, hv, by_head=True, compute_head_input=False)
                    if True or device == 'cpu': all_attentions2[i] = aw2.to('cpu')
                attn_output = torch.einsum('bnie,ni->bie', head_output, 1 - a)
                if multipliers[0] != 0: attn_output += torch.einsum('bnie,ni->bie', head_output, a) * multipliers[0]
                if multipliers[1] != 0: attn_output += torch.einsum('bnie,ni->bie', head_output2, a) * multipliers[1]
        for intm_head, mul in intm_heads:
            cat_intm_head_out = intm_head_output = head_output[:, intm_head[1]] # 1nie->1ie
            if len(relating_heads) > 0:
                cat_hs = relating_head_outputs.to(_device) # torch.cat(relating_head_outputs) # rje, r <= n_rel_heads
                cat_hs = torch.cat([hidden_states_attn - cat_hs.sum(0), hidden_states_mlp, cat_hs]) # (2+r)je
                cat_hs = custom_ln(b.ln_1, cat_hs, ln_mean, ln_std, bias=False)[0]
                # disable bias during cat ln forward and then add it back to hidden_states_mlp to pass equal test. Tricky!
                cat_hs[1] = cat_hs[1] - ln_mean / ln_std * b.ln_1.weight + b.ln_1.bias # bie, b=1
                wv, wo = get_head_weights(model, *intm_head)[2:]
                cat_intm_head_out = aw[:, intm_head[1]] @ (cat_hs @ wv) @ wo # 1ij,(rje,ed),de->rie
                cat_intm_head_out = pad(cat_intm_head_out, dim=0, to_len=2 + len(relating_heads))  # pad 2+r->2+n_rel_heads
                cat_intm_head_out = cat_intm_head_out.roll(-2, dims=0)  # (2+r)ie->(r+2)ie
                # print(i, 'intm_head 2', equal(intm_head_output[0], cat_intm_head_out.sum(0)))
            cat_intm_head_out = cat_intm_head_out * mul
            intm_head_outputs = cat_intm_head_out.unsqueeze(0) if intm_head_outputs is None \
                else torch.cat([intm_head_outputs, cat_intm_head_out.unsqueeze(0)])  # mrie or m1ie
            if mul != 1:
                head_output[:, intm_head[1]] = cat_intm_head_out.sum(0, keepdim=True) # rie->1ie
                attn_output = head_output.sum(1) # bnie->bie
        if len(rel_heads) > 0:
            relating_head_outputs = head_output[0, rel_heads].to('cpu') if relating_head_outputs is None \
                else torch.cat([relating_head_outputs, head_output[0, rel_heads].to('cpu')])  # rje
        if to_gpu and not mlp_to_gpu: block2cpu(b, mlp_to_gpu); attn_output = attn_output.to('cpu')

        if not my_isinstance(b, GPTJBlock): hidden_states = hidden_states + attn_output
        mlp_output, intermediate = mlp_forward(b, hidden_states, output_intermediate=True)
        if mlp_mask[i] is not None: mlp_output = einsum('bie,bi->bie', mlp_output, mlp_mask[i])
        if device == 'cpu':
            intermediates += (intermediate,)
            mlp_outputs += (mlp_output,)
        if my_isinstance(b, GPTJBlock): hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + mlp_output
        if len(relating_heads) > 0:
            hidden_states_attn += attn_output; hidden_states_mlp += mlp_output
        if to_gpu and mlp_to_gpu: block2cpu(b, mlp_to_gpu); hidden_states = hidden_states.to('cpu')
        if i == exit_layer: break

    (attn_outputs, _, head_inputs, head_outputs) = zip(*attn_fwd_outputs) \
        if device == 'cpu' else (None, None, None, None)  # all_attentions are skipped
    all_hidden_states += (hidden_states,) # both before and after ln_f
    hidden_states = self.ln_f(hidden_states)
    all_hidden_states += (hidden_states,)

    logits = model.lm_head(hidden_states)
    loss = compute_loss(logits, labels, reduction=loss_reduction) if labels is not None else None
    o = Outputs(
        inputs_embeds=inputs_embeds, position_embeds=position_embeds,
        attn_outputs=attn_outputs, head_inputs=head_inputs, head_outputs=head_outputs, 
        intermediates=intermediates, mlp_outputs=mlp_outputs,
        hidden_states=all_hidden_states, attentions=all_attentions, 
        logits=logits, loss=loss)
    if intm_head_outputs is not None:
        o.intm_head_outputs = intm_head_outputs.to('cpu')
    o.self_attrs = self_attrs # torch.stack(self_attrs)  # l*nmri->lnmri
    o.all_attentions2 = all_attentions2; o.cat_hqs = cat_hqs
    return o

def get_argmax_labels(model, hidden_states, labels, logits=None):
    if logits is None: logits = model.lm_head(model.transformer.ln_f(hidden_states))
    argmax_labels = labels.clone()
    argmax_labels[labels != -100] = logits.argmax(-1)[labels != -100]
    return argmax_labels

def get_prob_dist(d, topk=5, digits=3):
    return {k: round(math.exp(v), digits) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk]}

def show_predictions(tokenizer, example_strs, bos_indices, eos_indices, answers, 
        logits=None, labels=None, candidates=None, mask_logits_fn=None,
        k_shot=3, topk=5, loss_reduction='mean', sep='\t', verbose=True):
    use_openai_api = hasattr(logits, 'token_logprobs')  # isinstance(model, types.FunctionType)
    if use_openai_api: ans_nlls = []
    if not use_openai_api and mask_logits_fn is not None: logits = mask_logits_fn(logits)
    
    assert len(bos_indices) == len(example_strs), '%d != %d' % (len(bos_indices), len(example_strs))
    top1_corrects, answer_probs, candidate_probs = [], [], []
    convert_fn = tokenizer.convert_ids_to_tokens if True else partial(convert_ids_to_tokens, tokenizer=tokenizer)
    for i, (example_str, bos_i, eos_i, ans_ids) in enumerate(zip(example_strs, bos_indices, eos_indices, answers)):
        # eos_i = bos_i + 2  # show only the first answer token
        if use_openai_api:
            ans_prob_dist = [get_prob_dist(d, topk=topk) for d in logits.top_logprobs[bos_i + 1: eos_i]]
            ans_probs = [round(math.exp(lp), 3) for lp in logits.token_logprobs[bos_i + 1: eos_i]]
            if i >= k_shot: ans_nlls += [-lp for lp in logits.token_logprobs[bos_i + 1: eos_i]]
        else:
            ans_prob_dist = logits[0, bos_i: eos_i - 1].softmax(-1)
            ans_probs = numpy(ans_prob_dist[torch.arange(ans_prob_dist.size(0)), ans_ids], decimals=3)
        ans_tokens = convert_fn(ans_ids)
        for ans_id, ans_token, ans_prob, dist in zip(ans_ids, ans_tokens, ans_probs, ans_prob_dist):
            top1_correct = max(dist.items(), key=lambda x: x[1])[0] == ans_token.replace('Ġ', ' ') \
                if use_openai_api else (dist.argmax() == ans_id).item()
            top1_corrects.append(top1_correct)
            answer_probs.append(ans_prob)
            if candidates is not None:
                candidate_probs.append([dist[cand].item() for cand in candidates[i]] if not use_openai_api else
                    [dist.get(cand, 0.) for cand in [t.replace('Ġ', ' ') for t in convert_fn(candidates[i])]])
            if verbose: 
                print(('*' if top1_correct else ' ') + ans_token, ans_prob, dist if use_openai_api 
                    else show_topk(*dist.topk(topk), indices_fn=convert_fn), sep, example_str) 
    if use_openai_api:
        loss = (ans_nlls if loss_reduction == 'none' else sum(ans_nlls) / len(ans_nlls))
    else:
        loss = compute_loss(logits, labels, reduction=loss_reduction)
        loss = loss.item() if loss_reduction == 'mean' else loss[labels != -100].tolist()  # 'none'
    return loss, top1_corrects, answer_probs, candidate_probs

def predict(model, tokenizer, text, examples, k_shot=3, bos_token=' ->', eos_token=None, #'Ċ',
            custom_forward=True, verbose=True):
    input_ids, labels, ranges, *args = make_data_tuple( # args = [example_strs, bos_indices, eos_indices, answers]
        text, examples, tokenizer, k_shot=k_shot, bos_token=bos_token, eos_token=eos_token)
    candidates, answer_indices = None, None
    cxt, query, cands, *_ = examples[0]
    if cands is not None:
        candidates = [[tokenizer.encode(' ' + token)[0] for token in cands[-1]] for cxt, query, cands, *_ in examples]
        answer_indices = [get_answer_index(e) for e in examples]
    with torch.no_grad():
        o = forward0(model, input_ids.to(model.device), labels=labels.to(model.device), by_head=['head_input'], ranges=ranges) \
            if isinstance(model, nn.Module) and custom_forward else model(input_ids.to(getattr(model, 'device', 'cpu')))
        logits = o.logits
        if isinstance(logits, torch.Tensor): logits = logits.to('cpu').float()# softmax on cpu needs float32
    loss, top1_corrects, answer_probs, candidate_probs = show_predictions(
        tokenizer, *args, logits=logits, labels=labels, loss_reduction='mean',
        candidates=candidates, k_shot=k_shot, topk=3, verbose=verbose)
        
    if verbose == 1:
        print(loss, np.array(top1_corrects[k_shot:]).mean())
        f, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 2.4), sharex=True)
        x = [i + 0.5 for i in range(len(examples))] # to align with sns.heatmap
        _ = ax0.bar(x, top1_corrects, width=0.9, alpha=0.5);
        _ = ax0.plot(x, answer_probs, color='r');
        if answer_indices is not None:
            label_probs = F.one_hot(torch.LongTensor(answer_indices))
            _ = sns.heatmap(torch.cat([label_probs, torch.Tensor(candidate_probs)], dim=1).T, cbar=False, ax=ax1);
        plt.show()

    return [text, input_ids, labels, ranges] + args + [o,], \
        (loss, top1_corrects[k_shot:], answer_indices, answer_probs, candidate_probs)

def abbreviate_attn_pattern(attn_pattern): return attn_pattern.replace('bos', 'B').replace('query', 'Q').replace('ans', 'A')
def restore_attn_pattern(attn_pattern): return attn_pattern.replace('B', 'bos').replace('Q', 'query').replace('A', 'ans')

def attn_pattern2labels(ranges, attn_pattern, attn_size, k_shot=None, normalize=True):
    attn_pattern = restore_attn_pattern(attn_pattern)
    q, k = attn_pattern.split('->')  # e.g. 'bos->ans0'
    ranges_q = ranges[k_shot:] if k_shot is not None and not ('ans' in q and 'ans0' in k) else ranges
    attn_labels = torch.zeros(*attn_size)
    def get_slice(r, name):
        if name == '*': b, e = 0, attn_size[0]
        elif name.startswith('['): b, _ = getattr(r, name[1:]); e = b + 1
        elif name.endswith(']'): _, e = getattr(r, name[:-1]); b = e - 1
        elif name.endswith('+'): _, e = getattr(r, name[:-1]); b, e = e, e + 1
        else: b, e = getattr(r, name)
        return slice(b, e) if not isinstance(b, Iterable) else [slice(_b, _e) for _b, _e in zip(b, e)]
    if 'bos' in q and 'ans' in k and 'ans0' not in k:  # pattern for intermediary heads
        for rq in ranges_q:
            for rk in ranges:
                attn_labels[get_slice(rq, q), get_slice(rk, k)] = 1
    elif 'candidates' in k:
        for r in ranges_q:
            for s in get_slice(r, k):
                attn_labels[get_slice(r, q), s] = 1
    elif '<s>' in k:
        for r in ranges_q:
            attn_labels[get_slice(r, q), 0] = 1  # <s>
            if k == '~<s>': attn_labels[get_slice(r, q)] = 1 - attn_labels[get_slice(r, q)]
    else:
        for r in ranges_q: attn_labels[get_slice(r, q), get_slice(r, k)] = 1
    attn_labels = attn_labels.tril()
    if any(s in k for s in ['candidates', '<s>', 'example']): normalize = False  # used as attn_mask
    return attn_labels / ((attn_labels.sum(1, keepdim=True) + 1e-9) if normalize else 1)

def get_head_matching_scores(data_tuple, attn_pattern, k_shot=3, layer=None):
    text, input_ids, labels, ranges, *args, o = data_tuple
    attn_labels = attn_pattern2labels(ranges, attn_pattern, o.attentions[0].size()[-2:], k_shot=k_shot)
    attentions = list_get(o.attentions, layer, reduce_fn=torch.cat)  # lhij or hij
    matching_scores = ((attentions + 1e-9).log() * attn_labels).sum((-2, -1)) / attn_labels.sum()  # mean log-likelyhood, lhij->lh
    return matching_scores

# def mask_select(a, mask=None):  # a: attn_output0, bnid, mask: bi
#     bsz = a.size(0)
#     # mask = mask[None, :, :].expand(num_heads, -1, -1) # bi->nbi
#     a = rearrange(a, 'b n i d -> b i n d')[mask]  # bind[bi]->(bk)nd
#     a = rearrange(a, '(b k) n d -> b n k d', b=bsz)
#     return a

# def mask_expand(a, mask=None):  # a: head_output, bnke, mask: bi
#     out = torch.zeros(a.size(0), mask.size(1), a.size(1), a.size(3))  # bine
#     out[mask] = rearrange(a, 'b n k e -> (b k) n e')
#     return rearrange(out, 'b i n e -> b n i e')

# def head_select(a, heads=None):  # a: cat value, blnid
#     return torch.stack([a[:, l, h] for l, h in heads], dim=1)  # [blnid->bid]*k = bkid

# def head_expand(a, heads=None, L=None, H=None): # a: head_input, bkie
#     out = torch.zeros(a.size(0), L, H, a.size(-2), a.size(-1))  # blnie
#     for i, (l, h) in enumerate(heads): out[:, l, h] = a[:, i]
#     return out

# def get_head_io(head_io, w_o, num_heads=None): # head_input/output
#     w_o = rearrange(w_o, 'e (n d) -> n d e', n=num_heads)
#     return head_io @ w_o.to(head_io.device)

# def postprocess_output(o, select_fn, expand_fn):
#     if isinstance(o, tuple):
#         f, o, w = o
#         return expand_fn(f(select_fn(o), w))
#     assert isinstance(o, torch.Tensor), str(type(o))
#     return o

# def postprocess_head_output(head_output, mask):  # head_output: 1nid
#     select_fn, expand_fn = partial(mask_select, mask=mask), partial(mask_expand, mask=mask)
#     return postprocess_output(head_output, select_fn, expand_fn)

# def postprocess_head_inputs(head_inputs, heads):  # head_inputs: 1lnid
#     select_fn = partial(head_select, heads=heads)
#     expand_fn = partial(head_expand, heads=heads, L=head_inputs.size(1), H=head_inputs.size(2))
#     return postprocess_output(head_inputs, select_fn, expand_fn)

def sum_forward(model, outputs, labels=None, loss_reduction='per_example_mean', # attr_heads=None,
        embed_mask=None, mlp_mask=None, head_mask=None, neuron_mask=None, attn_weights=None, 
        reduce_fn=sum, truncate_layers=False, scaled=True, reshape=False, output_layer=None):
    embed_output = outputs.hidden_states[0]
    if embed_mask is not None:
        embed_output = einsum('bie,bi->bie', embed_output, embed_mask) # i=1 for embed_mask

    if output_layer is None: _l = len(outputs.mlp_outputs)
    else: _l = max(output_layer) if isinstance(output_layer, Iterable) else output_layer

    if reduce_fn == torch.cat and head_mask is None and attn_weights is not None:
        head_mask = einops.reduce(attn_weights, '1 l n i j -> 1 l n i', 'sum')
        attn_weights = None
    if head_mask is not None:
        # mask = labels != -100
        # head_outputs = [postprocess_head_output(o, mask) for o in outputs.head_outputs] # 1nid->1nie
        head_outputs = rearrange(list(outputs.head_outputs), 'l 1 n i e -> 1 l n i e')
        head_outputs = einsum('blnie,bln->blnie', head_outputs[:,:_l], head_mask[:,:_l])  # blni - i = bln
    elif neuron_mask is not None:
        head_outputs = einsum('blnie,blnie->blnie', neuron_mask[:,:_l], head_outputs[:,:_l])
    if attn_weights is not None:
        head_inputs = rearrange(list(outputs.head_inputs), 'l 1 n i e -> 1 l n i e')
        # head_inputs = postprocess_head_inputs(head_inputs, attr_heads)  # 1lnid->1lnie
        head_outputs = einsum('blnij,blnje->blnie', attn_weights[:,:_l], head_inputs[:,:_l])

    mlp_outputs = rearrange(list(outputs.mlp_outputs), 'l 1 i e -> 1 l i e')
    if mlp_mask is not None:
        mlp_outputs = einsum('blie,bl->blie', mlp_outputs[:,:_l], mlp_mask[:,:_l])  # bli - i = bl
    if reshape:  # for head amplication, obsolete
        assert reduce_fn == torch.cat
        L = (torch.einsum('bli->l', mlp_mask) > 0).sum().item() \
            if mlp_mask is not None and truncate_layers else len(outputs.mlp_outputs)
        attn_outputs = rearrange(head_outputs[:, :L], '1 l n i e -> l n i e')
        mlp_outputs = rearrange(mlp_outputs[:, :L], '1 l i e -> l 1 i e')
        padded_embed_output = embed_output.new(*((L,) + embed_output.size())).zero_() # l1ie
        padded_embed_output[0] = embed_output
        output = torch.cat([attn_outputs, mlp_outputs, padded_embed_output], dim=1) # lnie,l1ie,l1ie->l(n+2)ie
    elif output_layer is not None:  # default
        assert reduce_fn == sum
        if isinstance(output_layer, Iterable):
            output = [reduce_fn([embed_output,
                                torch.einsum('blnie->bie', head_outputs[:, :l]), 
                                torch.einsum('blie->bie', mlp_outputs[:, :l])])
                    for l in output_layer]
        else:
            attn_outputs = torch.einsum('blnie->bie', head_outputs[:, :output_layer])
            mlp_outputs = torch.einsum('blie->bie', mlp_outputs[:, :output_layer])
            output = reduce_fn([embed_output, attn_outputs, mlp_outputs]) # bie for sum
    else:  # for compatibility
        if reduce_fn == sum:
            attn_outputs = torch.einsum('blnie->bie', head_outputs)
            mlp_outputs = torch.einsum('blie->bie', mlp_outputs)
        elif reduce_fn == torch.cat:
            L = (torch.einsum('bli->l', mlp_mask) > 0).sum().item() \
                if mlp_mask is not None and truncate_layers else len(outputs.mlp_outputs)
            attn_outputs = rearrange(head_outputs[:, :L], '1 l n i e -> (l n) i e')
            mlp_outputs = rearrange(mlp_outputs[:, :L], '1 l i e -> l i e')
        output = reduce_fn([embed_output, attn_outputs, mlp_outputs]) # bie for sum, (1+(ln)+l)ie for cat

    logits, loss = None, None
    if labels is not None:
        ln_output = scaled_ln(model.transformer.ln_f, output, scaled=scaled)
        logits = model.lm_head(ln_output)
        loss = compute_loss(logits, labels, reduction=loss_reduction)
    return Outputs(hidden_states=(output,), logits=logits, loss=loss)

def scaled_input(input, num_points, baseline=None, requires_grad=True):
    assert input.size(0) == 1
    if baseline is None: baseline = torch.zeros_like(input)
    if False and num_points == 3:
        step = (input - baseline) / 10
        res = torch.cat([baseline + step * i for i in [0, 5, 10]], dim=0) #lxy
        # res = torch.cat([baseline + step * i for i in [0, ]], dim=0)
    else:
        step = (input - baseline) / num_points
        res = torch.cat([baseline + step * i for i in range(num_points + 1)], dim=0)
        # res = torch.cat([baseline + step * (i + 1) for i in range(num_points)], dim=0)  # XD
        # alphas = list(0.5 * (1 + np.polynomial.legendre.leggauss(num_points)[0])) # copied from captum
        # res = torch.cat([baseline + alpha * (input - baseline) for alpha in alphas], dim=0)
    if requires_grad: res.requires_grad_(True)
    return res #, step

def compose_forward_fns(forward_fns, **kwargs):
    def forward(model, outputs):
        for fn in forward_fns:
            if my_isinstance(outputs, Outputs) and len(outputs.hidden_states) > 0:
                outputs = outputs.hidden_states[-1]
            outputs = fn(model, outputs, **kwargs)
        _fn2str = partial(fn2str, excluded_keys=['outputs', 'ranges'])
        assert isinstance(outputs.hidden_states, tuple), f'{type(outputs.hidden_states)} {_fn2str(fn)}'
        assert len(outputs.hidden_states) in [0, 1], f'{len(outputs.hidden_states)} != 1 {_fn2str(fn)}'
        return outputs.hidden_states[0] if len(outputs.hidden_states) == 1 else None, \
            -outputs.loss if outputs.loss is not None else None, \
            outputs.logits
    return forward

def _attribute(forward_fn, model, x, post_forward_fn=[], num_points=10, batch_size=None):
    if batch_size is None: batch_size = num_points + 1
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=True)
    scaled_x, grad = {}, {}
    with torch.enable_grad():
        for key in x:
            scaled_x[key] = scaled_input(x[key], num_points)
            grad[key] = torch.zeros_like(x[key])
        ys = []
        for i in range(0, num_points, batch_size):
            scaled_x_ = OrderedDict({key: scaled_x[key][i: i + batch_size] for key in x})
            o = forward_fn(model, **scaled_x_)
            y, logits = post_forward_fn(model, o); ys.append(y)
            grad_ = torch.autograd.grad(y.flatten().unbind(), list(scaled_x_.values()))
            for j, key in enumerate(x.keys()):
                grad[key] += grad_[j].sum(dim=0, keepdim=True)
    attr = {key: (grad[key] / num_points * x[key]).squeeze(0) for key in x}
    return attr, torch.cat(ys), logits

def attribute(forward_fn, model, x, post_forward_fn=[], num_points=7, batch_size=8, forward_only=False):
    if batch_size is None: batch_size = num_points + 1
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=True)
    grad_keys = list(x.keys())  # [key for key in x if key != 'head_mask' or 'attn_weights' not in x]
    scaled_x, grad = {}, {}
    with torch.enable_grad() if not forward_only else torch.no_grad():
        for key in x:
            scaled_x[key] = scaled_input(x[key], num_points)
            if True or key in grad_keys: grad[key] = torch.zeros_like(x[key])
        ys, hidden_states = [], []  # negative loss and hidden_states
        for i in range(0, num_points + 1, batch_size):
            scaled_x_ = OrderedDict({key: scaled_x[key][i: i + batch_size] for key in x})
            if True: #with Timer('sum_forward'):
                o = forward_fn(model, **scaled_x_)
            hs, y, logits = post_forward_fn(model, o); ys.append(y); hidden_states.append(hs)
            if forward_only: continue
            if True: #with Timer('grad'):
                grad_ = torch.autograd.grad(y.flatten().unbind(), list(scaled_x.values()))
                # [v for k, v in scaled_x.items() if k in grad_keys])
            for j, key in enumerate(grad_keys):
                grad[key] += grad_[j].sum(dim=0, keepdim=True)
    hidden_states, ys = [torch.cat(ts) if ts[0] is not None else None for ts in [hidden_states, ys]]
    if forward_only: return None, hidden_states, ys, logits
    attr = {key: (grad[key] / num_points * x[key]).squeeze(0) for key in grad_keys}

    attn_attr = attr.get('attn_weights')
    neuron_attr = attr.get('neuron_mask')
    head_attr = torch.einsum('lnij->ln', attn_attr) if attn_attr is not None \
        else attr.get('head_mask')#.sum(-1)  # lni->ln  #torch.einsum('lnie->ln', neuron_attr)
    # head_attr = attr.get('head_mask')#.sum(-1)  # lni->ln
    mlp_attr = attr.get('mlp_mask')#.sum(-1, keepdim=False) # li->l
    emb_attr = attr.get('embed_mask')#.sum(-1, keepdim=True) # i->1
    attr = Attributions(attn=attn_attr, head=head_attr, neuron=neuron_attr, mlp=mlp_attr, embed=emb_attr)
    return attr, hidden_states, ys, logits

def attribute2(forward_fn, model, x, post_forward_fn):
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=False)
    with torch.no_grad():
        o = forward_fn(model, **x)
        y, logits = post_forward_fn(model, o)
    assert y.ndim == 1
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    # assert (y.size(0) - 1) % (H + 1) == 0
    # to_layer = (y.size(0) - 1) // (H + 1) # by solving equation b = 1 + ln + l
    # assert to_layer == (torch.einsum('blni->l', x['head_mask']) > 0).sum().item()
    embed_attr = y[:1].view(1)
    head_attr = y[1: 1 + L * H].view(L, H)  # ln
    mlp_attr = y[1 + L * H:]#.view(L, 1)  # l1
    return Attributions(embed=embed_attr, head=head_attr, mlp=mlp_attr)

def attribute_step(data_tuple, model, node, attribute_k=False):
    L = len(model.transformer.h)
    text, input_ids, labels, ranges, *_, o = data_tuple
    fns = path2fns(node, partial(node2fn, outputs=o, ranges=ranges, labels=labels))
    (output_layer, _labels) = (node.data.layer, None) if len(fns) > 0 else (L, labels)
    if _labels is not None and node.data.label_type == 'argmax_labels':  # for root
        _labels = get_argmax_labels(model, o.hidden_states[-2], _labels)
    to_layer = max(output_layer) if isinstance(output_layer, Iterable) else output_layer
    fwd_fn = partial(sum_forward, outputs=o, labels=_labels, output_layer=output_layer)
    keys = ['attn_weights', 'mlp_mask', 'embed_mask']
    x = OrderedDict((key, get_x(key, o, to_layer=to_layer)) for key in keys)
    attr, hs, ys, logits = attribute(fwd_fn, model, x, fns, num_points=3 if attribute_k else 7)
    # fwd_fn = partial(sum_forward, outputs=o, labels=_labels, reduce_fn=torch.cat, scaled=False)
    # attr2 = attr #attribute2(fwd_fn, model, x, fns)
    o.attn_attr[node.name] = attr.attn # associate non-averageable attn attr to current node. tricky
    del attr.attn  # avoid being reduced by reduce_objects
    return attr

def get_x(key, outputs, to_layer=None):
    L, H = len(outputs.attentions), outputs.attentions[0].size(1)
    qlen = outputs.attentions[0].size(1)
    # L dim removed when doing per-layer attribution
    if key == 'head_mask': x = torch.ones(1, L, H)#, qlen)
    elif key == 'neuron_mask': x = torch.ones(1, L, H, qlen, outputs.hidden_states[0].size(-1))
    elif key == 'mlp_mask': x = torch.ones(1, L)#, qlen)
    elif key == 'embed_mask': x = torch.ones(1, 1)#, qlen)
    elif key == 'attn_weights': x = rearrange(list(outputs.attentions), 'l 1 n i j -> 1 l n i j')
    if to_layer is not None and x.ndim >= 2 and x.size(1) == L: x[:, to_layer:] = 0
    return x

def plot_attrs(attrs, figsize=(4, 4), topk=None):
    fig, axs = plt.subplots(1, len(attrs), sharey=False, figsize=(figsize[0] * len(attrs), figsize[1]))
    if len(attrs) == 1: axs = [axs]
    titles, attrs = (attrs.keys(), attrs.values()) if isinstance(attrs, dict) else ([None] * len(attrs), attrs)
    for ax, a, title in zip(axs, attrs, titles):
        res = sns.heatmap(a, cbar=False, ax=ax)
        if title is not None: ax.set_title(title)
        _ = res.set_yticklabels(res.get_ymajorticklabels(), rotation=0)
        # res.tick_params(top=False, right=True, labeltop=False, labelright=True)
    plt.show()
    if topk is not None:
        for a in attrs: pprint(topk_md(a, topk))

def get_head_rank(head_attr, layer, head, topk=20):
    if head is not None:
        head2rank = {k: v for k, v in zip(zip(*topk_md(head_attr, topk)[:2]), range(topk))}
        return head2rank.get((layer, head), None)
    else: # mlp
        head2rank = {k: v for k, v in zip(zip(*topk_md(head_attr, topk)[:1]), range(topk))}
        return head2rank.get((layer,), None)

def data2str(data, head_only=False):
    step, topi, layer, head, label_type, attn_pattern, attribute_k, top_score = \
        data.step, data.topi, data.layer, data.head, data.label_type, data.attn_pattern, data.attribute_k, data.top_score
    def wrap(head): return head if head < 16 else 'm'
    s = ''
    if head is None:
        s += f'{layer}-m'
    elif not isinstance(layer, Iterable):
        s += f'{layer}-{wrap(head)}'
        if top_score is not None: s += f' {int(round(top_score * 100))}%'
        if not head_only and attn_pattern is not None: s += f' {abbreviate_attn_pattern(attn_pattern)}'
    else:
        s += ','.join([f'{l}-{wrap(h)}' for l, h in zip(layer, head)]) if top_score is None or len(layer) > 5 else \
            ','.join([f'{l}-{wrap(h)} {int(round(s * 100))}%' for l, h, s in zip(layer, head, top_score)])
    if head_only: return s
    head_str = s
    s = f'[{step}] '
    if topi is not None:
        s += f'@:{len(topi)} ' if isinstance(topi, Iterable) and list(topi) == list(range(topi[-1] + 1)) \
            else f'@{topi}'.replace(' ', '') + ' '
    s += head_str
    if label_type is not None and label_type != 'labels': s = s + ' ' + label_type.replace('_labels', '')
    if attribute_k: s = s + ' ' + 'attr_k'
    return s

def get_head_mlp_attr(data): return torch.cat([data.attr.head, data.attr.mlp.unsqueeze(-1)], dim=1)

def get_matched_head_attr(data):
    return data.attr.head * (reduce(torch.maximum, data.scores.values()).exp() if len(getattr(data, 'scores', {})) > 0 else 1)

def add_node(parent, layer=None, head=None, topi=None, head_attr_fn=None, label_type=None, attn_pattern=None, 
            step=None, attribute_k=False, top_score=None, dummy=False, force=False, verbose=True):
    if label_type: assert label_type in ['labels', 'argmax_labels'] or 'attn_labels' in label_type, label_type
    if parent is None:
        si = -1; node = Node('[-1] root' + (f' {label_type}' if label_type != 'labels' else ''))
        node.data = AttrData(step=si, label_type=label_type)
    else:
        if parent is not None and parent.data.attr is None and not force:
            print('parent has not been attributed yet, replace it instead of adding to it.')
            _id = id(parent); parent = parent.parent
            parent.children = [child for child in parent.children if id(child) != _id]
        if head_attr_fn is None: head_attr_fn = get_head_mlp_attr  # get_matched_head_attr
        head_attr = head_attr_fn(parent.data) if parent.data.attr is not None else None
        if layer is None or topi is not None:
            layer, head = topk_md(head_attr, 10, transpose=True)[topi][:2] if type(topi) == int \
                else np.array(topk_md(head_attr, 10)[:2])[:, topi] # list
        if head_attr is not None:
            if topi is None: topi = topi_md(head_attr, layer, head)
            if top_score is None:
                top_score = head_attr[layer, head] / head_attr.max()
                top_score = top_score.numpy() if isinstance(layer, Iterable) else top_score.item()
        si = parent.data.step
        if si == -1 and not dummy: assert label_type is not None
        # else: label_type = None
        if attn_pattern is None and label_type: attn_pattern, _ = get_attn_pattern_from_label_type(label_type)
        if step is None: step = si + 1
        data = AttrData(step=step, topi=topi, layer=layer, head=head, label_type=label_type, 
            attn_pattern=attn_pattern, attribute_k=attribute_k, top_score=top_score, dummy=dummy)
        node = Node(data2str(data), parent, prepend=dummy); node.data = data
    if verbose and parent is not None: plot_tree(node)
    return node

def match_attn_pattern(data_tuples, node_name, layer, head, attn_patterns):
    o = data_tuples[0][-1]; H = o.attentions[0].size(1)
    if head == H: return None, 0.
    matching_scores = defaultdict(float)
    for text, input_ids, labels, ranges, *args, o in data_tuples:
        aw = o.attn_attr[node_name][layer, head]
        aw_size = o.attentions[0][0, 0].size()  # l*[bnij]->ij
        attn_labels = [attn_pattern2labels(ranges, ap, aw_size) for ap in attn_patterns]
        for ap, al in zip(attn_patterns, attn_labels):
            matching_scores[ap] += (al * aw).sum().item()
    attn_pattern, matching_score = max(matching_scores.items(), key=lambda x: x[1])
    return attn_pattern, matching_score

def attribute_tree(data_tuples, model, node, max_step, topk=10, threshold_scores=[1/3, 1/2]):
    o = data_tuples[0][-1]; H = o.attentions[0].size(1)
    if node.data.step > max_step: return
    if getattr(node.data, 'attr', None) is None:
        with Timer(f'In attribute_tree: expanding {node.name}'):
            node.data.attr = mr(attribute_step)(data_tuples, model, node)
        # node.data.scores = {ap: mr(get_head_matching_scores)(data_tuples, ap, k_shot=k_shot)
        #     for ap in attn_patterns_by_step.get(node.data.step, [])}

        head_attr = get_head_mlp_attr(node.data)
        layers, heads, scores = topk_md(head_attr, topk * 4)
        scores = np.array(scores) / scores[0]
        topi = min(topk, (scores >= threshold_scores[0]).sum().item())
        label_type = 'argmax_attn_labels' if node.data.label_type == 'labels' else None
        child = add_node(node, topi=list(range(topi)), top_score=scores[:topi], dummy=True, verbose=False)
        print('In attribute_tree: add', child.name)
        for i, (layer, head, score) in enumerate(zip(layers, heads, scores)):
            child = None
            for c in node.children:
                d = c.data
                if not isinstance(d.layer, Iterable) and (d.layer, d.head, d.label_type) == (layer, head, label_type):
                    child = c
                    d.topi, d.top_score = i, score
                    d.attn_pattern = match_attn_pattern(data_tuples, node.name, layer, head, all_attn_patterns)[0]
                    child.name = data2str(d)
                    break

            if i >= topk or head == H or score < threshold_scores[1] or child is not None or node.data.step == 1: continue
            attn_pattern, _ = match_attn_pattern(data_tuples, node.name, layer, head, all_attn_patterns)
            # step = node.data.step + 1 #if attn_pattern not in ['bos->bos'] else node.data.step
            # if step == 2: continue

            child = add_node(node, topi=i, label_type=label_type, attn_pattern=attn_pattern, top_score=score, verbose=False)
            print('In attribute_tree: add', child.name)
            
    for child in node.children:
        if not child.data.dummy:
            attribute_tree(data_tuples, model, child, max_step, topk=topk, threshold_scores=threshold_scores)

def node2key(node):
    if node.parent is None: return 'root'
    s = data2str(node.data, head_only=True)
    parent_s = data2str(node.parent.data, head_only=True) + '>' if node.parent.parent is not None else ''
    return parent_s + s

def map_nodes(nodes, root):
    nodes[node2key(root)] = root
    for child in root.children: map_nodes(nodes, child)
    
def remove_node(nodes, key):
    if key not in nodes: return
    node = nodes[key]

    for child in node.children:
        remove_node(nodes, data2str(child.data, head_only=True))
    
    ids = [i for i, child in enumerate(node.parent.children) if child == node]
    assert len(ids) == 1, str(len(ids))
    del node.parent.children[ids[0]]
    del nodes[key]
    return

def plot_tree(node):
    root = node
    while root.parent is not None: root = root.parent
    node.name = '*' + node.name
    if node.data.attr is None: node.name += '...'
    print_tree(root)
    node.name = node.name.replace('*', '').replace('...', '')

def show_result(result, node=None, node_name='node', topk=10):
    print('\n'.join(result.texts[-1].split('\n')[:3]))
    if node is None: node = getattr(result, node_name); plot_tree(node)
    n = node; path = []
    while n is not None: path.append(n); n = n.parent
    for n in path[::-1]:
        data = n.data
        if data.attr is not None:
            print(n.name)
            H = data.attr.head.size(1)
            aux_scores = {'(bos->bos)': mr(get_head_matching_scores)(result.data_tuples, 'bos->bos', k_shot=3)} \
                if data.step == -1 or maybe_mr(lambda x: x == H)(data.head) else {} # 1st step (i.e. pred head) or mlp step
            scores = dict(data.scores, **aux_scores)
            scores = {k: v.exp() for k, v in scores.items()}
            plot_attrs(dict(scores, head_attr=get_head_mlp_attr(data),
                matched_head_attr=get_matched_head_attr(data)), topk=topk)
    return node, result.data_tuples

def get_argmax_attn_labels(o, layer, head, labels=None):
    # attn_labels = torch.einsum('bnij,bnj->bnij', o.attentions[layer], o.head_inputs[layer].norm(dim=-1)) # bnje->bnj
    # return attn_labels[0, head]  # 1nij->ij
    attn_labels = torch.einsum('bij,bj->bij', o.attentions[layer][:, head], o.head_inputs[layer][:, head].norm(dim=-1))
    if labels is not None: attn_labels = torch.einsum('bij,bi->ij', attn_labels, (labels != -100).float()) # b=1
    return attn_labels

def node2fn(node, outputs=None, ranges=None, labels=None):
    hidden_states0 = None
    k_node = getattr(node, 'k_node', None)
    if k_node is not None:
        l = k_node.data.layer
        fns = path2fns(k_node, partial(node2fn, outputs=outputs, ranges=ranges), except_last=True)
        fwd_fn = partial(sum_forward, outputs=outputs, output_layer=l)
        keys = ['embed_mask', 'mlp_mask', 'attn_weights']
        to_layer = max(l) if isinstance(l, Iterable) else l
        x = OrderedDict((key, get_x(key, outputs, to_layer=to_layer)) for key in keys)
        _, hidden_states0, _, _ = attribute(fwd_fn, node.model, x, fns, num_points=1, forward_only=True)
        hidden_states0 = hidden_states0[-1:]  # bsz dim 2->1

    d = node.data
    return partial(mixed_forward, layer=d.layer, head=d.head, labels=labels, label_type=d.label_type,
        outputs=outputs, attn_attr=outputs.attn_attr[node.parent.name], hidden_states0=hidden_states0, 
        ranges=ranges, attribute_k=d.attribute_k, scaled=True)

def node2path(node, except_last=False):
    nodes = []
    while 'root' not in node.name:
        nodes.append(node)
        if node.data.label_type is None: node = node.parent
        else: break
    return nodes if not except_last else nodes[:-1]

def path2fns(node, node2fn, except_last=False):
    return [node2fn(n) for n in node2path(node, except_last=except_last)]

# losses2 = []
# sum_output = head_outputs * 0
# # hq, hk, hv = block.ln_1(sum_output), block.ln_1(o.hidden_states[layer[0]]), None
# # with torch.no_grad(): attn_logits = attn_forward(block, hq, hk, hv)[1]
# # logits = attn_logits[:, head[0]]  # bnij->bij
# # loss = -torch.einsum('bij->b', logits.log_softmax(-1) * attn_labels / attn_labels.sum())
# # print(-1, -1, 0, loss.item())
# # losses2.append(loss.item())
# for l, h, v in list(zip(*topk_md(head_attr2[:layer[0]], head_attr2.numel())))[:]:
#     sum_output = sum_output + o.head_outputs[l][:, h]
#     hq, hk, hv = block.ln_1(sum_output), block.ln_1(o.hidden_states[layer[0]]), None
#     with torch.no_grad(): attn_logits = attn_forward(block, hq, hk, hv)[1]
#     logits = attn_logits[:, head[0]]  # bnij->bij
#     loss = -torch.einsum('bij->b', logits.log_softmax(-1) * attn_labels / attn_labels.sum())
#     print(l, h, v, loss.item())
#     losses2.append(loss.item())


# task_name = 'find majority'  ##?
# task_name = 'find special kind'  ****
# task_name = 'is same' / 'is same kind'  ****
# task_name = 'find special easy' ## 6-2
# task_name = 'A B C -> B' ##
# task_name = 'set diff' ##?
# task_name = 'A(BC->B' ##  6-1
# task_name = 'ABC,AXC->X' ##?
# task_name = 'reverse set diff' ##, *failed only on first position, GPT-3 has this problem, too
# task_name = 'reverse set diff v2' ## A*C,ABC->B
# task_name = 'find next easy' ## ABCDEF,\nBC->D, 6+2

# from https://discuss.pytorch.org/t/get-top-k-indices-values-of-all-rows/89354
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    r = tuple(reversed(out))
    return torch.cat([i.unsqueeze(-1) for i in r], dim=-1).cpu().tolist() if type(index) in [torch.Tensor] else r

def h2topk(h, k=4, return_probs=True):
    if not hasattr(h2topk, 'ln') or h2topk.ln.normalized_shape[0] != h.size(-1):
        h2topk.ln = nn.LayerNorm(h.size(-1))
#     r = model.lm_head(h2topk.ln(h))
    r = model.lm_head(h)
    if return_probs: r = r.softmax(-1)
    return r.topk(k, dim=-1) if k > 0 else r

def globalize(tensor):
    if tensor.dim() == 4: return tensor  # global attention
    assert tensor.dim() == 5, str(tensor.dim())
    assert tensor.size(1) == 1, str(tensor.size(1))  # num_blocks
    seq_len = tensor.size(3)
    return tensor.squeeze(1)[:, :, :, -seq_len:]  # (bsz, num_blocks, H, seq_len, block_len) -> (bsz, H, seq_len, seq_len)

def append_tokens_to_positions(position_tensor):
    positions = numpy(position_tensor)
    return ['%d %s' % (p, tokens[p]) for p in positions]

def getdelattr(obj, name):
    r = getattr(obj, name, None)
    if hasattr(obj, name): delattr(obj, name)
    return r

def try_delattr(obj, name):
    if hasattr(obj, name): delattr(obj, name)

def get_attn_module(block):
    m = block.attn
    if hasattr(m, 'attention'): m = m.attention  # for gpt-neo
    return m


def heatmap(a, figsize=(20, 1), cbar=False):
    _ = plt.figure(figsize=figsize)
    _ = sns.heatmap(numpy(a, decimals=None), cbar=cbar)
    plt.show()
    
def plot(a, figsize=(20, 2)):
    _ = plt.figure(figsize=figsize)
    _ = plt.plot(numpy(a))
    
def plot_hidden(hidden, topk=4):
    if hidden.dim() == 3 and hidden.size(0) == 1: hidden = hidden.squeeze(0)
    assert hidden.dim() == 2, str(hidden.dim())
    heatmap(hidden, figsize=(20, 5))
    hidden_mean = hidden.mean(dim=0)
    _ = plt.figure(figsize=(20, 2)); plt.xlim((0, hidden.size(1))); plt.plot(numpy(hidden_mean))
    return hidden_mean.topk(topk), hidden_mean.topk(topk, largest=False)

def plot_top_weight(weight, topk=4):
    wm = weight.norm(dim=-1)
    plot(wm, figsize=(20, 2))
    values, indices = wm.topk(topk)
    heatmap(weight[indices], figsize=(20, 1))
    return values, indices

def unravel(i): return i // hidden_size, i % hidden_size
def indices_fn(indices): return [unravel(i) for i in numpy(indices)]

# wvo = wo.matmul(wv)
# show_topk(*wvo.view(-1).topk(5), indices_fn=indices_fn)
# show_topk(*wvo.view(-1).topk(5, largest=False), indices_fn=indices_fn)

def attn_out_transform(self, attn_out, alpha=1.0):
    wv = self.v_proj.weight.view(H, -1, hidden_size)[head]
    i = wv.norm(dim=0).argmax().item()
    w0, w1 = wv[:, i], attn_out[0, head, src]
    attn_out[0, head, src] = w0 * (w1.max() / w0.max() + w1.min() / w0.min()) / 2 * alpha
    return attn_out

def get_detach_fn(pos=None):
    def detach(hidden):
        if pos is None: return hidden.detach()
        h0, h1, h2 = hidden[:, :pos], hidden[:, pos: pos + 1], hidden[:, pos + 1:]
        h1 = h1.detach()
        return torch.cat([h0, h1, h2], dim=1)
    return detach

def get_detach_heads_fn(kept_head=None):
    def detach_heads(attn_weights):
        if kept_head is None: return attn_weights.detach()
        assert attn_weights.dim() == 4
        h0, h1, h2 = attn_weights[:, :kept_head], attn_weights[:, kept_head: kept_head + 1], attn_weights[:, kept_head + 1:]
        h0, h2 = h0.detach(), h2.detach() 
        return torch.cat([h0, h1, h2], dim=1)
    return detach_heads

def get_scale_fn(factor=0):
    def scale(hidden): return hidden * factor
    return scale

def plot_attn(attn, tokens, ytokens=None, ystart=None, ystop=None, y_pos=None, x_pos=None, topk=None,
            use_imshow=False, annot=False, figsize=(10, 10), fontsize=10, transpose=False, ax=None):
    ytokens = ytokens or tokens
    if y_pos is None and topk is not None:
        attn_ = attn.clone(); attn_[:, 0] = 0  # null attn to start pos is ignored
        y_pos, x_pos, _ = topk_md(attn_, k=topk) 
    if ystart is not None:
        ystop = ystop or attn.size(0)
        attn = attn[ystart: ystop]
        ytokens = ytokens[ystart: ystop]
        if y_pos is not None: y_pos = [p - ystart for p in y_pos]
    square = attn.size(0) == attn.size(1)
    if ax is None:
        if not square:
            figsize2 = (attn.size(1), attn.size(0))
            a = min(s1 / s2 for s1, s2 in zip(figsize, figsize2))
            figsize = [s * a for s in figsize2]
    if transpose:
        attn = attn.T
        tokens, ytokens = ytokens, tokens
        x_pos, y_pos = y_pos, x_pos
        figsize = figsize[::-1]
    if ax is None: plt.figure(figsize=figsize)

    if use_imshow:
        ax.imshow(attn)#, cmap='hot')
        ax.set_xticks(np.arange(0, attn.size(1), 1)); ax.set_xticklabels(tokens)
        ax.set_yticks(np.arange(0, attn.size(0), 1)); ax.set_yticklabels(ytokens)
    else:
        kwargs = dict(linewidths=0.1, linecolor='grey') if y_pos is None else {}
        ax = sns.heatmap(numpy(attn), square=square, cbar=False, annot=annot, fmt='d', 
            xticklabels=tokens, yticklabels=ytokens, ax=ax, **kwargs)
    _ = ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=fontsize, rotation=90)
    _ = ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=fontsize, rotation=0)
    if transpose: ax.tick_params(right=True, labelright=True, left=False, labelleft=False)#, top=True, labeltop=True) # cause x3 slowdown!
    kwargs = dict(linewidths=0.5, color='grey')
    if y_pos is not None: ax.hlines(y=y_pos, xmin=0, xmax=attn.size(1)-0.5*use_imshow, **kwargs)  # max-0.5 for imshow
    if x_pos is not None: ax.vlines(x=x_pos, ymin=0, ymax=attn.size(0)-0.5*use_imshow, **kwargs)
    # plt.show()

def plot_attn_attr(data_tuple, model, tokenizer, node, l, h, attn_patterns=None, k_shot=0, plot_attr=True):
    text, input_ids, labels, ranges, *args, o = data_tuple
    tokens = [t.replace('Ġ', '').replace('Ċ', '-'*12) for t in tokenizer.convert_ids_to_tokens(input_ids[0])]
    aw_size = o.attentions[0][0, 0].size()  # l*[bnij]->ij
    bos_indices = args[1]; ystop = aw_size[0]
    if plot_attr:
        H = o.attentions[0].size(1)
        fns = path2fns(node, partial(node2fn, outputs=o, ranges=ranges, labels=labels))
        fwd_fn = partial(sum_forward, outputs=o, output_layer=l)
        label_type = 'labels' if len(fns) == 0 else None
        fn = partial(mixed_forward, layer=l, head=h, labels=labels, label_type=label_type, outputs=o, ranges=ranges)
        keys = ['embed_mask', 'mlp_mask', 'attn_weights']
        to_layer = max(l) if isinstance(l, Iterable) else l
        x = OrderedDict((key, get_x(key, o, to_layer=to_layer)) for key in keys)
        _, _, ys, logits = attribute(fwd_fn, model, x, [fn] + fns, num_points=3, forward_only=True); print(ys)
        if isinstance(logits, (list, tuple)): logits = sum(logits)
        if logits.size(-1) == model.lm_head.out_features:  # lm_logits
            _ = show_predictions(tokenizer, *args, logits=logits[-1:], labels=labels, topk=4, sep='\t')
        elif logits.size(-1) == input_ids.size(1): # attn_logits, bij
            ystart = bos_indices[k_shot]
            attn_labels = attn_pattern2labels(ranges, node2path(node)[-1].data.attn_pattern or 'bos->ans0]', aw_size)
            y_pos, x_pos = attn_labels.nonzero().T
            plot_attn(logits[-1].softmax(dim=-1), tokens, ystart=ystart, ystop=ystop, y_pos=y_pos, x_pos=x_pos,
                fontsize=9, transpose=True, figsize=(20, 20)); plt.show()  # bij->ij
        if isinstance(h, Iterable) or h == H: return

    aw = o.attentions[l][0, h]
    if plot_attr: aa = o.attn_attr[node.name][l, h]
    attns, ystart = ([aw, aa], bos_indices[k_shot]) if plot_attr else ([aw], 0)
    if attn_patterns and attn_patterns[0].startswith('ans'): # 'ans]->ans0]' or 'ans]->ans0+' for relating heads
        ystart, ystop = 0, ystop - ystart

    _, axs = plt.subplots(1, len(attns), sharex=False, sharey=False,
                figsize=(20*len(attns)*(ystop-ystart)/aw_size[0], 20))
    if len(attns) == 1: axs = [axs]
    y_pos, x_pos = None, None
    if attn_patterns is not None:
        all_attn_labels = [attn_pattern2labels(ranges, ap, aw_size) for ap in attn_patterns]
        if len(all_attn_labels) == 1:
            attn_labels = all_attn_labels[0]
        else:  # steps that have multiple attn patterns, e.g. ['ans->ans0]', 'ans->ans0+'] of relating heads
            all_attn_labels = [(al, (al * aa).sum()) for al in all_attn_labels]
            attn_labels = max(all_attn_labels, key=lambda x: x[1])[0] # find the pattern that best matches aa
        y_pos, x_pos = attn_labels.nonzero().T
    if True: #with Timer():
        for ax, a in zip(axs, attns):
            plot_attn(a, tokens, ax=ax, ystart=ystart, ystop=ystop, y_pos=y_pos, x_pos=x_pos,
                fontsize=8, transpose=True, use_imshow=False)
        plt.show()

attn_patterns_by_step = {-1: ['bos->ans0]'], 0: ['bos->ans]', 'bos->query]'], 1: ['ans]->ans0]', 'ans]->ans0+']}
all_attn_patterns = join_lists(attn_patterns_by_step.values()) + ['bos->bos']

def plot_attn_attrs(data_tuples, model, tokenizer, node, topi=[0], head_attr_fn=None, attn_patterns=None, mix=False, **kwargs):
    if head_attr_fn is None: head_attr_fn = get_head_mlp_attr  # or get_matched_head_attr
    heads = np.array(topk_md(head_attr_fn(node.data), 10)[:2])[:, topi]
    heads = zip(*heads) if not mix else [heads]
    attn_patterns=attn_patterns or attn_patterns_by_step.get(node.data.step)
    for l, h in heads:
        s = f'{l}-{h}' if not mix else ' + '.join([f'{_l}-{_h}' for _l, _h in zip(l, h)])
        print(' -> '.join([s] + [n.name for n in node2path(node)]))
        for data_tuple in data_tuples:
            plot_attn_attr(data_tuple, model, tokenizer, node, l, h, attn_patterns=attn_patterns, **kwargs)

def cluster(emb, labels, n_clusters=3):
    assert emb.shape[0] == labels.shape[0], '%d != %d' % (emb.shape[0], labels.shape[0])
    centroids = emb.reshape(n_clusters, len(labels) // n_clusters, emb.shape[-1]).mean(axis=1)
    kmeans = KMeans(n_clusters=n_clusters)#, init=centroids)
    labels_ = kmeans.fit(emb).labels_
    for label in list(set(labels)):
        if Counter(labels_[labels == label]).most_common()[0][1] < (labels == label).sum():# - abs(label):
#             print(label)
            return False, labels_
    return True, labels_

def visualize_by_pca(emb, labels):
    pca = PCA(n_components=2)
    data = pca.fit_transform(emb)
    _ = plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('jet', 3))
    _ = plt.colorbar()
    plt.show()

def get_query(self, h):
    query = self.q_proj(h)
    query = self._split_heads(query, self.num_heads, self.head_dim)
    query = query[0, head2, src:src+1]
    return query

def get_key(self, h):
    key = self.k_proj(h)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    key = key[0, head2, :]
    return key

def plot_tgt_attn(a, ax=None, title=None):
#     print(a.view(-1)[tgt_positions[4:]].mean())
    labels = np.array(tokens).reshape(nrows, -1)
    relative_tgt_positions = tgt_positions % a.size(1) # == ncols + 3
    right_attn = a.argmax(1) == relative_tgt_positions
    yticklabels = ['' if i else 'x' for i in right_attn]
    if ax is None:
        _ = plt.figure(figsize=(2.5 * a.size(1) / 9, 5 * a.size(0) / 24))
        _ = sns.heatmap(numpy(a) ,cbar=False, annot=labels, fmt='', xticklabels=False, yticklabels=yticklabels)
        if title is not None: plt.title(title)
    else:
        _ = sns.heatmap(numpy(a) ,cbar=False, annot=labels, fmt='', xticklabels=False, yticklabels=yticklabels, ax=ax)
        if title is not None: ax.set_title(title)
#     plt.show()


def gen_detach_pairs(module, exit_module, detach_type='output'):
    assert detach_type in ['output', 'residual']
    pairs = []
    for block in blocks:
        if module in [block, get_attn_module(block)]: pairs += [(block, 'ffn_%s_transform' % detach_type)]
        elif block == exit_module: break
        elif pairs: pairs += [(block, 'attn_%s_transform' % detach_type), (block, 'ffn_%s_transform' % detach_type)]
    return pairs

def gen_detach_heads_tuples(module, exit_module, kept_layer, kept_head):
    tuples = None
    for i, block in enumerate(blocks):
        if module in [block, get_attn_module(block)]: tuples = []
        elif block == exit_module: break
        elif tuples is not None:
            tuples.append((get_attn_module(block), 'attn_weights_transform',
                          get_detach_heads_fn(kept_head=kept_head if i == kept_layer else None)))
    return tuples

# def forward(module, names, values=None, exit_module=None, extra_tuples=None,
#             detach_type=None, detach_pos=None, kept_layer=None, kept_head=None):
#     if type(names) != list: names, values = [names], [values]
#     if type(names) == list and type(values) != list: values = [values for _ in range(len(names))]
#     for name, value in zip(names, values): setattr(module, name, value)
#     if exit_module is not None: setattr(exit_module, 'exit', True)
#     if extra_tuples is not None:
#         for m, name, val in extra_tuples: setattr(m, name, val)
#     if detach_type is not None:
#         detach_pairs = gen_detach_pairs(module, exit_module, detach_type=detach_type)
#         for m, name in detach_pairs: setattr(m, name, get_detach_fn(detach_pos))
#     if kept_head is not None:
#         detach_tuples = gen_detach_heads_tuples(module, exit_module, kept_layer=kept_layer, kept_head=kept_head)
#         for m, name, fn in detach_tuples: setattr(m, name, fn)
#     try: outputs = model(**inputs, output_attentions=True, output_hidden_states=exit_module is not None)
#     finally:
#         if values[0] is None: embs = [getattr(module, name) for name in names]
#         for name in names: try_delattr(module, name)
#         if exit_module is not None: try_delattr(exit_module, 'exit')
#         if detach_type is not None:
#             for m, name in detach_pairs: try_delattr(m, name)
#         if kept_head is not None:
#             for m, name, _ in detach_tuples: try_delattr(m, name)
#         if extra_tuples is not None:
#             for m, name, _ in extra_tuples: try_delattr(m, name)
#     if values[0] is None and len(names) == 1: embs = embs[0]
#     return embs if values[0] is None else outputs


# def test(hidden, query, key=None, logits=None, always_show=False):
#     if logits is None:
#         if key is None:
#             key = self.k_proj(hidden)
#             key = self._split_heads(key, self.num_heads, self.head_dim)[0, head2]
#         logits = (query * key).sum(dim=-1)
#     else:
#         always_show = True
#     cand_pos = torch.LongTensor(cand_positions).view(-1, n_candidates)
#     is_extremal = [logits[p] == logits[cand_pos[i]].max() for i, p in enumerate(tgt_positions)]
#     if always_show or sum(is_extremal[1:]) / len(tgt_positions[1:]) > 0.9:
#         logits[0] = logits[1]
#         plot(logits)
#         _ = plt.xticks(range(len(logits)), tokens)
#         for p, b in zip(tgt_positions, is_extremal): plt.axvline(x=p, color='gray' if b else 'r')
#         plt.show()
#         probs = logits[cand_positions].view(-1, n_candidates).softmax(-1)[cand_is_tgt]
#         print(numpy(probs), '\n', probs.mean())
#         return True
#     return False 


def plot_tgt_attn_losses(labels, losses, losses1):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(20, 4))
    losses, losses1 = [int(l*100) for l in losses], [int(l*100) for l in losses1]
    rects1 = ax.bar(x - width/2, losses, width, label='loss')
    rects2 = ax.bar(x + width/2, losses1, width, label='loss1')
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(labels)
    _ = ax.legend()
    _ = ax.bar_label(rects1, padding=3)
    _ = ax.bar_label(rects2, padding=3)


def create_mask(from_positions, to_positions, accum=False):
    mask = torch.zeros(1, seq_len, seq_len)
    for i in range(0, nrows):
        if not accum:
            mask[:, from_positions[i], to_positions[i]] = 1
        else:
            mask[:, from_positions[i], to_positions[:i]] = 1 / i if i > 0 else 0
    return mask

# combined_weights = {}

# def get_combined_w(layer, head, qk=False):
#     if (layer, head, qk) in combined_weights: return combined_weights[(layer, head, qk)]
#     wq, wk, wv, wo = get_head_weights(layer, head)
#     w = torch.matmul(wq, wk) if qk else torch.matmul(wv, wo)
#     combined_weights[(layer, head, qk)] = w
#     return w


# ans_positions = bos_indices + 1
# src = bos_indices[-1].item()
# pred_label = outputs.logits[0, src].argmax().item()

# tokens = [token.replace('Ġ', '').replace('Ċ', '^') for token in tokenizer.tokenize(text)]
# seq_len = len(tokens)
# answer = tokens[-2]  # tokens[-1] == '^'

# cand_range = range(eos_indices[-2] + 1, bos_indices[-1])
# n_candidates = len(cand_range); assert n_candidates >= 1, str(n_candidates)
# tgt = [i for i in cand_range if ans_fn(tokens[i]) == answer][0] if n_candidates > 1 else cand_range[0]
# # cand_positions = [i for i, token in enumerate(tokens[:-1]) if '^' in tokens[max(0, i - n_candidates): i]]

# tgt_positions = []
# for i in range(len(ans_positions)):
#     ans_pos, prev_ans_pos = ans_positions[i], ans_positions[i - 1] if i - 1 >= 0 else -1
#     for pos in range(prev_ans_pos + 1, ans_pos):
#         if ans_fn(tokens[pos]) == tokens[ans_pos]: tgt_positions.append(pos)
# tgt_positions = torch.LongTensor(tgt_positions)
# assert len(tgt_positions) == len(ans_positions), '%d != %d' % (len(tgt_positions), len(ans_positions))
# # cand_is_tgt = torch.LongTensor(cand_positions).view(-1, n_candidates) == tgt_positions.unsqueeze(-1)





# # x = torch.Tensor([[100., 100.02, 100.],
# #                 [0.1, 1, 0.1]])
# # x = torch.Tensor([[100., 100.1, 100.],
# #                 [0.1, 1, 0.1]])
# x = torch.Tensor([[100.1, 100., 100.1],
#                 [0.1, 1, 0.1]])
# _ = x.requires_grad_(True)
# x.softmax(-1)

# m = torch.ones(1, x.size(0))#; _ = m.requires_grad_(True)
# num_points = 10
# m = scaled_input(m, num_points)
# xm = einsum('ni,bn->bi', x, m); _ = xm.requires_grad_(True)
# y = xm.log_softmax(-1)[:, 1]
# gm = torch.autograd.grad(y.unbind(), m)[0]
# gm.mean(0)
# x.sum(0).log_softmax(-1)