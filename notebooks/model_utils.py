# from ast import pattern
from collections import OrderedDict
from difflib import restore
# from typing import Iterable
from collections.abc import Iterable
from lib2to3.pgen2 import token
from matplotlib import scale  # same as typing.Iterable?
import numpy as np
import math
from dataclasses import dataclass
from functools import reduce, partial
from itertools import chain, product, combinations, cycle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA

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

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

from common_utils import numpy, einsum, my_isinstance, convert_ids_to_tokens, show_topk, topk_md, \
    equal, join_lists, iterable, pad, Timer, maybe_map


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
    topk: int = None
    layer: int = None
    head: int = None
    label_type: str = None
    attribute_k: bool = False
    attr: Attributions = None

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

from bitsandbytes.nn import Linear8bitLt

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
    self = fixed_pos_embedding
    if not hasattr(self, 'sin_cached'):
        seq_len = 2048
        if gpt_neox_style:
            rotary_emb = RotaryEmbedding(x.shape[-1], 2048, base=10000)
            self.cos_cached, self.sin_cached = rotary_emb(x, seq_len=seq_len)
        else:
            dim = x.shape[-1]
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
            sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).float()
            self.sin_cached, self.cos_cached = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
    return self.sin_cached, self.cos_cached

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
    # i(r/2) -> 11i(r/2) -> 11ir
    # sin, cos = map(lambda t: t[None, None, offset : x.shape[2] + offset, :].repeat(1, 1, 1, 2), sincos)
    sin, cos = map(lambda t: t[:, :, offset : x.shape[2] + offset, :], sincos) # gpt_neox_style fixed_pos_embedding
    return (x * cos) + (rotate_half(x) * sin)  # bnir,11ir->bnir

def apply_rotary_pos_emb(query_rot, key_rot, seq_len=2048, offset=0, is_gpt_neox=False):
    sincos = fixed_pos_embedding(key_rot, seq_len=seq_len, gpt_neox_style=is_gpt_neox)
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
    for name in self.backup_names: setattr(self, name + '0', getattr(self, name))

def restore_heads(self):
    for name in self.backup_names:
        setattr(self, name , getattr(self, name + '0'))
        # delattr(self, name + '0')

def trim_heads(self, kept_heads):
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        # assert getattr(self, proj_name).bias is None #lxy，报错
        weight = getattr(self, proj_name).weight
        if proj_name == 'out_proj':
            patterns = ['e (n d) -> n d e', 'n d e -> e (n d)']
            size = (self.head_dim * len(kept_heads), self.embed_dim)
        else:
            patterns = ['(n d) e -> n d e', 'n d e -> (n d) e']
            size = (self.embed_dim, self.head_dim * len(kept_heads))
        weight = rearrange(weight, patterns[0], n=self.num_heads)[kept_heads] #[kept_heads,head_dim,embed_dim]
        weight = rearrange(weight, patterns[1])
        proj = nn.Linear(*size, bias=False)
        proj.weight = nn.Parameter(weight)
        setattr(self, proj_name, proj)
    self.num_heads = len(kept_heads)

def attn_forward(block, hq, hk, hv, by_head=None, #compute_head_input=True,
                attention_mask=None, head_mask=None, attn_weights=None): # gptneo
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
            attn_weights=None, attn_labels=None, hidden_states0=None, attribute_k=False, trim=True, scaled=True):
    if isinstance(layer, Iterable):  # tuple, list or np.ndarray
        # assert labels is None and attn_labels is not None
        # assert isinstance(attn_labels, (list, tuple))
        # assert isinstance(hidden_states0, (list, tuple))
        # assert len(attn_labels) == len(hidden_states) == len(hidden_states0) == len(layer) == len(head)
        if not isinstance(hidden_states, (list, tuple)): hidden_states = [hidden_states] * len(layer)
        kwargs_list = [{} for _ in range(len(layer))]
        if labels is not None:
            assert len(labels) == len(attn_weights) == len(layer), f'{len(labels)}, {len(attn_weights)}, {len(layer)}'
            kwargs_list = [{'labels': l, 'attn_weights': aw} for l, aw in zip(labels, attn_weights)]
        if attn_labels is not None:
            assert len(attn_labels) == len(hidden_states0) == len(layer), f'{len(attn_labels)}, {len(hidden_states0)}, {len(layer)}'
            kwargs_list = [{'attn_labels': l, 'hidden_states0': h} for l, h in zip(attn_labels, hidden_states0)]
        # outputs = [head_forward(model, hs, l, h, loss_reduction=loss_reduction,
        #     attn_labels=al, hidden_states0=hk, trim=trim, scaled=scaled)
        #     for hs, l, h, al, hk in zip(hidden_states, layer, head, attn_labels, hidden_states0)]
        outputs = [head_forward(model, hs, l, h, loss_reduction=loss_reduction, trim=trim, scaled=scaled, **kwargs)
            for hs, l, h, kwargs in zip(hidden_states, layer, head, kwargs_list)]
        return Outputs(hidden_states=[o.hidden_states for o in outputs],
            logits=[o.logits for o in outputs], loss=sum(o.loss for o in outputs))

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
        backup_heads(self)
        try:
            trim_heads(self, [head])
            if attn_weights is not None: attn_weights = attn_weights[:, [head]] # bnij->b1ij
            _, attn_logits, _, head_output = attn_forward(block, hq, hk, hv, attn_weights=attn_weights, by_head=by_head)
            if head_output is not None: assert head_output.size(1) == 1, str(head_output.size())
            head = 0  # used to index head_output and attn_logits
            restore_heads(self)
        except Exception:
            restore_heads(self)
            raise  # print(traceback.format_exc())
    else:
        _, attn_logits, _, head_output = attn_forward(block, hq, hk, hv, attn_weights=attn_weights, by_head=by_head)
    if head_output is not None: head_output = head_output[:, head]
    logits, loss = None, None
    if labels is not None:
        logits, loss = lm_head_forward(model, head_output, labels=labels, loss_reduction=loss_reduction, scaled=scaled)
    elif attn_labels is not None:
        # may do some attn_logits masking here
        logits = attn_logits[:, head]  # bnij->bij
        # print('in head_forward, logits =', torch.einsum('bij->b', logits * torch.ones_like(logits).tril()))
        if not attribute_k: logits = logits.log_softmax(-1)
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

def lm_head_forward(model, hidden_states, labels=None, loss_reduction=None, compact=True, scaled=False):
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

def forward0(model, inputs, labels=None, loss_reduction=None, by_head=None, attribute_layer=None, 
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
        attn_output, *attn_fwd_output = attn_forward(b, h, h, h, by_head=by_head,
                            head_mask=head_mask[i], attn_weights=attn_weights[i])
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

def get_argmax_labels(model, hidden_states, labels):
    logits = model.lm_head(model.transformer.ln_f(hidden_states))
    argmax_labels = labels.clone()
    argmax_labels[labels != -100] = logits.argmax(-1)[labels != -100]
    return argmax_labels

def locate_answers(input_ids, tokenizer, bos_token='Ġ->', eos_token='Ċ', nrows=None):
    assert input_ids.size(0) == 1  # bsz == 1
    bos_id = tokenizer.convert_tokens_to_ids(bos_token)
    bos_indices = (input_ids[0] == bos_id).nonzero().squeeze(1).tolist()
    # print(bos_indices)
    if nrows is not None:
        assert nrows == len(bos_indices)
    else:
        nrows = len(bos_indices)
    if eos_token is not None:
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        eos_indices = (input_ids[0] == eos_id).nonzero()[-nrows:].squeeze(1).tolist()
    else:
        # eos_indices = bos_indices[1:] + [input_ids.size(1)]
        eos_indices = [bos_i + 2 for bos_i in bos_indices]
    # labels = torch.ones(input_ids.size(0), input_ids.size(1) - 1).long() * (-100)
    labels = torch.ones_like(input_ids) * (-100)
    answers = []
    for bos_i, eos_i in zip(bos_indices, eos_indices):
        # eos_i = bos_i + 2  # show only the first answer token
        ans_ids = input_ids[0, bos_i + 1: eos_i]
        labels[0, bos_i: eos_i - 1] = ans_ids
        answers.append(ans_ids.numpy())
    return bos_indices, eos_indices, answers, labels

# bos_token='▁is'; eos_token='</s>' for s2s
def make_data_tuple(text, tokenizer, k_shot=3, bos_token='Ġ->', eos_token='Ċ', s2s=False):
    examples = text.strip().split('\n')
    input_ids = tokenizer.encode(text, return_tensors='pt')
    # tokens = tokenizer.tokenize(text)
    # # tokenize without tokenization artifact -> needed for visualization, from unseal
    # tokens = list(map(tokenizer.convert_tokens_to_string, map(lambda x: [x], tokens)))

    bos_indices, eos_indices, answers, labels = locate_answers(input_ids, tokenizer, bos_token=bos_token, eos_token=eos_token)
    if s2s:  # for t5 models
        bos_i, eos_i = bos_indices[-1], eos_indices[-1]
        assert eos_i == input_ids.size(1) - 1, f'{eos_i} != {input_ids.size()}[1] - 1'
        assert tokenizer.convert_ids_to_tokens(input_ids[0, -1].item()) == eos_token == '</s>', \
            f"{tokenizer.convert_ids_to_tokens(input_ids[0, -1].item())} != '</s>'"
        input_ids = torch.cat([input_ids[:, : bos_i + 1], input_ids[:, -1:]], dim=1) # append trailing '</s>'
        answers, labels = answers[-1:], labels[:, bos_i: eos_i - 1]
        bos_indices, eos_indices = [bos_i - bos_i], [eos_i - bos_i]
    else:
        labels[:, :bos_indices[k_shot]] = -100  # 只算k_shot个示例后的loss
    return input_ids, labels, examples, bos_indices, eos_indices, answers

def get_prob_dist(d, topk=5, digits=3):
    return {k: round(math.exp(v), digits) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk]}

def show_predictions(tokenizer, examples, bos_indices, eos_indices, answers, 
        logits=None, labels=None, candidates=None, mask_logits_fn=None,
        k_shot=3, topk=5, loss_reduction='mean', sep='\t', verbose=True):
    use_openai_api = hasattr(logits, 'token_logprobs')  # isinstance(model, types.FunctionType)
    if use_openai_api: ans_nlls = []
    if not use_openai_api and mask_logits_fn is not None: logits = mask_logits_fn(logits)
    
    assert len(bos_indices) == len(examples), '%d != %d' % (len(bos_indices), len(examples))
    top1_corrects, answer_probs, candidate_probs = [], [], []
    convert_fn = tokenizer.convert_ids_to_tokens if True else partial(convert_ids_to_tokens, tokenizer=tokenizer)
    for i, (example, bos_i, eos_i, ans_ids) in enumerate(zip(examples, bos_indices, eos_indices, answers)):
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
                    else show_topk(*dist.topk(topk), indices_fn=convert_fn), sep, example) 
    if use_openai_api:
        loss = (ans_nlls if loss_reduction == 'none' else sum(ans_nlls) / len(ans_nlls))
    else:
        loss = compute_loss(logits, labels, reduction=loss_reduction)
        loss = loss.item() if loss_reduction == 'mean' else loss[labels != -100].tolist()  # 'none'
    return loss, top1_corrects, answer_probs, candidate_probs

def predict(model, tokenizer, text, _examples, k_shot=3, bos_token='Ġ->', eos_token='Ċ', verbose=True):
    input_ids, labels, *args = make_data_tuple( # args = [examples, bos_indices, eos_indices, answers]
        text, tokenizer, k_shot=k_shot, bos_token=bos_token, eos_token=eos_token)
    candidates = [[tokenizer.encode(' ' + token)[0] for token in cands[0]] for _, _, cands, _ in _examples] \
        if _examples is not None else None
    with torch.no_grad():
        # logits = model(input_ids.to(getattr(model, 'device', 'cpu'))).logits
        # if isinstance(logits, torch.Tensor): logits = logits.to('cpu').float()# softmax on cpu needs float32
        with Timer('forward0'): o = forward0(model, input_ids.to(model.device), labels=labels.to(model.device),
                    output_hidden_states=True, by_head=['head_input'])
        logits = o.logits
    loss, top1_corrects, answer_probs, candidate_probs = show_predictions(
        tokenizer, *args, logits=logits, labels=labels, loss_reduction='mean',
        candidates=candidates, k_shot=k_shot, topk=3, verbose=verbose)
    if verbose: print(loss)
    attn_attr = {}
    return [text, input_ids, labels] + args + [o, attn_attr]
    return loss, top1_corrects, answer_probs, candidate_probs

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
        return -outputs.loss, outputs.logits
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
    with torch.enable_grad():
        for key in x:
            scaled_x[key] = scaled_input(x[key], num_points)
            if True or key in grad_keys: grad[key] = torch.zeros_like(x[key])
        ys = []
        for i in range(0, num_points, batch_size):
            scaled_x_ = OrderedDict({key: scaled_x[key][i: i + batch_size] for key in x})
            with Timer('sum_forward'): o = forward_fn(model, **scaled_x_)
            y, logits = post_forward_fn(model, o); ys.append(y)
            if forward_only: continue
            with Timer('grad'): grad_ = torch.autograd.grad(y.flatten().unbind(), list(scaled_x.values()))
                # [v for k, v in scaled_x.items() if k in grad_keys])
            for j, key in enumerate(grad_keys):
                grad[key] += grad_[j].sum(dim=0, keepdim=True)
    if forward_only: return None, torch.cat(ys), logits
    attr = {key: (grad[key] / num_points * x[key]).squeeze(0) for key in grad_keys}

    attn_attr = attr.get('attn_weights')
    neuron_attr = attr.get('neuron_mask')
    head_attr = torch.einsum('lnij->ln', attn_attr) if attn_attr is not None \
        else attr.get('head_mask')#.sum(-1)  # lni->ln  #torch.einsum('lnie->ln', neuron_attr)
    # head_attr = attr.get('head_mask')#.sum(-1)  # lni->ln
    mlp_attr = attr.get('mlp_mask')#.sum(-1, keepdim=False) # li->l
    emb_attr = attr.get('embed_mask')#.sum(-1, keepdim=True) # i->1
    attr = Attributions(attn=attn_attr, head=head_attr, neuron=neuron_attr, mlp=mlp_attr, embed=emb_attr)
    return attr, torch.cat(ys), logits

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

def attribute22(forward_fn, model, x, post_forward_fn):
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=False)
    with torch.no_grad():
        o = forward_fn(model, **x, truncate_layers=True)
        y, logits = post_forward_fn(model, o)
    assert y.ndim == 1
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    head_attr, mlp_attr = torch.zeros(L, H), torch.zeros(L)
    assert (y.size(0) - 1) % (H + 1) == 0
    to_layer = (y.size(0) - 1) // (H + 1) # by solving equation b = 1 + ln + l
    assert to_layer == (torch.einsum('bli->l', x['mlp_mask']) > 0).sum().item()
    L = to_layer
    embed_attr = y[:1]#.view(1, 1)
    head_attr[:], mlp_attr[:] = embed_attr.item(), embed_attr.item()
    head_attr[:L] = y[1: 1 + L * H].view(L, H)  # ln
    mlp_attr[:L] = y[1 + L * H:]#.view(L, 1)  # l1
    return Attributions(embed=embed_attr, head=head_attr, mlp=mlp_attr)

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

def plot_attr(attr, attr2):
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8, 4))  # (12, 4)
    def concat_attrs(head_attr, mlp_attr, embed_attr):
        # return torch.cat([head_attr, einops.repeat(mlp_attr, 'l -> l 2'), einops.repeat(embed_attr, '1 -> l 2', l = head_attr.size(0))], dim=1)
        return torch.cat([head_attr, einops.repeat(mlp_attr, 'l -> l 2'), ], dim=1)
    for ax, a in zip(axs, [concat_attrs(attr.head, attr.mlp, attr.embed), concat_attrs(attr2.head, attr2.mlp, attr2.embed)]):
        res = sns.heatmap(a, cbar=False, ax=ax)
        _ = res.set_yticklabels(res.get_ymajorticklabels(), rotation=0)
        # res.tick_params(top=False, right=True, labeltop=False, labelright=True)
    plt.show()

def get_head_rank(head_attr, layer, head, topk=20):
    if head is not None:
        head2rank = {k: v for k, v in zip(zip(*topk_md(head_attr, topk)[:2]), range(topk))}
        return head2rank.get((layer, head), None)
    else: # mlp
        head2rank = {k: v for k, v in zip(zip(*topk_md(head_attr, topk)[:1]), range(topk))}
        return head2rank.get((layer,), None)

def data2str(data):
    i, topk, layer, head, label_type, attribute_k = data.step, data.topk, data.layer, data.head, data.label_type, data.attribute_k
    # s = f'[{i}] top{topk} {layer}' if head is None else f'[{i}] top{topk} {layer}-{head}'
    s = f'[{i}] top{topk} '
    if head is None: s += f'{layer}'
    elif not isinstance(layer, Iterable): s += f'{layer}-{head}'
    else: s += ','.join([f'{l}-{h}' for l, h in zip(layer, head)])

    if label_type is not None: s = s + ' ' + label_type
    if attribute_k: s = s + ' ' + 'attr_k'
    return s

def get_argmax_attn_labels(o, layer, head, labels=None):
    # attn_labels = torch.einsum('bnij,bnj->bnij', o.attentions[layer], o.head_inputs[layer].norm(dim=-1)) # bnje->bnj
    # return attn_labels[0, head]  # 1nij->ij
    attn_labels = torch.einsum('bij,bj->bij', o.attentions[layer][:, head], o.head_inputs[layer][:, head].norm(dim=-1))
    if labels is not None: attn_labels = torch.einsum('bij,bi->ij', attn_labels, (labels != -100).float()) # b=1
    return attn_labels

def node2fn(model, node, outputs, labels, attn_attr):
    d = node.data
    i, layer, head, label_type, attribute_k = d.step, d.layer, d.head, d.label_type, d.attribute_k
    if head is None:
        return partial(mlp_forward, layer=layer) if label_type is None \
            else partial(mlp_forward, layer=layer, labels=labels)
    def get_kwargs(layer, head):
        if label_type in ['argmax_attn_labels', 'attn_labels']:
            attn_labels = get_argmax_attn_labels(outputs, layer, head, labels=labels) \
                if label_type.startswith('argmax') else attn_attr[node.parent.name][layer, head]
            return {'attn_labels': attn_labels, 'hidden_states0': outputs.hidden_states[layer]}
        else:
            assert label_type in ['labels', 'argmax_labels'], label_type
            _labels = get_argmax_labels(model, outputs.head_outputs[layer][:, head], labels) \
                if label_type == 'argmax_labels' else labels
            return {'labels': _labels, 'attn_weights': outputs.attentions[layer]}
    kwargs = maybe_map(get_kwargs, layer, head)

    # if label_type in ['argmax_attn_labels', 'attn_labels']:
    #     def get_attn_labels(layer, head, label_type):
    #         return get_argmax_attn_labels(outputs, layer, head, labels=labels) \
    #             if label_type.startswith('argmax') else attn_attr[node.parent.name][layer, head]
    #     if isinstance(layer, Iterable): # tuple, list or np.ndarray
    #         attn_labels = [get_attn_labels(l, h, label_type) for l, h in zip(layer, head)]
    #         hidden_states0 = [outputs.hidden_states[l] for l in layer]
    #     else:
    #         attn_labels = get_attn_labels(layer, head, label_type)
    #         # attn_labels = attn_labels / (attn_labels.sum(-1, keepdim=True) + 1e-9)  # ij->i1 # don't normalize attn attr
    #         hidden_states0 = outputs.hidden_states[layer]
    #     kwargs = {'hidden_states0': hidden_states0, 'attn_labels': attn_labels, 'attribute_k': attribute_k}
    # else:  # label_type in ['labels', 'argmax_labels']
    #     def get_labels(layer, head, label_type):
    #         return get_argmax_labels(model, outputs.head_outputs[layer][:, head], labels) \
    #             if label_type == 'argmax_labels' else labels
    #     if isinstance(layer, Iterable): # tuple, list or np.ndarray
    #         labels = [get_labels(l, h, label_type) for l, h in zip(layer, head)]
    #         attn_weights = [outputs.attentions[l] for l in layer]
    #     else:
    #         labels = get_labels(layer, head, label_type)
    #         attn_weights = outputs.attentions[layer]
    #     kwargs = {'attn_weights': attn_weights, 'labels': labels}
    return partial(head_forward, layer=layer, head=head, **kwargs)

def get_head_weights(model, layer, head=None, transpose=True, absorb_ln=False):
    m = model.transformer.h[layer].attn
    H = m.num_heads
    (qkv_pattern, o_pattern) = ('(n d) e -> n d e', 'e (n d) -> n e d') \
        if not transpose else ('(n d) e -> n e d', 'e (n d) -> n d e')
    wq, wk, wv = [rearrange(getattr(m, name).weight.data, qkv_pattern, n=H)
                for name in ['q_proj', 'k_proj', 'v_proj']]
    if absorb_ln:
        gamma = model.transformer.h[layer].ln_1.weight.data
        pattern = qkv_pattern.split('->')[1].replace(' ', '')
        pattern = pattern + ',e->' + pattern  # nde,e->nde or ned,e->ned
        wq, wk, wv = [torch.einsum(pattern, w, gamma) for w in [wq, wk, wv]]
    wo = rearrange(getattr(m, 'out_proj').weight.data, o_pattern, n=H)
    if head is not None: wq, wk, wv, wo = [w[head] for w in [wq, wk, wv, wo]]
    return wq, wk, wv, wo

def combine_weights(weights, qk=True, with_embedding=False, BA=False):
    wq, wk, wv, wo = weights
    wqt = wq.t()
    if with_embedding:
        wqt, wk = we.t().mm(wqt), wk.mm(we)
        wo, wv = wu.mm(wo), wv.mm(we)
    if BA: return wk.mm(wqt) if qk else wv.mm(wo)
    return wqt.mm(wk) if qk else wo.mm(wv)

def plot_eigv(w, start_i=0, end_i=None, alpha=0.1, plot=True):
    if w.size(0) == w.size(1): w = w.eig()[0]
    else: assert w.size(1) == 2
    # w = w.detach()#.numpy()
    x, y = w[:, 0], w[:, 1]
    eigv_positivity = x.sum() / (x**2 + y**2).sqrt().sum()
    eigv_reality = x.abs().sum() / (x**2 + y**2).sqrt().sum()
    if plot:
        if start_i is None: start_i = 0
        if end_i is None: end_i = len(w)
        start_i = 0
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x[start_i: end_i], y[start_i: end_i], '.', alpha=alpha); plt.show()
        start_i = 1
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x[start_i: end_i], y[start_i: end_i], '.', alpha=alpha); plt.show()
    return eigv_positivity.item(), eigv_reality.item()

def filter_eigv(w, v, q, verbose=True):
    from scipy.stats.stats import pearsonr
    w2 = []
    for i, (wi, vi) in enumerate(zip(w, v.T)):
        corrcoef, p_val = pearsonr(q @ vi, range(q.size(0)))
        if abs(corrcoef) >= 0.1: print(i, wi, round(corrcoef, 3))
        if corrcoef >= 0.1 and wi[1] == 0: continue
        w2.append(wi.tolist())
    w2 = torch.Tensor(w2)
    def stat(x):
        return '%.1f * %d - %.1f * %d' % (x[x > 0].mean().item(), (x > 0).sum().item(), x[x < 0].abs().mean().item(), (x < 0).sum().item())
    if verbose: print(' ->\n'.join([stat(w[:, 0]), stat(w2[:, 0])]))
    return w2

def get_eigv(m): return plot_eigv(m.eig()[0], plot=False)

@torch.no_grad()
def compute_eigv_positivity(model, L, H, use_ln=False):
    we = model.transformer.wte.weight.data.t()
    wu = model.lm_head.weight.data
    _e = mlp_forward(model.transformer.h[0], we.T) + we.T

    eigv_positivity = torch.zeros((L, H, 3)) 
    for layer in tqdm(range(L)):
        e = model.transformer.h[layer].ln_1(_e) if use_ln else _e
        for head in range(H):
            wq, wk, wv, wo = get_head_weights(model, layer, head, transpose=True)
            # A, B = wu, ln_f(e @ wv @ wo).T
            A, B = wu @ wo.T, (e @ wv).T; eig_ov = get_eigv_pos(B @ A)
            q, k = (we.T @ wq), (we.T @ wk).T; eig_qk0 = get_eigv_pos(k @ q)
            q, k = (e @ wq), (e @ wk).T; eig_qk = get_eigv_pos(k @ q)
            eigv_positivity[layer, head] = torch.Tensor([eig_qk0, eig_qk, eig_ov])
    return eigv_positivity

def get_affinities(w1, w2):
    pattern = 'mce,ned->mncd'
    if w1.ndim == 4: pattern = 'p' + pattern
    a = torch.einsum(pattern, w1, w2).norm(dim=(-2,-1))
    return a, a / torch.einsum('m,n', w1.norm(dim=(-2,-1)), w2.norm(dim=(-2,-1)))

def _get_affinities(w1, w2): return (w1 @ w2).norm(dim=(-2,-1)) / w1.norm(dim=(-2,-1))  # mnde,ed->mndd->mn

def get_affinities3(w1, w2, w3):
    a = torch.einsum('mce,nef->mncf', w1, w2)
    # m = torch.einsum('mnce,oed->mnocd', a, w3) # very subtle bug! Result is wrong when o=m! Don't know why!
    # a = m.norm(dim=(-2,-1)) # mnodd->mno
    a = rearrange([(a @ w).norm(dim=(-2, -1))  # mnce,ed->mncd->mn
        for w in w3], 'o m n -> m n o')   # Result is right. Also more mem efficient
    return a, a / torch.einsum('m,n,o->mno', w1.norm(dim=(-2,-1)), w2.norm(dim=(-2,-1)), w3.norm(dim=(-2,-1)))

def _get_affinities3(w1, w2, w3, chunks=8):  # not faster
    assert w2.size(0) == 1, str(w2.size())
    a = torch.einsum('mce,nef->mncf', w1, w2)
    a = rearrange([(a @ w).norm(dim=(-2, -1))  # m1ce,o'ed->mo'cd->mo'
        for w in w3.chunk(chunks)], 'chunks m o -> m 1 (chunks o)') \
        if chunks is not None else (a @ w3).norm(dim=(-2, -1)).unsqueeze(1)  # m1ce,oed->mocd->mo->m1o  # cuda out of mem
    return a, a / torch.einsum('m,n,o->mno', w1.norm(dim=(-2,-1)), w2.norm(dim=(-2,-1)), w3.norm(dim=(-2,-1)))

def get_ov_affinities(model, l1, wv1, wo1, layer_range):
    wvs0, wos0 = zip(*[get_head_weights(model, l0, transpose=True)[2:] for l0 in layer_range])
    wvs0 = rearrange(list(wvs0), 'l n e d -> l n e d')
    wos0 = rearrange(list(wos0), 'l n d e -> l n d e')
    # na0 = _get_affinities(wvs0 @ wos0, wv1)
    na0 = _get_affinities(wos0, wv1)  # much faster with similar results
    # [(l, h, v, eigv_positivity[l, h], k_comp_max[l, h], pos_heads2val.get((l, h))) 
    #   for l, h, v in list(zip(*topk_md(na0, 20)))]
    return na0

def get_qk_affinities(model, l1, wv1, wo1, layer_range):
    wqs2, wks2 = zip(*[get_head_weights(model, l2, transpose=True)[:2] for l2 in layer_range])
    wqs2 = rearrange(list(wqs2), 'l n e d -> l n e d')
    wks2 = rearrange(list(wks2), 'l n e d -> l n e d')
    # na2 = _get_affinities(wks2 @ wqs2.transpose(-2, -1), wo1.T)  # lhed,lhde->lhee aff with ed
    na2 = _get_affinities(wqs2.transpose(-2, -1), wo1.T)  # lhde aff with ed, much faster with similar results
    H = wqs2.size(1)
    na2 = torch.cat([torch.zeros(min(layer_range), H), na2])  # cat(l1+1 n, l-l1-1,n)->ln
    # [(i, f'{l}-{h}', v, eigv_positivity[l, h], k_comp_max[l, h]) for i, (l, h, v) in enumerate(zip(*topk_md(na2, 30)))]
    return na2

Id = lambda x: x

@torch.no_grad()
def compute_eigv_pos012(model, l1, h1, wv1, wo1, heads0, heads2, eigv_positivity,
                    heads_1=None, use_ln=False, _e=None, verbose=False):
    # with torch.no_grad():
    if True:
        blocks = model.transformer.h
        eigv_pos012 = {}
        if use_ln: ln1 = blocks[l1].ln_1
        if not verbose and len(heads0) >= 10: heads0 = tqdm(heads0)
        for l0, h0 in heads0:
            # if k_comp0 > k_comp0_thld: continue # gpt2-large's 15-4, 17-5, 18-4 > 0.98
            wv0, wo0 = get_head_weights(model, l0, h0, transpose=True)[2:]
            hq0 = wv0 @ wo0 @ wv1 @ wo1
            if heads_1 is not None:
                # l_1, h_1 = heads_1[0]
                # wv_1, wo_1 = get_head_weights(model, l_1, h_1, transpose=True)[2:]
                ln_1 = Id #blocks[heads_1[0][0]].ln_1
                wvo_1 = sum(torch.matmul(*get_head_weights(model, l, h, transpose=True)[2:]) for l, h in heads_1)
                hq0 = wvo_1 @ hq0
            if use_ln: 
                ln0 = blocks[l0].ln_1
                ln1 = ln0 = Id
                # hq = ln1((ln0(_e) @ wv0 @ wo0)) @ wv1 @ wo1 if heads_1 is None else \
                #     ln1((ln0(ln_1(_e) @ wvo_1) @ wv0 @ wo0)) @ wv1 @ wo1
                if heads_1 is None:
                    hq = ln1((ln0(_e) @ wv0 @ wo0)) @ wv1 @ wo1 if isinstance(_e, torch.Tensor) else \
                        rearrange([ln1((ln0(e) @ wv0 @ wo0)) @ wv1 @ wo1 for e in _e[: l0 + 1]], 'l v e -> l v e')
                else:
                    hq = ln1((ln0(ln_1(_e) @ wvo_1) @ wv0 @ wo0)) @ wv1 @ wo1 if isinstance(_e, torch.Tensor) else \
                        rearrange([ln1((ln0(ln_1(e) @ wvo_1) @ wv0 @ wo0)) @ wv1 @ wo1 for e in _e[: l0 + 1]], 'l v e -> l v e')
            for l2, h2 in heads2:
                wq2, wk2 = get_head_weights(model, l2, h2, transpose=True)[:2]
                q0, k0 = hq0 @ wq2, wk2
                eigv_pos0 = get_eigv_pos(k0.T @ q0)
                eigv_pos = eigv_pos0
                if use_ln: 
                    ln2 = Id #blocks[l2].ln_1
                    hk = _e if isinstance(_e, torch.Tensor) else \
                        rearrange(_e[: l0 + 1], 'l v e -> l v e')
                    q, k = ln2(hq) @ wq2, ln2(hk) @ wk2
                    eigv_pos = get_eigv_pos(k.T @ q) if isinstance(_e, torch.Tensor) else \
                        [get_eigv_pos(_k.T @ _q) for _q, _k in zip(q, k)] # lvd->l*vd
                if verbose:
                    # if heads_1 is not None: print(f'{l_1}-{h_1}', end='\t')
                    print(f'{l0}-{h0}, {l1}-{h1}, {l2}-{h2}', eigv_pos0, eigv_pos, eigv_positivity[l2, h2])
                eigv_pos012[(l0, h0, l2, h2)] = eigv_pos0, eigv_pos
    return eigv_pos012

wn2i = OrderedDict(zip('qkvo', range(4)))  # q:0, k:1, v:2, o:3

def weightprod(model, heads, pattern, weBTA=None, BA=True, use_frobenius=False, absorb_ln=False):
    iterables = [head for head in heads if iterable(head)]
    assert len(iterables) == 0, str(iterables)
    if isinstance(pattern, str): pattern = pattern.split()  # e.g. 'vo vo qk' -> ['vo', 'vo', 'qk']
    if pattern[0] in 'eEu':
        assert pattern[-1] == pattern[0] or pattern[-1] == 'u'
        assert len(pattern) - len(heads) == 2, f'len({pattern}) - {len(heads)} != 2'
        assert weBTA is not None
        heads = [weBTA] + heads
        pattern[0] = 'x'; del pattern[-1]
    assert len(heads) == len(pattern), f'{len(heads)} != len({pattern})'
    last_dim = None
    ws = []
    for head, wns in zip(heads, pattern):
        def add_w(w):
            nonlocal last_dim
            if last_dim is not None and w.size(-2) != last_dim:
                w = w.transpose(-2, -1)
                assert w.size(-2) == last_dim, f'{last_dim}, {w.size()}'
            ws.append(w); last_dim = w.size(-1)
        if wns in ['x']:
            add_w(head)
        else:
            weights = get_head_weights(model, *head, absorb_ln=absorb_ln)
            for wn in wns: add_w(weights[wn2i[wn]])
    if use_frobenius:
        assert weBTA is None
        i = -2
        A, B = reduce(torch.matmul, ws[:i]), reduce(torch.matmul, ws[i:])
        B0, B1 = ws[i:]  # A @ B0 @ B1 is much faster than A @ B!
        r1 = (A * B.transpose(-2, -1)).sum(dim=(-2, -1))  # frobenius inner product
        r2 = (A @ B0 @ B1).norm(dim=(-2, -1)) # (A @ B).norm(dim=(-2, -1))
        denorm = A.norm(dim=(-2, -1)) * B.norm(dim=(-2, -1))
        assert r1.size() == r2.size() == denorm.size()
        r1, r2 = r1 / denorm, r2 / denorm
        if r1.ndim == 0: r1, r2 = r1.view(1), r2.view(1)
        return torch.stack([r1, r2], -1) # fro inner product and fro norm, size b2 (b>=1)
    for i in reversed(range(len(ws))): # find the cutting point with local minimal dim, e.g. head_dim
        if ws[i - 1].size(-2) > ws[i].size(-2): break
    A, B = reduce(torch.matmul, ws[:i]), reduce(torch.matmul, ws[i:])
    # print('A:', [w.size() for w in ws[:i]], A.size())
    # print('B:', [w.size() for w in ws[i:]], B.size())
    return B @ A if BA else A @ B

def iweightprod(model, heads, pattern, **kwargs):
    iterables = [head for head in heads if iterable(head)]
    assert len(iterables) == 1, str(iterables)
    if len(iterables) == 1:
        iterable_head = iterables[0]
        iterable_heads = zip(*[head if iterable(head) else cycle([head]) for head in heads])
        # assert len(iterable_head) == len(list(zip(*[head if iterable(head) else cycle([head]) for head in heads])))
        for _head, _heads in zip(iterable_head, iterable_heads):
            assert len([h for h in _heads if iterable(h)]) == 0
            prod = weightprod(model, list(_heads), pattern, **kwargs)
            yield _head, prod

def get_iter_len(heads): return len([head for head in heads if iterable(head)][0])

@torch.no_grad() 
def compute_eigvs(model, heads, pattern, weBTA=None, l1_range_fn=None, **kwargs):
    pattern = pattern.split()
    if pattern[0] in 'eEu':
        assert pattern[-1] == pattern[0] or pattern[-1] == 'u'
        assert len(pattern) - len(heads) == 2, f'len({pattern}) - {len(heads)} != 2'
        assert weBTA is not None
        heads = [weBTA] + heads
        pattern = ['x'] + pattern[1:-1]
    assert pattern[-1] != 'x'

    iterables = [head for head in heads if iterable(head)]
    assert len(iterables) in [1, 2], pattern if isinstance(pattern, str) else ' '.join(pattern)
    if len(iterables) == 1:
        eigvs = torch.Tensor([[get_eigv(prod) for prod in prods.to('cpu')]
            for _, prods in tqdm(iweightprod(model, heads, pattern), total=get_iter_len(heads))])
        return eigvs

    blocks = model.transformer.h
    L, H = len(blocks), blocks[0].attn.num_heads
    if l1_range_fn == None : l1_range_fn = lambda l0: range(l0 + 1, L)
    heads0, heads1 = heads[:-1], heads[-1:]
    pattern0, pattern1 = ' '.join(pattern[:-1]), ' '.join(pattern[-1:])
    assert isinstance(heads0[-1], tuple) or iterable(heads0[-1]), str(heads0[-1])
    assert iterable(heads1[0]), str(heads1[0])
    last_layer0 = heads0[-1][0] if isinstance(heads0[-1], tuple) else None
    eigvs = torch.zeros(L, H, L, H, 2)
    use_frobenius = kwargs.get('use_frobenius', False)
    if use_frobenius: eigvs = eigvs.to(blocks[0].attn.k_proj.weight.device) # cuda
    # itotal0 = len([head for head in heads0 if iterable(head)][0])
    for (l0, h0), prod0 in tqdm(iweightprod(model, heads0, pattern0, BA=False), total=get_iter_len(heads0)):
        last_layer = l0 if last_layer0 is None else last_layer0
        l1_range = l1_range_fn(last_layer)
        ihead1 = [(l1, None) for l1 in l1_range] # layer by layer for fast computation
        _, prods = zip(*list(iweightprod(model, [prod0] + [ihead1], 'x ' + pattern1, **kwargs))) # generator -> list
        prods = torch.cat(prods)
        if not use_frobenius: prods = prods.to(eigvs.device) # compute eigv on gpu is slower, so move to cpu first
        ihead1 = list(product(l1_range, range(H)))  # head by head
        assert len(ihead1) == len(prods), f'{len(ihead1)} != {prods.size()}[0]'
        for (l1, h1), prod in zip(ihead1, prods):
            if (l1, h1) in heads1[0]:
                if prod.ndim == 2 and prod.size(0) == prod.size(1):
                    eigvs[l0, h0, l1, h1] = torch.Tensor(get_eigv(prod)) # (positivity, reality)
                else:  # use_frobenius
                    assert prod.size() == torch.Size([2])
                    eigvs[l0, h0, l1, h1] = prod # (fro inner product, fro norm)
    if use_frobenius: eigvs = eigvs.to('cpu')
    return eigvs

def T(input): return input.transpose(-2, -1)

def is_tril_empty(a):
    assert a.ndim in [2, 3] and a.size(-2) == a.size(-1)  # (ln)(ln) or k(ln)(ln)
    tril_mask = torch.ones(a.size()[-2:]).tril() - torch.eye(a.size(-1))
    return (a[(torch.ones_like(a) * tril_mask).bool()] == 0).float().mean() > 0.6

def complete_tril(a):
    if a.ndim in [2, 3]:  # (ln)(ln) or k(ln)(ln)
        assert a.size(-2) == a.size(-1)
        return a - a.tril() + T(a).tril() if is_tril_empty(a) else a
    assert a.ndim == 5, str(a.size())  # lnln2
    L, H = a.size()[:2]
    return rearrange(complete_tril(rearrange(a, 'l0 n0 l1 n1 k -> k (l0 n0) (l1 n1)')),
            'k (l0 n0) (l1 n1) -> l0 n0 l1 n1 k', l0=L, n0=H, l1=L, n1=H)
   
def get_conductivity(eigv_positivity012, l1, h1, plot=False, figsize=(5, 2)):
    x = eigv_positivity012.get((l1, h1))
    if x is None: return 0.
    if plot: plt.figure(figsize=figsize); plt.hist(x, 20); _ = plt.title(f'{l1}-{h1}'); plt.show()
    return np.abs(np.array(x)).mean()

def get_positional_score(model, qlen=128, offset=-1, substract_null_attn=True):
    attentions = forward(model, torch.randint(100, 25000, size=(1, qlen))).attentions
    attentions = torch.cat(attentions)  # l*1nij->lnij
    null_attn = attentions[:, :, abs(offset):, 0].clone()  # l n i-1
    if substract_null_attn: null_attn[:, :, 0] = 0
    else: null_attn[:] = 0
    return (attentions.diagonal(offset=offset, dim1=-2, dim2=-1) / (1 - null_attn)).mean(-1)

def plot_k_comp(heads, k_compositions, pos_heads2val):
    ls, hs = zip(*heads)
    _ = plt.figure(figsize=(20, len(heads) * 0.5))
    _ = sns.heatmap(torch.cat([k_compositions[list(ls), list(hs)], torch.Tensor(list(pos_heads2val.values())).unsqueeze(0)]), cbar=False,
        xticklabels=[f'{l}-{h}' for l, h in pos_heads2val.keys()], yticklabels=[f'{l}-{h}' for l, h in heads])

def add_cols(head_tuples, attr_dicts):
    if not isinstance(attr_dicts, (tuple, list)): attr_dicts = [attr_dicts]
    return [[l, h, *v] + [attr_dict[l, h] for attr_dict in attr_dicts] for l, h, *v in head_tuples]

def add_rows(head_rows, attr_dicts):
    if not isinstance(attr_dicts, (tuple, list)): attr_dicts = [attr_dicts]
    return head_rows + tuple([np.array([attr_dict.numpy()[l, h] for l, h, _ in zip(*head_rows)])
                            for attr_dict in attr_dicts])

def embed_by_pairwise_sim_matrix(sim):
    distance = 1 - sim  # 1 - [-1, 1] = [2, 0]. like cosine distance
    _ = distance.fill_diagonal_(0)
    distance = distance.clamp(min=0)
    # distance = distance[H:, H:]  # remove layer 0
    # algo = MDS(n_components=2, dissimilarity='precomputed')
    algo = TSNE(n_components=2, metric='precomputed', square_distances=True)
    # squared in TSNE._fit to recovery cosine distance:
    # https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/manifold/_t_sne.py#L917-L918
    # TSNE uses squared euclidean distance which is proportional to the cosine distance when data is L2 normalized:
    # https://stackoverflow.com/questions/36545434/cosine-similarity-tsne-in-sklearn-manifold (search "squared" and "square root")
    # https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance
    distance = distance ** .5 
    result = algo.fit_transform(distance)
    return result

def small_to_big_pairwise_sim_tensor(t00, t01, t11):
    def T(eigvs): return rearrange(eigvs, 'l0 n0 l1 n1 k -> l1 n1 l0 n0 k')
    return rearrange([
        rearrange([t00, t01], 'b l0 n0 l1 n1 k -> l0 n0 b l1 n1 k'),
        rearrange([T(t01), t11], 'b l0 n0 l1 n1 k -> l0 n0 b l1 n1 k')
    ], 'B l0 n0 b l1 n1 k -> B l0 n0 b l1 n1 k')

def embed_by_pairwise_sim_tensor(sim4, mask):
    # einsum is the simpliest way of doing this
    _sim4 = sim4[torch.einsum('akg,blh->akgblh', mask, mask)]  # bLNbLN->(blnbln), b=2
    lh = int(math.sqrt(_sim4.size(0)))
    _sim4 = _sim4.view(lh, lh) # (bln) (bln)

    _result = embed_by_pairwise_sim_matrix(_sim4) # bln2
    result = torch.ones(sim4.size()[:3] + (2,)) * float('nan')  # bLN2
    result[mask] = torch.Tensor(_result)
    return result

def get_knn(pairwise_sim, heads, src_left=True, topk=10, sim_threshold=0.9):
    if isinstance(heads, tuple): heads = [heads]  # (2, 11) -> [(2, 11)]
    knn = {}
    for l0, h0 in heads:
        neighbours = pairwise_sim[l0, h0, :, :] if src_left else pairwise_sim[:, :, l0, h0]
        for l1, h1, sim in topk_md(neighbours, k=topk, transpose=True):
            if sim >= sim_threshold: knn[(l1, h1)] = max(knn.get((l1, h1), -1), sim)
    return knn

def rescale(a, threshold, upper_bound=1): return (a - threshold) * 1. / (upper_bound - threshold)

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

def plot_attn(attn, tokens, ytokens=None, ystart=None, ystop=None, y_pos=None, x_pos=None,
            use_imshow=False, annot=False, figsize=(10, 10), ax=None):
    ytokens = ytokens or tokens
    if ystart is not None:
        ystop = ystop or attn.size(0)
        attn = attn[ystart: ystop]
        ytokens = ytokens[ystart: ystop]
        y_pos = [p - ystart for p in y_pos]
    square = attn.size(0) == attn.size(1)
    if not square:
        figsize2 = (attn.size(1), attn.size(0))
        a = min(s1 / s2 for s1, s2 in zip(figsize, figsize2))
        figsize = [s * a for s in figsize2]
    # ytokens = tokens if square else [str(i) for i in range(attn.size(0))]
    if ax is None: plt.figure(figsize=figsize)
    if use_imshow:
        ax.imshow(attn)#, cmap='hot')
        ax.set_xticks(np.arange(0, attn.size(1), 1)); ax.set_xticklabels(tokens)
        ax.set_yticks(np.arange(0, attn.size(0), 1)); ax.set_yticklabels(ytokens)
    else:
        kwargs = dict(linewidths=0.1, linecolor='grey') if y_pos is None else {}
        res = sns.heatmap(numpy(attn), square=square, cbar=False, annot=annot, fmt='d', 
            xticklabels=tokens, yticklabels=ytokens, ax=ax, **kwargs)
    kwargs = dict(linewidths=0.5, color='grey')
    if y_pos is not None: ax.hlines(y=y_pos, xmin=0, xmax=attn.size(1), **kwargs)  # max-0.5 for imshow
    if x_pos is not None: ax.vlines(x=x_pos, ymin=0, ymax=attn.size(0), **kwargs)
    # _ = res.set_xticklabels(res.get_xmajorticklabels(), fontsize=10, rotation=0)
    # _ = res.set_yticklabels(res.get_ymajorticklabels(), fontsize=10, rotation=0)
    # res.tick_params(top=True, right=True, labeltop=True, labelright=True) # cause x3 slowdown!
    # plt.show()

def cluster(emb, labels, n_clusters=3):
    assert emb.shape[0] == labels.shape[0], '%d ！= %d' % (emb.shape[0], labels.shape[0])
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

# def get_head_weights(layer, head):
#     m = get_attn_module(blocks[layer])
#     wq = m.q_proj.weight.view(H, -1, hidden_size)[head]
#     wk = m.k_proj.weight.view(H, -1, hidden_size)[head]
#     wv = m.v_proj.weight.view(H, -1, hidden_size)[head]
#     wo = m.out_proj.weight.view(hidden_size, H, -1)[:, head]
# #     return wq, wk, wv, wo
#     return wq.t(), wk, wv.t(), wo.t()

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