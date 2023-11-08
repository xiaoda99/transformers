import sys
import os
import time
from collections import OrderedDict, defaultdict
# from typing import Iterable
from collections.abc import Iterable
import numpy as np
import math
import random
import dataclasses
from dataclasses import dataclass, field, fields
from copy import deepcopy
from functools import reduce, partial
from itertools import chain, product, combinations, cycle, groupby
import types
from tqdm import tqdm
from pprint import pprint
import pickle
import gzip
import seaborn as sns
import matplotlib.pyplot as plt
# from pt_model_recorder import BatchShareMemRecorder
# from sklearn.manifold import TSNE, MDS
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F 

import einops
from einops import rearrange

from LLAMATokenizer import LLAMATokenizer
from transformers import LlamaTokenizer, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention, GPTNeoBlock
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJModel
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXModel, GPTNeoXForCausalLM
from transformers.models.xglm.modeling_xglm import XGLMForCausalLM, XGLMAttention, XGLMDecoderLayer
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaDecoderLayer, LlamaModel

sys.path.insert(0, './pptree')
from pptree import Node, print_tree

from common_utils import numpy, einsum, my_isinstance, convert_ids_to_tokens, show_topk, topk_md, topi_md, \
    equal, join_lists, iterable, pad, Timer, maybe_map, reduce_objects, mr, maybe_mr, list_get, fn2str, Printer, lget

from child_utils_nrk_2 import make_data_tuple, get_answer_index, generate
from weight_analysis import get_head_weights

@dataclass
class Outputs:
    inputs_embeds: torch.FloatTensor = None
    position_embeds: torch.FloatTensor = None
    attn_outputs: tuple = ()
    values: tuple = ()
    attn_outs: tuple = ()
    head_inputs: tuple = ()
    head_outputs: tuple = ()
    intermediates: tuple = ()
    mlp_outputs: tuple = ()
    ln_states: tuple = ()
    hidden_states: tuple = ()
    attentions: tuple = ()
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None
    loss: torch.FloatTensor = None
    attn_attr: OrderedDict = field(default_factory=OrderedDict)

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
    H: int = None
    label_type: str = None
    attn_pattern: str = None
    attribute_k: bool = False
    attr: Attributions = None  # for children
    logit_attr: Attributions = None  # for children
    ap_scores: dict = field(default_factory=dict)  # for root
    # head_label_types: set = field(default_factory=set)  # for root
    attributed_heads: set = field(default_factory=set)  # for root
    attr_ap_scores: dict = None  # for children
    ap_score: float = None  # for self
    attr_ap_score: float = None  # for self
    icl_score: float = None  # for self
    top_score: list = None  # for self
    sensitivity: float = None  # for self
    _attn_pattern: str = None  # for sorting
    _attr_ap_score: float = None  # for sorting
    top_heads: list = None
    dummy: bool = False
    mixed: bool = False

@dataclass
class Result:
    task: tuple = None
    trans_args: dict = None
    gen_args: dict = None
    all_examples: list = None
    texts: list = None
    all_bos_tokens: list = None
    data_tuples: list = None
    root: Node = None
    node: Node = None
    mean_loss: float = None
    mean_acc: float = None

def result2dict(result):
    return {k: v for k, v in result.__dict__.items() if k not in ['task', 'data_tuples']}

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

name2mapping = {
    'vicuna': dict(model_name='model', emb_name='embed_tokens', layer_name='layers', ln_f_name='norm', n_layers=60),
    'gpt-j-6B': dict(n_layers=28),
    'gpt-neox': dict(model_name='gpt_neox', emb_name='embed_in', layer_name='layers', ln_f_name='final_layer_norm', head_name='embed_out', n_layers=44),
}

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
    elif my_isinstance(model, LlamaForCausalLM):
        model.transformer = model.model
        model.transformer.wte = model.transformer.embed_tokens
        model.transformer.h = model.transformer.layers
        model.transformer.ln_f = model.transformer.norm
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
        elif my_isinstance(block, LlamaDecoderLayer):
            block.attn = block.self_attn
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
            else:#if not hasattr(self, 'q_proj'):
                weight_chunks = self.query_key_value.weight.view(  # (n3d)e->n(3d)e->3*[nde]
                    self.num_heads, 3 * self.head_dim, self.embed_dim).chunk(3, dim=1)
                bias_chunks = self.query_key_value.bias.view( # n3d->n(3d)->3*[nd]
                    self.num_heads, 3 * self.head_dim).chunk(3, dim=1)
                for proj_name, w, b in zip(['q_proj', 'k_proj', 'v_proj'], weight_chunks, bias_chunks):
                    proj = nn.Linear(self.embed_dim, self.embed_dim)
                    proj.weight = nn.Parameter(rearrange(w, 'n d e -> (n d) e'))
                    proj.bias = nn.Parameter(rearrange(b, 'n d -> (n d)'))
                    setattr(self, proj_name, proj)
        elif my_isinstance(self, LlamaAttention):
            device = self.q_proj.weight.device
            max_positions = model.config.max_position_embeddings
            self.register_buffer(
                "bias",
                torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                    1, 1, max_positions, max_positions
                ).to(device),
                persistent=False,
            )
            self.register_buffer("masked_bias", torch.tensor(-1e9).to(device), persistent=False)

            self.attn_dropout = nn.Dropout(0)
            self.resid_dropout = nn.Dropout(0)
            for old_name, new_name in [('hidden_size', 'embed_dim'), ('o_proj', 'out_proj')]:
                setattr(self, new_name, getattr(self, old_name))

def unify_tokenizer(tokenizer):
    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.newline_token = '<0x0A>'
        tokenizer.newword_prefix = '▁'
    elif isinstance(tokenizer, GPT2Tokenizer):
        tokenizer.newline_token = 'Ċ'
        tokenizer.newwrod_prefix = 'Ġ'

def get_device_map(model_name='transformer', emb_name='wte', layer_name='h', n_layers=60,
                   ln_f_name='ln_f', head_name='lm_head', devices=[0, 1]):
    if len(devices) == 1: return {model_name: devices[0], head_name: devices[0]}
    dm = {f'{model_name}.{emb_name}': devices[0]}
    if len(devices) == 2:
        for i in range(n_layers // 2): dm[f'{model_name}.{layer_name}.{i}'] = devices[0]
        for i in range(n_layers // 2, n_layers): dm[f'{model_name}.{layer_name}.{i}'] = devices[1]
        dm[f'{model_name}.{ln_f_name}'] = devices[1]
        dm[f'{head_name}'] = devices[1]
    elif len(devices) == 3:
        for i in range(n_layers // 3): dm[f'{model_name}.{layer_name}.{i}'] = devices[0]
        for i in range(n_layers // 3, n_layers // 3 * 2): dm[f'{model_name}.{layer_name}.{i}'] = devices[1]
        for i in range(n_layers // 3 * 2, n_layers): dm[f'{model_name}.{layer_name}.{i}'] = devices[2]
        dm[f'{model_name}.{ln_f_name}'] = devices[2]
        dm[f'{head_name}'] = devices[2]
    return dm

def name_with_device(name, device):
    return name + '_' + str(device).replace(':', '')  # '_cuda0' or '_cpu'

def getattr_on_device(obj, name, device):
    name_w_dev = name_with_device(name, device)
    if hasattr(obj, name_w_dev): return getattr(obj, name_w_dev)
    return getattr(obj, name) if str(device) == 'cpu' else None

def clone_module_to(module, name, device, dtype=None, remove_on_redo=True, switch=False, gpu_module=None):
    if dtype is None: dtype = torch.float32 if str(device) == 'cpu' else torch.float16
    if hasattr(module, name_with_device(name, device)):
        # for i in range(10):
        #     name_w_dev = name_with_device(name, torch.device(f'cuda:{i}'))
        #     if hasattr(module, name_w_dev): delattr(module, name_w_dev)
        if remove_on_redo: delattr(module, name_with_device(name, device)); return module
    else:    
        with Timer(f"cloning {module.__class__.__name__}{getattr(module, 'layer', '')}.{name} to {device}"):
            getattr(module, name).requires_grad_(False)  # don't know if useful or not for saving mem
            setattr(module, name_with_device(name, device), deepcopy(getattr(module, name)).to(device, dtype=dtype)
                if gpu_module is None else getattr(gpu_module, name))
            setattr(module, name_with_device(name, 'cpu'), getattr(module, name))
    if switch: setattr(module, name, getattr_on_device(module, name, device))
    return module

def switch_module_to(module, name, device):
    setattr(module, name, getattr_on_device(module, name, device))

def mem_usage(device, unit=1024 * 1024, cxt_mem=654):  # obtained by running temp = torch.Tensor([1.0]).to('cuda:0')
    mems = [n // unit for n in [torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device,)]]
    return mems + [mems[1] + cxt_mem]

def clone_model_to(model, device, dtype=None):
    assert model.device == torch.device('cpu'), f'Only CPU model can be cloned to GPU, but model is already on {model.device}.'
    assert not hasattr(model, 'lm_head_cpu'), 'model already cloned!'
    if dtype is None: dtype = torch.float32 if str(device) == 'cpu' else torch.float16
    mu = mem_usage(device)
    for i, block in enumerate(model.transformer.h):
        clone_module_to(block, 'ln_1', device, dtype=dtype)
        clone_module_to(block.attn, 'out_proj', device, dtype=dtype)
    clone_module_to(model.transformer, 'ln_f', device, dtype=dtype)
    # keep lm_head in fp32 to improve accuracy of logits computation, same as attn_logits. May be unnecessary?
    clone_module_to(model, 'lm_head', device, dtype=dtype)  # torch.float32
    print(f'mem_usage before / after clone_model_to: {mu} / {mem_usage(device)}')
    return model

def switch_model_to(model, device):
    for i, block in enumerate(model.transformer.h):
        setattr(block, 'ln_1', getattr_on_device(block, 'ln_1', device))
        setattr(block.attn, 'out_proj', getattr_on_device(block.attn, 'out_proj', device))
    switch_module_to(model.transformer, 'ln_f', device)
    switch_module_to(model, 'lm_head', device)

def scaled_ln(ln, x, mean=None, std=None, scaled=True):
    # if not scaled: return ln(x), None
    self = ln
    state = {}
    if mean is None:
        if hasattr(self, 'bias'):
            mean = x.mean(dim=-1, keepdim=True)
            state['mean'] = mean.detach()
            # if scaled: mean = mean[-1:].detach()  # # use the mean of the last original (unscaled) example in the batch or not?
        else:
            mean = torch.zeros(1).to(x.device) # LlamaRMSNorm
    if std is None:
        var = ((x - mean).to(dtype=torch.float32) ** 2).mean(dim=-1, keepdim=True)  # to prevent **2 overflow in fp16
        if not hasattr(self, 'eps'): self.eps = self.variance_epsilon  # LlamaRMSNorm
        std = (var + self.eps).sqrt()#.to(dtype=x.dtype)
        state['std'] = std.detach()
        if scaled: std = std[-1:].detach()  # use the std of the last original (unscaled) example in the batch
    y = ((x - mean.to(x.device)) / std.to(x.device)).to(dtype=x.dtype) * self.weight
    if hasattr(self, 'bias'): y = y + self.bias
    return y, state

def custom_ln(ln, x, mean=None, std=None, bias=True):
    self = ln
    if mean is None:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
    y = (x - mean) / std * self.weight + self.bias if bias else x / std * self.weight
    return y, mean.detach(), std.detach()

def embed_forward(transformer, inputs, output_embeds=True):
    self = transformer
    input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs.input_ids
    hidden_states = inputs_embeds = self.wte(input_ids)
    position_embeds = None
    if not my_isinstance(transformer, (GPTJModel, GPTNeoXModel, LlamaModel)):
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

def fixed_pos_embedding(x, seq_len=None):
    if seq_len is None: seq_len = x.shape[2]
    self = fixed_pos_embedding; device = x.device
    self.sin, self.cos = {}, {}
    if device not in self.sin: #not hasattr(self, 'sin'):
        dim = x.shape[-1]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(2048), inv_freq).to(device).float()
        self.sin[device], self.cos[device] = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
    return self.sin[device][:seq_len], self.cos[device][:seq_len]

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

def apply_rotary_pos_emb(query_rot, key_rot, offset=0, rotary_emb=None):
    sincos_fn, apply_fn = (fixed_pos_embedding, apply_rotary_pos_emb_every_two) \
        if rotary_emb is None else (rotary_emb, apply_rotary_pos_emb_half)  # gpt-neox
    sincos = sincos_fn(key_rot, seq_len=key_rot.size(2))  # (bnjd')[2]->j
    if rotary_emb is not None:
        cos, sin = sincos
        # GPTNeoXRotaryEmbedding already moves to device but LlamaRotaryEmbedding doesn't, so move here 
        sincos = (sin.to(query_rot.device), cos.to(query_rot.device))
    key_rot = apply_fn(key_rot, sincos, offset=offset)
    query_rot = apply_fn(query_rot, sincos, offset=offset)
    return query_rot, key_rot

def bmm_attn_scores(self, query, key):  # adapted from modeling_gpt_neox.py
    query_batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_batch_size, _, key_length, _ = key.size()
    batch_size = query_batch_size
    if query_batch_size != key_batch_size:
        assert query_batch_size == 1 or key_batch_size == 1  # the larger one is scaled for attribution
        batch_size = max(query_batch_size, key_batch_size)
        if key_batch_size == 1: key = key.expand(batch_size, -1, -1, -1)
        if query_batch_size == 1: query = query.expand(batch_size, -1, -1, -1)
    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    attn_scores = torch.zeros(batch_size * num_attention_heads, query_length, key_length,
        dtype=query.dtype, device=key.device)
    attn_scores = torch.baddbmm(attn_scores, query, key.transpose(1, 2), beta=1.0,
        alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor))
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
    return attn_scores

def _attn(self, query, key, value, attention_mask=None):
    if my_isinstance(self, (GPTNeoSelfAttention, GPTJAttention)):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query, key = query.to(torch.float32), key.to(torch.float32)
    attn_scores = bmm_attn_scores(self, query, key) if my_isinstance(self, GPTNeoXAttention) \
        else torch.matmul(query, key.transpose(-1, -2)) # bnid,bnjd->bnij

    # turns out gptneo fold_scaling_into_initializer, and uses float32_logits. 
    # see https://crfm.stanford.edu/2021/08/26/mistral.html (Diagnosing Numerical Instability, Eureka!)
    # and https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/attention.py#L517&66
    if not my_isinstance(self, (GPTNeoSelfAttention, GPTNeoXAttention)):
        attn_scores = attn_scores / (query.size(-1) ** 0.5)

    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
    # self.masked_bias can not be used because it will turn to -inf when casting model to fp16
    # attn_scores = torch.where(causal_mask, attn_scores, self.masked_bias.to(attn_scores.dtype))
    # adapted from GPTJAttention._attn which uses torch.finfo(attn_scores.dtype).min
    mask_value = -1e9 if attn_scores.dtype == torch.float32 else -1e4 # torch.finfo(attn_scores.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_scores = torch.where(causal_mask.to(attn_scores.device), attn_scores, mask_value)
    if attention_mask is not None: attn_scores = attn_scores + attention_mask
    if value is None: return None, attn_scores

    attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype) \
        if my_isinstance(self, LlamaAttention) else nn.functional.softmax(attn_scores, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value) # bnij,bnjd->bnid
    # self.query2, self.key2, self.value2, self.attn_weights2, self.attn_output2 = query.to('cpu').float(), key.to('cpu').float(), value.to('cpu').float(), attn_weights.to('cpu').float(), attn_output.to('cpu').float()  # XD debug
    return attn_output, attn_weights

def backup_heads(self):
    self.backup_names = ['num_heads', 'q_proj', 'k_proj', 'v_proj', 'out_proj']
    for name in self.backup_names: setattr(self, name + '0', getattr(self, name))

def restore_heads(self):
    # assert self.num_heads0 == 16, f'{self.num_heads0} != 16'  # for gpt-j
    for name in self.backup_names:
        setattr(self, name , getattr(self, name + '0'))
        # delattr(self, name + '0')

def trim_heads(self, kept_heads, proj_names=None, new_weight=False, device='cpu'):
    if proj_names is None: proj_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    backup_heads(self)
    for proj_name in proj_names:
        if str(device) != 'cpu':
            proj_name_dev = (f"{proj_name}_{'_'.join(str(i) for i in kept_heads)}"
                            f"_{str(device).replace(':', '')}")  # e.g. q_proj_12_cuda7
            if hasattr(self, proj_name_dev):
                setattr(self, proj_name, getattr(self, proj_name_dev))
                continue
        weight = getattr(self, proj_name).weight
        bias = getattr(self, proj_name).bias
        if proj_name == 'out_proj':
            patterns = ['e (n d) -> n d e', 'n d e -> e (n d)']
            size = (self.head_dim * len(kept_heads), self.embed_dim)
        else:
            patterns = ['(n d) e -> n d e', 'n d e -> (n d) e']
            size = (self.embed_dim, self.head_dim * len(kept_heads))
        weight = rearrange(weight, patterns[0], n=self.num_heads)[kept_heads] # kde
        weight = rearrange(weight, patterns[1]).detach()
        if proj_name != 'out_proj' and bias is not None: # out_proj.bias is ignored
            bias = rearrange(bias, '(n d) -> n d', n=self.num_heads)[kept_heads]
            bias = rearrange(bias, 'n d -> (n d)').detach()
        if new_weight: weight = weight.clone().detach()
        proj = nn.Linear(*size, bias=bias is not None)
        proj.weight = nn.Parameter(to(weight, device))
        if bias is not None: proj.bias = nn.Parameter(to(bias, device))
        setattr(self, proj_name, proj)
        if str(device) != 'cpu': setattr(self, proj_name_dev, proj)
    self.num_heads = len(kept_heads)

def compose_head(model, attn_mod, composed_head):
    qk_head, ov_head = composed_head
    wq, wk = get_head_weights(model, *qk_head, transpose=False)[:2] \
        if isinstance(qk_head, tuple) else (None, None)
    wv, wo = get_head_weights(model, *ov_head, transpose=False)[2:] \
        if isinstance(ov_head, tuple) else (None, None)
    
    for proj_name, w in zip(['q', 'k', 'v', 'out'], [wq, wk, wv, wo]):
        if w is None:
            proj = None  # use attn_pattern as qk
        else:
            proj = nn.Linear(*w.size(), bias=False)
            proj.weight = nn.Parameter(w)
        setattr(attn_mod, f'{proj_name}_proj', proj)
    attn_mod.num_heads = 1

def remove_composed_heads(attn_mod):
    try_delattr(attn_mod, 'composed_heads')
    # for proj_name in ['q', 'k', 'v', 'out']:
    #     try_delattr(attn_mod, f'composed_{proj_name}_proj')

def get_all_composed_heads(model):
    return [(i, block.attn.composed_heads) for i, block in enumerate(model.transformer.h)
        if getattr(block.attn, 'composed_heads', None) is not None]

def composed_heads2str(model):
    if not hasattr(model, 'transformer'): return ''  # for ununified models, e.g. llama
    all_composed_heads = join_lists([composed_heads for l, composed_heads in get_all_composed_heads(model)])
    d = defaultdict(list)
    for qk_head, ov_head in all_composed_heads: d[qk_head].append(ov_head)
    
    def head2str(head): return f'{head[0]}-{head[1]}' if type(head) != str else abbreviate_attn_pattern(head)
    s = '_'.join(head2str(qk_head) + ':' + ','.join(head2str(h) for h in ov_heads) for qk_head, ov_heads in d.items())
    if s != '': s = '_' + s
    return s

def scale_to(tensor, target):
    assert tensor.size() == target.size(), f'{tensor.size()} != {target.size()}'
    tensor = tensor * (target.norm(dim=-1) / tensor.norm(dim=-1)).unsqueeze(-1)
    assert torch.allclose(tensor.norm(dim=-1), target.norm(dim=-1))  # debug
    return tensor

def maybe_composed_attn_forward(attn_fwd_outputs, model, block, hq, hk, hv, by_head=None,
                                ranges=None, reuse_attn_weights=True, **kwargs):
    self = block.attn
    composed_heads = getattr(self, 'composed_heads', None)
    if composed_heads: by_head = list(set((by_head or []) +  ['head_output']))
    attn_output, aw, value, attn_out, head_input, head_output = attn_forward(
        block, hq, hk, hv, by_head=by_head, **kwargs)
    
    if composed_heads:
        backup_heads(self)
        head_output_sum0 = head_output.sum(1)
        for (qk_head, ov_head), range_i in zip(composed_heads, self.ranges_i):
            try:
                compose_head(model, self, (qk_head, ov_head))
                assert block.layer in [qk_head[0], ov_head[0]], \
                    f'{block.layer} not in {[qk_head[0], ov_head[0]]}'
                is_relating_head = qk_head[0] == block.layer

                attn_size = aw.size()[-2:]
                all_attentions, *_ = zip(*attn_fwd_outputs)
                aw2 = None
                if not is_relating_head and reuse_attn_weights:
                    aw2 = attn_pattern2labels(ranges, qk_head, attn_size)[0] \
                        if isinstance(qk_head, str) else all_attentions[qk_head[0]][0, qk_head[1]]
                    aw2 = rearrange(aw2, 'i j -> 1 1 i j')  # bmij

                _, aw2, value2, attn_out2, head_input2, head_output2 = attn_forward(
                    block, hq, hk, hv, by_head=by_head, attn_weights=aw2)

                assert range_i.endswith('->*')
                if isinstance(qk_head, str):
                    assert qk_head.split('->')[0] == range_i.split('->')[0], \
                        f'{qk_head}.q != {range_i}.q'
                range_i = attn_pattern2labels(ranges, range_i, attn_size)[0][:, 0] != 0  # ij-i
                h = qk_head[1] if is_relating_head else ov_head[1]
                i = 0
                aw[:, h, range_i] = aw2[:, i, range_i]  # bnij->brj = bmij->brj
                head_output[:, h, range_i] = head_output[:, h, range_i] * int(is_relating_head) + \
                    head_output2[:, i, range_i]  # bnie->bre = bmie->bre
                if value is not None:
                    pass # TODO: deal with value which is also used in sum_forward for attribution
                if attn_out is not None:
                    attn_out[:, h, range_i] = attn_out[:, h, range_i] * int(is_relating_head) + \
                        attn_out2[:, i, range_i]  # bnid->bre = bmid->brd
            finally: restore_heads(self)
        attn_output = attn_output + head_output.sum(1) - head_output_sum0  # bnie->bie
    return attn_output, aw, value, attn_out, head_input, head_output

def attn_forward(block, hq, hk, hv, by_head=None,
                attention_mask=None, head_mask=None, attn_weights=None):
    self = block.attn  # block.attn.attention already renamed
    query, key, value = None, None, None
    if hasattr(self, 'qkv_proj'):  # gpt-neox int8
        qkv = self.qkv_proj(hq)  # hq == hk == hv
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_dim)
        query, key, value = [rearrange(a, 'b i n d -> b n i d') # [bind] * 3 -> [bnid] * 3
            for a in qkv.view(*new_qkv_shape).chunk(3, dim=-1)] # bin(3d) -> [bind] * 3
        if hv is None: value = None
    else:
        if hq is not None and hk is not None and attn_weights is None:
            key = self.k_proj(hk)
            key = _split_heads(key, self.num_heads, self.head_dim)  # bnid
            if hq.ndim == 3:  # bie
                query = self.q_proj(hq)
                query = _split_heads(query, self.num_heads, self.head_dim) # bidi
            else:
                assert hq.ndim == 4  # bnid, computed in cat_attn_forward
                # if rotary: hq = rearrange(hq, 'b n i d -> b i n d')
                assert hq.size()[1:] == key.size()[1:], f'{hq.size()} != {key.size()}'
                query = hq
        if hv is not None: value = _split_heads(self.v_proj(hv), self.num_heads, self.head_dim)
    rotary = my_isinstance(self, (GPTJAttention, GPTNeoXAttention, LlamaAttention))
    if rotary and query is not None and key is not None:
        if hasattr(self, 'rotary_dim'):
            k_rot = key[:, :, :, : self.rotary_dim]
            q_rot = query[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]
            q_pass = query[:, :, :, self.rotary_dim :]
        else:  # LlamaAttention
            k_rot, q_rot = key, query

        rotary_emb=self.rotary_emb if my_isinstance(self, (GPTNeoXAttention, LlamaAttention)) else None
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, rotary_emb=rotary_emb)

        if hasattr(self, 'rotary_dim'):
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key, query = k_rot, q_rot
    attn_out, attn_weights = _attn(self, query, key, value, attention_mask) \
        if attn_weights is None else (attn_weights.to(value.device) @ value, attn_weights)  # XD
    if value is None: return None, attn_weights, None, None, None, None

    if head_mask is not None: attn_out = einsum('bnid,bni->bnid', attn_out, head_mask)

    attn_output = _merge_heads(attn_out, self.num_heads, self.head_dim) # bnid->bi(nd)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    head_input, head_output = None, None
    if by_head and ('head_input' in by_head or 'head_output' in by_head):
        w_o = self.out_proj.weight.data
        do_bmm = True # not (any('0' in s for s in by_head) or w_o.dtype == torch.int8)
        if w_o.dtype == torch.int8: w_o = self.out_proj.weight.data0  # float16 weight on cpu
        if do_bmm: w_o = rearrange(w_o, 'e (n d) -> n d e', n=self.num_heads)
        else: f = partial(get_head_io, num_heads=self.num_heads)
        # for smaller models (e.g. gpt-j) on gpu, @ then to-cpu may be faster than to-cpu then @
        if 'head_output' in by_head: #any('head_output' in s for s in by_head):
            # head_output = attn_output.to('cpu').float() @ w_o.to('cpu').float() if do_bmm else (f, attn_output.to('cpu').float(), w_o)
            head_output = attn_out @ w_o  # bnid,nde->bnie
        if 'head_input' in by_head: #any('head_input' in s for s in by_head):
            # head_input = value.to('cpu').float() @ w_o.to('cpu').float() if do_bmm else (f, value.to('cpu').float(), w_o.float())
            head_input = value @ w_o  # bnid,nde->bnie
    if not by_head or 'value' not in by_head: value = None
    if not by_head or 'attn_out' not in by_head: attn_out = None
    return attn_output, attn_weights, value, attn_out, head_input, head_output

def head_forward(model, hidden_states, layer, head, ln_state=None, logits_mask=None, lnf_state=None, labels=None, loss_reduction=None,
            attn_weights=None, attn_mask=None, attn_labels=None, hidden_states0=None, attribute_k=False, trim=True, scaled=True):
    assert not isinstance(layer, Iterable)  # should use mixed_forward, which is clearer and simpler
    block = model.transformer.h[layer]
    # only hq and hv can be scaled, not hk
    hk = block.ln_1(hidden_states0 if hidden_states0 is not None else hidden_states)
    h, _ = scaled_ln(block.ln_1, hidden_states, scaled=scaled, **(ln_state or {})) #if scaled else block.ln_1(hidden_states)
    by_head = None
    if attn_weights is not None:
        hq, hv = None, h
        by_head = ['head_output']
    elif attn_labels is not None:
        hq, hv = h, None # return attn_logits instead of attn_weights by passing None hv 
        if attribute_k: hq, hk = hk, hq

    self = block.attn
    if trim:
        try:
            trim_heads(self, [head], device=model.lm_head.weight.device)
            if attn_weights is not None: attn_weights = attn_weights[:, [head]] # bnij->b1ij
            _, attn_logits, *_, head_output = attn_forward(block, hq, hk, hv,
                attn_weights=attn_weights, by_head=by_head)
            if head_output is not None: assert head_output.size(1) == 1, str(head_output.size())
            head = 0  # used to index head_output and attn_logits
        finally: restore_heads(self)
    else:
        _, attn_logits, *_, head_output = attn_forward(block, hq, hk, hv,
            attn_weights=attn_weights, by_head=by_head)
    if head_output is not None: head_output = head_output[:, head]
    logits, loss = None, None
    if labels is not None:
        logits, loss = lm_head_forward(model, head_output, logits_mask=logits_mask, lnf_state=lnf_state,
            labels=labels, loss_reduction=loss_reduction, scaled=scaled)
    elif attn_labels is not None:
        logits = attn_logits[:, head]  # bnij->bij
        if attn_mask is not None:
            # mask_value = torch.finfo(logits.dtype).min
            mask_value = -1e9 if logits.dtype == attn_mask.dtype == torch.float32 else -1e4  # else shoud not happen for gpt-j. see _attn()
            logits = logits + attn_mask * mask_value
        logprobs = logits.log_softmax(-1) if not attribute_k else logits
        causal_mask = self.bias[:, :, :logits.size(-2), :logits.size(-1)].squeeze(0)  # 11ij->1ij, may be unneccesary?
        # per_example_sum. per_example_mean is hard to define when using unormalized attn attr  # bug fix
        loss = -torch.einsum('bij->b', logprobs * causal_mask.to(attn_labels.device) * attn_labels)#.to(hq.dtype)
    return Outputs(hidden_states=(head_output,) if head_output is not None else (), logits=logits, loss=loss)

def mlp_forward(block, hidden_states, layer=None, gate=None, output_intermediate=False, residual=False,
                ln_state=None, logits_mask=None, lnf_state=None, labels=None, loss_reduction=None, scaled=False):
    if layer is not None:
        model = block
        block = model.transformer.h[layer]
    if block.ln_2.weight.device != hidden_states.device:  # ln_2s and mlps are lazily moved to GPU in attribution
        for name in ['ln_2', 'mlp']:
            clone_module_to(block, name, hidden_states.device, remove_on_redo=False, switch=True)
    try:
        hidden_states, ln2_state = scaled_ln(block.ln_2, hidden_states, scaled=scaled, **(ln_state or {}))
        if layer is None: return block.mlp(hidden_states, output_intermediate=output_intermediate), ln2_state
        hidden_states = hidden_states * int(residual)*0 + block.mlp(hidden_states, gate=gate, output_intermediate=False)
        logits, loss = lm_head_forward(model, hidden_states, logits_mask=logits_mask, lnf_state=lnf_state,
            labels=labels, loss_reduction=loss_reduction, scaled=scaled) \
            if labels is not None else (None, None)
    finally:
        if block.ln_2.weight.device != hidden_states.device:
            for name in ['ln_2', 'mlp']: switch_module_to(block, name, 'cpu')
    return Outputs(hidden_states=(hidden_states,), logits=logits, loss=loss)

def parse_label_type(label_type): # e.g. 'attn_labels:bos->ans0],3'
    if ':' not in label_type: return label_type, None, None
    attn_pattern, k_shot = ('bos->ans0]', 3) if label_type.endswith(':') else \
        (label_type.split(':')[1].split(',')[0], (int(label_type[-1]) if label_type[-1].isdigit() else None))
    return label_type.split(':')[0], attn_pattern, k_shot

def mixed_forward(model, hidden_states, layer, head, labels=None, label_type=None, attn_pattern=None,
    outputs=None, attn_attr=None, hidden_states0=None, ranges=None, attribute_k=False, **kwargs):
    # TODO: modify input arg kwargs in-place is not good practice
    kwargs['logits_mask'] = getattr(outputs, 'logits_mask', None)
    kwargs['lnf_state'] = outputs.ln_states[-1]
    if not isinstance(layer, Iterable): layer, head = [layer], [head]
    L = len(model.transformer.h); H = outputs.attentions[0].size(1)
    if any(h == H for h in head): assert label_type is None or 'attn_labels' not in label_type
    logits_after_sum = label_type and 'attn_labels' not in label_type
    if not isinstance(hidden_states, (list, tuple)): hidden_states = [hidden_states] * len(layer)
    else: assert len(layer) == len(head) == len(hidden_states), \
        f'{len(layer)}, {len(head)}, {len(hidden_states)}'

    def get_kwargs(layer, head, hs):
        if label_type and 'attn_labels' in label_type:
            assert head < H
            attn_size = outputs.attentions[0].size()[-2:]
            if label_type.startswith('argmax'):
                attn_labels = get_argmax_attn_labels(outputs, layer, head, labels=labels)
            elif ':' in label_type:
                _, ap, k_shot = parse_label_type(label_type)
                attn_labels, _ = attn_pattern2labels(ranges, ap, attn_size, k_shot=k_shot)
            else:
                attn_labels = attn_attr[layer, head] * outputs.attentions[layer][0, head]
                attn_labels = attn_labels * (attn_labels > 0)
            kwargs = {'attn_labels': attn_labels, 'hidden_states0': hidden_states0
                    if hidden_states0 is not None else outputs.hidden_states[layer]}
            if '/' in label_type:
                mask_ap = (attn_pattern.split('->')[0] if attn_pattern else 'bos') + \
                    '->' + label_type.split('/')[1]
                attn_mask = attn_pattern2labels(ranges, mask_ap, attn_size)[0]
                # attn_mask[attn_labels.abs().sum(1) == 0] = 0  # to set attn_mask[:k_shot] = 0
                # ensure that attn_mask fully contains attn_labels to avoid inf loss
                kwargs['attn_labels'] = attn_labels * attn_mask
                kwargs['attn_mask'] = (1 - attn_mask).tril()
        else:
            kwargs = {'attn_weights': outputs.attentions[layer]} \
                if head < H else {'gate': lget(outputs.intermediates, layer)}
            if label_type in ['labels', 'argmax_labels']:
                kwargs['labels'] = get_argmax_labels(model, outputs.head_outputs[layer][:, head], labels) \
                    if label_type == 'argmax_labels' else labels
        kwargs = {k: v.to(hs.device, dtype=hs.dtype if 'labels' not in k and k != 'attn_mask' else None)
                for k, v in kwargs.items()}
        kwargs['ln_state'] = to(outputs.ln_states[layer][int(head < H)], hs.device)
        return kwargs

    fwd_outputs = [head_forward(model, hs, l, h, attribute_k=attribute_k, **get_kwargs(l, h, hs), **kwargs) 
        if h < H else mlp_forward(model, hs, layer=l, **get_kwargs(l, h, hs), **kwargs)
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
    else: logits = None  # attn_logits is not useful
    hidden_states = (hidden_states,) if hidden_states is not None else ()
    return Outputs(hidden_states=hidden_states, logits=logits, loss=loss)

def lm_head_forward(model, hidden_states, logits_mask=None,labels_mask=None, labels=None, loss_reduction=None,
                lnf_state=None, return_logit_diff=False, compact=False, scaled=False):
    if compact and labels is not None:
        if labels.size(0) != hidden_states.size(0): labels = einops.repeat(labels, '1 i -> b i', b=hidden_states.size(0))
        valid_flags = labels != -100
        n_valid = torch.einsum('bi->b', valid_flags)[0].item()
        hidden_states = rearrange(hidden_states[valid_flags], '(b i) e -> b i e', b=hidden_states.size(0), i=n_valid)
        labels = rearrange(labels[valid_flags], '(b i) -> b i', b=labels.size(0), i=n_valid)
    hidden_states, _ = scaled_ln(model.transformer.ln_f, hidden_states, scaled=scaled, **(lnf_state or {}))
    # Keep the logits and loss computation in fp32 to improve accuracy, same as attn_logits
    hidden_states = hidden_states.to(model.lm_head.weight.dtype)
    logits = model.lm_head(hidden_states)
    if logits_mask is not None:
        logits_mask = logits_mask.to(logits.device, dtype=logits.dtype) #* (1e4 if logits.dtype == torch.float16 else 1e9)
        logits = logits + logits_mask
    else:
        logits_mask = torch.zeros_like(logits)
    loss = None
    if labels is not None:
        if return_logit_diff:
            b = (labels != -100).squeeze(0)  # 1i->i
            _logits, _labels = logits[:, b], labels[:, b]  # biv->bkv, 1i->1k
            _logits_mask = (logits_mask == 0)[:, b]  # biv->bkv
            _logits_mean = (_logits * _logits_mask).sum(-1) / _logits_mask.sum(-1)  # bk = bkv->bk / bkv->bk
            _logits = _logits - _logits_mean.unsqueeze(-1)  # bkv - bk->bk1 = bkv
            # loss as the NEGATIVE of logit diff
            loss = -_logits[:, torch.arange(_logits.size(1)), _labels[0]].sum(-1)  # bkv->bk->b, per_example loss
        else:
            if labels_mask is not None:
                labels = labels.clone()
                labels[labels_mask == 0] = -100
            loss = compute_loss(logits, labels, reduction=loss_reduction)
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
            loss = torch.einsum('bi->b', loss) / torch.einsum('bi->b', labels != -100)
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
        dtype = torch.float32 if str(device) == 'cpu' else torch.float16
        return a.to(device, dtype=dtype)
    elif isinstance(a, dict):
        return {k: to(v, device) for k, v in a.items()}
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
    all_hidden_states, intermediates, attn_outputs, mlp_outputs, ln_states = (), (), (), (), ()
    attn_fwd_outputs = []
    for i, b in enumerate(self.h):
        if from_layer is not None and i < from_layer: continue
        if i == detach_layer: hidden_states = hidden_states.detach()
        if output_hidden_states: all_hidden_states += (to(hidden_states, output_device),)
        hidden_states = hidden_states.to(b.ln_1.weight.device) # switch device on PP stage boundaries
        # input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs.input_ids
        # device = input_ids.device
        # batch_size, seq_length = input_ids.size()
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        # attention_mask = torch.ones(
        #     (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
        # )
        # attention_mask = model.transformer._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, 0
        # )
        # hidden_states = b(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]
        h, ln1_state = scaled_ln(b.ln_1, hidden_states, scaled=False)
        
        # attn_output, aw, _ = b.attn(h, attention_mask=attention_mask, position_ids=position_ids, output_attentions=True)
        # attn_fwd_output = (aw, None, None, None, None)
        attn_output, *attn_fwd_output = maybe_composed_attn_forward(attn_fwd_outputs, model, b, h, h, h,
            by_head=by_head, head_mask=head_mask[i], attn_weights=attn_weights[i], ranges=ranges)
        attn_fwd_output = [to(o, output_device) for o in attn_fwd_output]
        attn_fwd_outputs.append(attn_fwd_output)
        
        if False and i == special_head[0]:  # obsolete
            head_output = list(rearrange(head_output, 'b n i e -> n b i e'))  # nbie->n*bie
            head_output[special_head[1]] = head_output[special_head[1]] * special_head_multiplier
            special_head_outputs = torch.cat([special_head_outputs, head_output[special_head[1]]]) \
                if special_head_outputs is not None else head_output[special_head[1]] # sie, s in [1, 2] for 8-1, 12-10
            attn_output = sum(head_output)
        if False and i in multiplied_layers:  # obsolete
            multiplier = get_multiplier(b, hidden_states, aw, head_input, special_head_outputs, labels, i,
                attr_threshold=attr_threshold, base_multiplier=base_multiplier)
            attn_output = torch.einsum('bnie,ni->bie', head_output, multiplier)
        parallel_attn_mlp = my_isinstance(b, (GPTJBlock, GPTNeoXLayer))
        if not parallel_attn_mlp: hidden_states = hidden_states + attn_output
        (mlp_output, intermediate), ln2_state = mlp_forward(b, hidden_states, output_intermediate=True)
        if mlp_mask[i] is not None: mlp_output = einsum('bie,bi->bie', mlp_output, mlp_mask[i])
        intermediates += (to(intermediate, output_device),)
        hidden_states = (hidden_states + mlp_output) if not parallel_attn_mlp else \
            (attn_output + mlp_output + hidden_states)  # order matters!
        attn_output = to(attn_output, output_device); attn_outputs += (attn_output,)
        mlp_output = to(mlp_output, output_device); mlp_outputs += (mlp_output,)
        ln_states += ((to(ln1_state, output_device), to(ln2_state, output_device)),)
    all_attentions, values, attn_outs, head_inputs, head_outputs = zip(*attn_fwd_outputs)
    if output_hidden_states: all_hidden_states += (to(hidden_states, output_device),)
    hidden_states, lnf_state = scaled_ln(self.ln_f, hidden_states, scaled=False)
    ln_states += (to(lnf_state, output_device),)
    if output_hidden_states: all_hidden_states += (to(hidden_states, output_device),)

    logits = model.lm_head(hidden_states)
    loss = compute_loss(logits, labels, reduction=loss_reduction) if labels is not None else None
    logits, loss = to(logits, output_device), to(loss, output_device)

    def maybe_none(l): return l if l[0] is not None else None
    return Outputs(
        inputs_embeds=inputs_embeds, position_embeds=position_embeds,
        attn_outputs=attn_outputs, values=maybe_none(values), attn_outs=maybe_none(attn_outs),
        head_inputs=maybe_none(head_inputs), head_outputs=maybe_none(head_outputs), 
        intermediates=intermediates, mlp_outputs=mlp_outputs, ln_states=ln_states,
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

def show_predictions(tokenizer, example_strs, bos_indices, eos_indices, answers, candidates, answer_indices,
        logits=None, labels=None, mask_logits=False, logits_bias=None,
        k_shot=3, topk=5, loss_reduction='mean', sep='\t', verbose=True):
    # if k_shot == 0: mask_logits = True  # ioi task
    if isinstance(mask_logits, types.FunctionType): logits = mask_logits(logits); mask_logits = False
    use_openai_api = hasattr(logits, 'token_logprobs')  # isinstance(model, types.FunctionType)
    ans_nlls = []
    labels_mask = torch.zeros_like(labels)
    logits_mask = torch.zeros_like(logits) if candidates is not None else None

    assert len(bos_indices) == len(example_strs), '%d != %d' % (len(bos_indices), len(example_strs))
    top1_corrects, answer_probs, candidate_probs = [], [], []
    convert_fn = tokenizer.convert_ids_to_tokens if True else partial(convert_ids_to_tokens, tokenizer=tokenizer)
    for i, (example_str, bos_i, eos_i, ans_ids) in enumerate(zip(example_strs, bos_indices, eos_indices, answers)):
        ans_tokens = convert_fn(ans_ids)
        for j, (ans_id, ans_token) in enumerate(zip(ans_ids, ans_tokens)):#, ans_probs, ans_prob_dist):
            logits_str = ''
            if use_openai_api:
                dist = get_prob_dist(logits.top_logprobs[bos_i + 1 + j], topk=topk)
                ans_prob = round(math.exp(logits.token_logprobs[bos_i + 1 + j]), 3)
                if i >= k_shot: ans_nlls.append(-logits.token_logprobs[bos_i + 1 + j])
                top1_correct = max(dist.items(), key=lambda x: x[1])[0] == ans_token.replace('Ġ', ' ')
                if candidates is not None:
                    candidate_probs.append([dist.get(cand, 0.) for cand in [t.replace('Ġ', ' ')
                        for t in convert_fn(candidates[i])]])
                topk_stat = dist
            else:
                logits_ = logits[0, bos_i + j]; dist = logits_.softmax(-1)  # mqy
                # print('In show_predictions: logits_.topk =', logits_.topk(topk), 'dist.topk =', dist.topk(topk), 'candidates =', candidates)  # nrk debug
                labels_mask[:, bos_i + j] = logits_.argmax() == ans_id
                if candidates is not None and j == 0:
                    if logits_bias is not None:  # logits_bias promotes all candidates of the task. Better than logits_mask?
                        logits_ = logits_ + logits_bias.to(logits_.device)
                        dist = logits_.softmax(-1)
                        # logits_bias x 3 (approaching logits_mask) to include more labels
                        labels_mask[:, bos_i + j] = (logits_ + (logits_bias * 2).to(logits_.device)).argmax() == ans_id
                        logits_mask[:, bos_i + j] = logits_bias
                    else:  # logits_mask promotes candidates of the current example
                        logits_mask[:, bos_i] = -1e4; logits_mask[:, bos_i, candidates[i]] = 0
                        # logits_mask[:, bos_i, candidates[i]] = 1e3
                        masked_logits_ = logits_ + logits_mask[0, bos_i] #* (1e4 if logits.dtype == torch.float16 else 1e9)
                        if mask_logits: logits_ = masked_logits_; dist = logits_.softmax(-1)
                        # amend labels_mask with masked_logits which is used in attribution by default
                        labels_mask[:, bos_i + j] = masked_logits_.argmax() == ans_id
                    logits_str = ' '.join(('*' if cand == ans_id else '') +
                        f'{tokenizer.convert_ids_to_tokens(cand)}:{logits_[cand].item():.3f}'
                        for cand in set(candidates[i]).union({logits_.argmax().item()}))
                    candidate_probs.append([dist[cand].item() for cand in candidates[i]])
                ans_prob = round(dist[ans_id].item(), 3)
                top1_correct = (dist.argmax() == ans_id).item()
                topk_stat = show_topk(*dist.topk(topk), indices_fn=convert_fn)
                # print('In show_predictions: topk_stat =', topk_stat)  # nrk debug

            if i == k_shot and j == 0: k_shot_len = len(top1_corrects)
            top1_corrects.append(top1_correct)
            answer_probs.append(ans_prob)
            if verbose: 
                print(('*' if top1_correct else ' ') + ans_token, ans_prob, topk_stat, sep, example_str, logits_str)
    if use_openai_api:
        loss = (ans_nlls if loss_reduction == 'none' else sum(ans_nlls) / len(ans_nlls))
    else:
        loss = compute_loss(logits, labels, reduction=loss_reduction)
        loss = loss.item() if loss_reduction == 'mean' else loss[labels != -100]#.tolist()  # 'none'
        
    if verbose:
        print(loss, np.array(top1_corrects[k_shot_len:]).mean())
        f, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 2.4), sharex=True)
        x = [i + 0.5 for i in range(len(example_strs))] # to align with sns.heatmap
        _ = ax0.bar(x, top1_corrects, width=0.9, alpha=0.5)
        _ = ax0.plot(x, answer_probs, color='r')
        if candidates is not None and answer_indices is not None and answer_indices[0] is not None:
            label_probs = F.one_hot(torch.LongTensor(answer_indices))
            _ = sns.heatmap(torch.cat([label_probs, torch.Tensor(candidate_probs)], dim=1).T, cbar=False, ax=ax1)
        plt.show()
    return loss, top1_corrects[k_shot_len:], answer_probs, candidate_probs, logits_mask, labels_mask

def trim_outputs(outputs):
    return Outputs(
        # hidden_states=outputs.hidden_states,  # for sum_forward's forward_only mode
        # attentions=outputs.attentions,
        # attn_attr=outputs.attn_attr,
        logits=outputs.logits
    )

def is_trimmed_outputs(outputs): return outputs.mlp_outputs == ()

def is_cls_candidates(candidates):
    return len(set(candidates) & {'True', 'False', 'Yes', 'No'}) > 0  # set intersection is not empty

def predict(model, tokenizer, text, examples, k_shot=3, bos_token=' ->', eos_token=None, #'Ċ',
            logits_bias=None, custom_forward=True, by_head=None, trim=False, verbose=True, outputs = None):
    if by_head is None: by_head = ['value', 'attn_out']
    input_ids, labels, ranges, *args = make_data_tuple( # args = [example_strs, bos_indices, eos_indices, answers]
        text, examples, tokenizer, k_shot=k_shot, bos_token=bos_token, eos_token=eos_token)
    if outputs is not None:
        assert isinstance(tokenizer, LLAMATokenizer)
        o = outputs
    else:
        with torch.no_grad():
            o = forward0(model, input_ids.to(model.device), by_head=by_head, ranges=ranges) \
                if isinstance(model, nn.Module) and custom_forward else \
                model(input_ids.to(getattr(model, 'device', 'cpu')))
            # for name in ['logits', 'loss']: # move outputs of native forward to cpu
            for field in fields(o):
                name = field.name
                # workaround for the weird bug of CausalLMOutputWithPast tampering values
                if name == 'loss': setattr(o, name, None); continue
                v = getattr(o, name)
                if isinstance(v, torch.Tensor) and v.device != torch.device('cpu'):
                    setattr(o, name, v.to('cpu').float())

    logits = o.logits.float() # add lxy
    loss, top1_corrects, answer_probs, candidate_probs, logits_mask, labels_mask = show_predictions(
        tokenizer, *args, logits=logits, logits_bias=logits_bias, labels=labels, loss_reduction='mean',
        k_shot=k_shot, topk=3, verbose=verbose)
    # logits_mask is used when rel1 is equal relation and cxt_len > 1 (checked in make_data_tuple)
    cxt, query, cands, *_ = examples[0]
    if cands is not None and (cands[-2] == cands[-1] or is_cls_candidates(cands[-1])) and logits_mask is not None: # rel1 is copy or g2c task
        # print('In predict: set logits_mask')
        o.logits_mask = logits_mask
    # once evaluation completes, revise labels to use only correctly predicted labels for subsequent attribution
    # if labels_mask is not None:
    #     labels[labels_mask == 0] = -100
    #     #assert not (labels == -100).all() add lxy, 测试准确率存在bug
    o.labels_mask = labels_mask
    if trim: o = trim_outputs(o)

    data_tuple = [text, input_ids, labels, ranges] + args + [o]
    eval_result = (loss, top1_corrects, answer_probs, candidate_probs)
    return data_tuple, eval_result

def getLLAMAOutputs(texts, tokenizer, model, only_logits = True, use_recorder=False):
    #'7B' #13B, 30B 65B      # layers 32 40 60 80
    batch_outputs = ()
    if use_recorder:
        dicts ={'7B': 32, '13B': 40, '30B': 60, '65B': 80}
        layers = dicts.get(layers, 32)
        recorder = BatchShareMemRecorder(prefix="LLAMA-7B_0313_")
        recorder.clear() 
        with open('/nas2/hyy/llama/input.txt', 'a') as f:
            f.write("====".join(texts).replace('\n','@@') + '\n')
        with Timer('wait llama model write data'):
            while not recorder.exists('output-output-final'): time.sleep(0.2)

        hidden_states_rules = ['tok_embeddings-output'] + \
            [layer + str(number) + name for layer, number, name in product(['layers.'], range(layers), ['-output'])] +['norm-output']
        mlp_rules = [layer + str(number)  + name for layer, number, name in product(['layers.'], range(layers), ['.feed_forward-output'])]
        attention_rules = [layer + str(number)  + name for layer, number, name in product(['layers.'], range(layers), ['.attention-output'])]

        batch_all_hidden_states, batch_mlp_outputs, batch_attn_outputs, \
            batch_all_attentions, batch_values, batch_attn_outs = [], [], [], [], [], []
        batch_logits = list(recorder._load('output-output')[0].unsqueeze(1))
        lengths = [len(tokenizer.tokenize(text)) for text in texts]
        
        if not only_logits:
            batch_all_hidden_states = [list(recorder._load(key)[0].unsqueeze(1)) for key in hidden_states_rules]
            batch_mlp_outputs = [list(recorder._load(key)[0].unsqueeze(1)) for key in mlp_rules]
            
            def converse(values):
                return [list(value.unsqueeze(dim = 1)) for value in values]
            
            batch_attn_outputs, batch_all_attentions, batch_values, batch_attn_outs = \
                zip(*[converse(recorder._load(key)) for key in attention_rules])
            attn_fwd_output = lengths, zip(*batch_all_hidden_states), zip(*batch_mlp_outputs),\
                zip(*batch_attn_outputs),zip(*batch_all_attentions),zip(*batch_values),zip(*batch_attn_outs), batch_logits
            
            for length, all_hidden_states, mlp_outputs, attn_outputs, all_attentions, values, attn_outs,logits in zip(*attn_fwd_output):
                all_hidden_states = tuple(hidden_states[:, :length, :] for hidden_states in all_hidden_states)
                mlp_outputs = tuple(mlp_output[:, :length, :] for mlp_output in mlp_outputs)
                attn_outputs = tuple(attn_output[:, :length, :] for attn_output in attn_outputs)
                all_attentions = tuple(all_attention[:, :, :length, :length] for all_attention in all_attentions)
                values = tuple(value[:, :, :length, :] for value in values)
                attn_outs = tuple(attn_out[:, :, :length, :] for attn_out in attn_outs)
                logits = logits[:, :length, :]
                o = Outputs(inputs_embeds=all_hidden_states[0], attn_outputs=attn_outputs, values=values, 
                    attn_outs=attn_outs,mlp_outputs=mlp_outputs, hidden_states=all_hidden_states, 
                    attentions=all_attentions, logits= logits)
                batch_outputs += (o,)
        else:
            for length, logits in zip(lengths, batch_logits):
                logits = logits[:, :length, :]
                o = Outputs(logits = logits)
                batch_outputs += (o,)
        
        recorder.clear()
    else:
        if only_logits:
            #print(len(texts), texts)
            for text in texts:
                inputs_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).long().to(model.device)[:,1:] # rm bos id
                #print('inputs_ids', inputs_ids.shape, inputs_ids)
                #inputs_ids[0,0] = 0 # mqy replace bos token id -1 as 0 
                out = model.forward(inputs_ids)
                logits = out['logits'].cpu()
                logits = torch.cat([logits[:,:1],logits], dim=1)
                o = Outputs(logits = logits)
                batch_outputs += (o,)
    return batch_outputs

def result2dict(result):
    return {k: v for k, v in result.__dict__.items() if k in ['mean_loss', 'mean_acc', 'answer_probs', 'texts']}
    
def generate_and_predict_batch(model, tokenizer, task, nrows, k_shot, batch_size,
            logits_bias=None, ioi_dataset=None,mathlogic_dataset=None, by_head=None, trim=False,custom_forward=True,
            verbose=True, result=None, save_label = False, test_acc = False, **gen_args):
    if result is None:
        if ioi_dataset is not None:
            indices_groups = np.arange(batch_size * nrows).reshape(batch_size, nrows)
            texts, all_examples, all_bos_tokens = [], [], []
            for indices in indices_groups:
                # texts.append('\n'.join(ioi_dataset.sentences[i] for i in indices) + '\n')#wab
                texts.append(''.join(ioi_dataset.sentences[i] for i in indices))#wab
                all_examples.append([{k: ioi_dataset.word_idx[k][i] for k in ioi_dataset.word_idx} for i in indices])
                all_bos_tokens.append(None)
        elif mathlogic_dataset is not None:  #nrk
            all_examples, texts, all_bos_tokens = mathlogic_dataset
        else:
            all_examples, texts, all_bos_tokens = zip(*[generate(task, nrows=nrows, tokenizer=tokenizer,
                plot=False, verbose=False, **gen_args) for i in range(batch_size)])
        
        #texts = [text.replace('The boy', 'He').replace('The girl', 'She') for text in texts] # mqy test another text format
        #for e in all_examples:
        #    for item in e:
        #        item[1] = item[1].replace('boy', 'he').replace('girl', 'she')
        #        item[2] = tuple([t.replace('boy', 'he').replace('girl', 'she') for t in item[2][0]] + list(item[2][1:])) 
        #texts = [text.replace(' not', '') for text in texts] # mqy test another text format

        result = Result(task=task, gen_args=gen_args, all_examples=all_examples, texts=texts, all_bos_tokens=all_bos_tokens)
    else:
        all_examples, texts, all_bos_tokens = result.all_examples, result.texts, result.all_bos_tokens
    for text in texts[:1]: print('\n'.join(text.strip('\n').split('\n')[:3]))
    if batch_size == 1: return result

    if result.data_tuples is None or is_trimmed_outputs(result.data_tuples[0][-1]):
        assert not my_isinstance(tokenizer, LLAMATokenizer)
        batch_outputs = getLLAMAOutputs(texts, tokenizer, gen_args.get('llama_size', '7B'), False) \
            if my_isinstance(tokenizer, LLAMATokenizer) else () # add by lxy
        # add lxy get labels to construct dataset
        if save_label:
            input_ids, labels, ranges, *args = map(list, zip(*[make_data_tuple(
            text, examples, tokenizer, k_shot=k_shot, bos_token=bos_tokens, eos_token=None)
            for i, (examples, text, bos_tokens) in enumerate(zip(all_examples, texts, all_bos_tokens))]))
            task_dict = {k: v for k, v in result.__dict__.items() if k in ['texts']}
            task_dict['bos'] = [[data.bos for data in r] for r in ranges]
            task_dict['input_ids'] = [input_id.tolist() for input_id in input_ids]
            return task_dict
            # return result
        if True: # with Timer('In generate_and_predict_batch: predict'):
            result.data_tuples, result.eval_results = map(list, zip(*[predict(model, tokenizer, text, examples,
                k_shot=k_shot, bos_token=bos_tokens, logits_bias=logits_bias, by_head=by_head,
                trim=trim, custom_forward=custom_forward, verbose=verbose,
                outputs= batch_outputs[i] if my_isinstance(tokenizer, LLAMATokenizer) else None)
                for i, (examples, text, bos_tokens) in enumerate(zip(all_examples, texts, all_bos_tokens))
                if True or any(s in text[24:] for s in ['dangerous'])]))
                #if i in [0, 4, 6, 17, 21, 32, 37, 38, 39, 45, 47, 55, 57]])) # mqy improve gender task acc temporarily
        loss, acc, answer_probs, *_ = zip(*result.eval_results)
        result.mean_loss, result.mean_acc = np.array(loss).mean(), np.array(join_lists(acc)).mean()
        result.answer_probs = np.array(answer_probs).mean(0)
        print(result.mean_loss, result.mean_acc, result.answer_probs[k_shot:].mean())
        plt.plot(result.answer_probs); plt.show()
    if test_acc:
        result.texts = texts[:1]
        return result2dict(result)
    return result

def show_predictions_by_data_tuples(model, tokenizer, data_tuples, k_shot, to_layer=None, verbose=True):
    losses, acc = [], []
    for _, _, labels, ranges, *args, o in data_tuples:
        if to_layer is not None:
            L = len(model.transformer.h)
            hs = o.hidden_states[to_layer]
            # hs = o.mlp_outputs[to_layer]
            for b in model.transformer.h[max(L - 2, to_layer):]:
                hs = hs + mlp_forward(b, hs)[0]
            hs = model.transformer.ln_f(hs)
            logits = model.lm_head(hs)
            logits = logits + o.logits_mask #* (1e4 if logits.dtype == torch.float16 else 1e9)
        else:
            logits = o.logits.float()
        loss, top1_corrects, *_ = show_predictions(
            tokenizer, *args, logits=logits, labels=labels, loss_reduction='mean',
            # candidates=candidates, answer_indices=answer_indices,
            k_shot=k_shot, topk=3, verbose=verbose)
        losses.append(loss); acc += top1_corrects[k_shot:]
    return np.array(losses).mean(), np.array(acc).mean()

def data_tuples_g(mt, result, k_shot, slice_obj):
    model, tokenizer = mt
    if isinstance(slice_obj, int): slice_obj = slice(slice_obj)
    for examples, text, bos_tokens in zip(result.all_examples[slice_obj],
                        result.texts[slice_obj], result.all_bos_tokens[slice_obj]):
        yield predict(model, tokenizer, text, examples, k_shot=k_shot, 
            bos_token=bos_tokens, verbose=False)[0]

def abbreviate_attn_pattern(attn_pattern):
    return attn_pattern.replace('bos', 'B').replace('query', 'Q').replace('tgt', 'T').\
        replace('ans', 'A').replace('sep', 'S').replace('rel', 'R').replace('cls', 'C').replace('ans0','A0').replace('tgt0', 'T0')

def restore_attn_pattern(attn_pattern):
    return attn_pattern.replace('B', 'bos').replace('Q', 'query').replace('T', 'tgt').\
        replace('A', 'ans').replace('S', 'sep').replace('R', 'rel').replace('C', 'cls').replace('A0','ans0').replace('T0','tgt0')

def is_intermediary(attn_pattern):
    # return attn_pattern.endswith('^')
    return attn_pattern.startswith('bos->ans') and not attn_pattern.startswith('bos->ans0') \
        or attn_pattern == 'bos->bos^'

def is_relating(attn_pattern):
    return attn_pattern.startswith('ans]->') or attn_pattern.startswith('ans->')

def strip_range_label(label): 
    for s in ['[', ']', '+', '-', '^']: label = label.replace(s, '')
    return label

def range_contains(range0, range1):
    return range0[0] <= range1[0] and range1[1] <= range0[1]

def get_range(range, label):
    # if(getattr(range, strip_range_label(label)) == None): return ; #add by nrk some location is None, may report TypeError: cannot unpack non-iterable NoneType object
    if hasattr(range, label): return getattr(range, label)
    _range = getattr(range, strip_range_label(label))
    if _range is None: return None # mqy
    # assert not label.endswith('^'), label
    b, e = _range
    if label.startswith('['): return b, b + 1
    elif label.endswith(']'): return e - 1, e
    elif label.endswith('+'): return e, e + 1
    elif label.endswith('-'): return b - 1, b
    elif label.endswith('^'): return b, e
    else: assert False, label

def attn_pattern2labels(ranges, attn_pattern, attn_size, k_shot=0, attribute_k=False, normalize=False):
    attn_pattern = restore_attn_pattern(attn_pattern)
    q, k = attn_pattern.split('->')  # e.g. 'bos->ans0'
    cross_example = k.endswith('^'); k = k.replace('^', '')
    ranges_q = ranges[k_shot:] if attribute_k or not is_relating(attn_pattern) \
        else ranges[: len(ranges) - k_shot]
    if k in ['ans0s', 'ntgts'] and getattr(ranges_q[0], k) is None:
        k = 'example'  # tasks with cxt_len=1 has no 'ans0s' or 'ntgts'. Fall back to 'example'
    attn_labels = torch.zeros(*attn_size)
    def get_slice(r, name):
        try:
            if name == '*': b, e = 0, attn_size[0]
            elif name.startswith('['): b, _ = getattr(r, name[1:]); e = b + 1
            elif name.endswith(']'): _, e = getattr(r, name[:-1]); b = e - 1
            elif name.endswith('+'): _, e = getattr(r, name[:-1]); b, e = e, e + 1
            elif name.endswith('-'): b, _ = getattr(r, name[:-1]); b, e = b - 1, b
            else: b, e = getattr(r, name)
        except Exception as e:
            print(f'In attn_pattern2labels: name = {name} not in {[k for k, v in r.__dict__.items() if v]}?')
            raise e
        return slice(b, e) if not isinstance(b, Iterable) else [slice(_b, _e) for _b, _e in zip(b, e)]
    for i, r in enumerate(ranges_q):
        if cross_example: # is_intermediary(attn_pattern):
            for rk in ranges[: i]: attn_labels[get_slice(r, q), get_slice(rk, k)] = 1
            # if True:  # only to prev bos, not to self
            #     attn_labels = attn_labels * (1 - torch.eye(attn_labels.size(0)))
        elif any(s in k for s in ['ans0s', 'ntgts']): # 'candidates']:
            rqs, rks = get_slice(r, q), get_slice(r, k)
            if not isinstance(rqs, Iterable): rqs = [rqs]
            if not isinstance(rks, Iterable): rks = [rks]
            raw_k_len = len(rks)
            assert len(rqs) == len(rks) or len(rks)==1 or len(rqs) == 1
            rqs, rks = rqs * int(max(len(rqs), len(rks)) / len(rqs)), rks * int(max(len(rqs), len(rks)) / len(rks))  # broadcast
            for _rq, _rk in zip(rqs, rks):
                attn_labels[_rq, _rk] = 1 / raw_k_len
            #normalize = True # normalize when one q -> multi k 
            #for s in get_slice(r, k):
            #    attn_labels[get_slice(r, q), s] = 1
                #rq = get_slice(r, q)
                #for _rq, _s in product(rq, s): # mqy ans0s ->ans0s
                #    attn_labels[_rq, _s] = 1
        elif '<s>' in k:
            attn_labels[get_slice(r, q), 0] = 1
        elif k == 'null':
            attn_labels[get_slice(r, q), :] = 0  # equivalent to pass
        else:
            attn_labels[get_slice(r, q), get_slice(r, k)] = 1
    if k == '~<s>':
        # attn_labels = 1 - attn_labels  # 0,1,1,...
        attn_labels = -attn_labels  # -1,0,0,...
        # attn_labels = 1 - 2 * attn_labels  # -1,1,1,...
    attn_labels = attn_labels.tril()
    if any(s in k for s in ['ans0s', '<s>', 'example']): # 'candidates', 
        normalize = False  # used as attn_mask
    if normalize:
        attn_labels = attn_labels / (attn_labels.sum(1, keepdim=True) + 1e-9)
    return attn_labels, [get_slice(r, q) for r in ranges_q]

cat = torch.cat

def gather_by_heads(tensors, heads):
    # attentions (l*[1nij]) / values (l*[1nid]) / head_inputs/head_outputs (l*[1nie])
    return torch.cat([tensors[l][:, h] for l, h in heads])  # k*[1ix]->kix

def scatter_by_heads(tensors, heads, size):
    out = torch.zeros(size + tensors.size()[1:])
    for (l, h), tensor in zip(heads, tensors): out[l, h] = tensor
    return out

def sum_forward(model, outputs, labels=None, loss_reduction='per_example_mean',
        embed_mask=None, mlp_mask=None, head_mask=None, neuron_mask=None, attn_weights=None,
        forward_only=False, from_layer=0, output_layer=None, sum_output=None, attr_heads=None,
        reduce_fn=sum, truncate_layers=False, scaled=True, reshape=False, device=None):
    blocks = model.transformer.h; H = blocks[0].attn.num_heads
    o = outputs
    assert len(o.mlp_outputs) > 0 or len(o.hidden_states) > 2  # latter for forward_only
    has_mask = any(m is not None for m in [embed_mask, mlp_mask, head_mask, neuron_mask, attn_weights])
    if output_layer is None: _l = len(o.mlp_outputs) or len(o.hidden_states) - 2
    else: _l = max(output_layer) if isinstance(output_layer, Iterable) else output_layer
    l_ = from_layer
    kept_dim, combine_fn = ('l', partial(torch.cat, dim=1)) \
        if isinstance(output_layer, Iterable) else ('', sum)
    # if labels is not None and getattr(o, 'labels_mask', None) is not None:
    #     labels = labels.clone()
    #     labels[o.labels_mask == 0] = -100
    kwargs = dict(labels=labels, loss_reduction=loss_reduction, scaled=scaled)  # for lm_head_forward
    kwargs['logits_mask'] = getattr(o, 'logits_mask', None)
    kwargs['lnf_state'] = o.ln_states[-1]
    if not has_mask:  # for logit attribution
        # assert isinstance(getattr(o, 'ln_states', None), (tuple, list))
        # kwargs['lnf_state'] = o.ln_states[-1]
        kwargs['return_logit_diff'] = True
    logits, loss = None, None
    
    device = device or model.lm_head.weight.device
    on_gpu = False # device != torch.device('cpu')
    if on_gpu: mu = mem_usage(device)

    if forward_only:
        output = to(einsum('bie,b->bie', o.hidden_states[_l], mlp_mask.mean(1)), device)  # 1ie,(bl->b)->bie
        if labels is not None: logits, loss = lm_head_forward(model, output, **kwargs)
        return Outputs(hidden_states=(output,), logits=logits, loss=loss)

    if sum_output is not None and attr_heads is not None:
        # TODO: check that out_proj.bias can be safely ignored
        if o.values is not None:
            output = einsum('bkij,kjd->bkid', attn_weights,
                to(gather_by_heads(o.values, attr_heads), device))
            output = einsum(f'blid,lde->b{kept_dim}ie', output, model.wos)
        else:  # obsolete. Using head_inputs, though faster, consumes too much memory
            output = einsum(f'blij,lje->b{kept_dim}ie', attn_weights,
                to(gather_by_heads(o.head_inputs), device))
        if isinstance(output_layer, Iterable):
            assert len(sum_output) == len(output_layer), f'{len(sum_output)} != {len(output_layer)}'
            _output0 = output.sum(1)  # bkie->bie
            for i, ol in enumerate(output_layer):
                mask = torch.Tensor([l < ol for l, _ in attr_heads])
                _output = torch.einsum('bkie,k->bie', output, to(mask, device)) \
                    if mask.sum() < len(attr_heads) else _output0
                sum_output[i] = (to(sum_output[i], device) - _output).detach() + _output
            output = sum_output
        else:
            output = (to(sum_output, device) - output).detach() + output
        if labels is not None: logits, loss = lm_head_forward(model, output, **kwargs)
        if on_gpu: print(f'mem_usage when enter / exit sum_forward stage2: {mu} / {mem_usage(device)}. len = {output.size(1)}')
        return Outputs(hidden_states=(output,), logits=logits, loss=loss)

    embed_output = to(o.hidden_states[0], device)
    if embed_mask is not None:
        embed_output = einsum('bie,bi->bie', embed_output, embed_mask) # i=1 for embed_mask
        
    if reduce_fn == torch.cat and head_mask is None and attn_weights is not None:
        head_mask = einops.reduce(attn_weights, '1 l n i j -> 1 l n i', 'sum')
        attn_weights = None
    if head_mask is not None:
        if o.attn_outs is not None:
            attn_outputs = None
            for i in range(l_, _l):
                attn_output = einsum('bnid,bn->bind', to(o.attn_outs[i], device), head_mask[:, i])
                attn_output = rearrange(attn_output, 'b i n d -> b i (n d)')
                attn_output = blocks[i].attn.out_proj(attn_output)  # bie,ee->bie
                if kept_dim == 'l': attn_output = attn_output.unsqueeze(1)  # bie->blie, l=1
                attn_outputs = attn_output if attn_outputs is None else \
                    combine_fn([attn_outputs, attn_output])
        else:  # obsolete. Using head_outputs, though faster, consumes too much memory
            # head_outputs = [postprocess_head_output(o, labels != -100) for o in o.head_outputs] # 1nid->1nie
            attn_outputs = einsum(f'lnie,bln->b{kept_dim}ie',
                to(cat(o.head_outputs[l_: _l]), device), head_mask[:, l_: _l])
            if blocks[l_].attn.out_proj.bias is not None:
                bias = rearrange([blocks[i].attn.out_proj.bias for i in range(l_, _l)], 'l e -> l 1 e')
                attn_outputs = attn_outputs + einsum(f'l1e->{kept_dim}1e', bias)
    elif neuron_mask is not None:  # obsolete
        head_outputs = einsum('blnie,blnie->blnie', neuron_mask[:, : _l], head_outputs[:, : _l])
    elif attn_weights is not None:
        # head_inputs = postprocess_head_inputs(head_inputs, attr_heads)  # 1lnid->1lnie
        # Curiously, einsum uses much less GPU mem than matmul then sum!
        # attn_outputs = (attn_weights[:, l_: _l] @ head_inputs[:, l_: _l]).sum(dim=(1,2))  # blnij,blnje->blnie->bie
        attn_outputs = einsum(f'blnij,lnje->b{kept_dim}ie',
            to(attn_weights[:, l_: _l], device), to(cat(o.head_inputs[l_: _l]), device))
        if blocks[l_].attn.out_proj.bias is not None:
            bias = rearrange([blocks[i].attn.out_proj.bias for i in range(l_, _l)], 'l e -> l 1 e')
            attn_outputs = attn_outputs + einsum(f'l1e->{kept_dim}1e', bias)
    else:  # for logit attribution
        attn_outputs = torch.cat([einsum('bnid,nde->bnie', to(o.attn_outs[i], device),
            rearrange(blocks[i].attn.out_proj.weight, 'e (n d) -> n d e', n=H))  # wo
            for i in range(l_, _l)], dim=0)  # [1nie]*l->lnie
        
    if has_mask and l_ > 0:
        prev_attn_outputs = einsum(f'lie,bi->b{kept_dim}ie',
            to(cat(o.attn_outputs[: l_]), device), embed_mask.clone().detach())
        attn_outputs = combine_fn([prev_attn_outputs, attn_outputs])

    if mlp_mask is not None:
        mlp_outputs = einsum(f'lie,bl->b{kept_dim}ie',
            to(cat(o.mlp_outputs[: _l]), device), mlp_mask[:, : _l])
    else:  # for logit attribution
        mlp_outputs = to(cat(o.mlp_outputs[l_: _l]), device)  # [1ie]*l->lie

    if not has_mask: # or reshape:  # for logit attribution or head amplication
        reduce_fn = torch.cat
        # L = (torch.einsum('bli->l', mlp_mask) > 0).sum().item() \
        #     if mlp_mask is not None and truncate_layers else len(outputs.mlp_outputs)
        # attn_outputs = rearrange(head_outputs[:, :L], '1 l n i e -> l n i e')
        mlp_outputs = rearrange(mlp_outputs, 'l i e -> l 1 i e')
        # padded_embed_output = embed_output.new(*((L,) + embed_output.size())).zero_() # l1ie
        # padded_embed_output[0] = embed_output
        # output = torch.cat([attn_outputs, mlp_outputs, padded_embed_output], dim=1) # lnie,l1ie,l1ie->l(n+2)ie
        output = torch.cat([attn_outputs, mlp_outputs], dim=1) # lnie,l1ie->l(n+1)ie
        output = rearrange(output, 'l n i e -> (l n) i e')
    elif output_layer is not None:  # default
        assert reduce_fn == sum
        if isinstance(output_layer, Iterable):
            output = [reduce_fn([embed_output,
                                attn_outputs[:, : l].sum(1),  # blie->bie
                                mlp_outputs[:, : l].sum(1),  # blie->bie
                                ]) for l in output_layer]
        else:
            output = reduce_fn([embed_output, attn_outputs, mlp_outputs])  # bie
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

    if labels is not None: logits, loss = lm_head_forward(model, output, **kwargs)
    if on_gpu: print(f'mem_usage when enter / exit sum_forward: {mu} / {mem_usage(device)}. len = {output.size(1)}')
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

def check_abnormal_tensor(tensor, tensor_name):
    if tensor.dtype == torch.float32: return  # typically on cpu
    nan_pct, inf_pct = tensor.isnan().float().mean(), tensor.isinf().float().mean()
    if nan_pct > 0 or inf_pct > 0:
        print('In check_abnormal_tensor:', tensor_name, end=' ')
        if nan_pct > 0: print(f'nan_pct = {nan_pct}', end=' ')
        if inf_pct > 0: print(f'inf_pct = {inf_pct}', end=' ')
        print()

def attribute(forward_fn, model, x, post_forward_fn=[], num_points=7, forward_only=False):
    batch_size = num_points + 1  # bsz can not be smaller due to scaled_ln
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=True)
    grad_keys = list(x.keys())  # [key for key in x if key != 'head_mask' or 'attn_weights' not in x]
    scaled_x, grad = OrderedDict(), OrderedDict()
    with torch.enable_grad() if not forward_only else torch.no_grad():
        for key in x:
            scaled_x[key] = scaled_input(x[key], num_points)
            grad[key] = 0.
        ys, hidden_states = [], []  # negative loss and hidden_states
        for i in range(0, num_points + 1, batch_size):
            scaled_x_ = OrderedDict({key: scaled_x[key][i: i + batch_size] for key in x})
            o = forward_fn(model, forward_only=forward_only, **scaled_x_)
            hidden_states0 = o.hidden_states[-1]  # maybe reused as sum_output for sum_forward
            hs, y, logits = post_forward_fn(model, o); ys.append(y); hidden_states.append(hs)
            for tensor, name in [(o.hidden_states[-1], 'hidden_states_in'), (hs, 'hidden_states_out'), 
                                (y, 'y'), (logits, 'logits')]:
                if isinstance(tensor, torch.Tensor): check_abnormal_tensor(tensor, name)
            if forward_only: continue
            #grad_ = torch.autograd.grad((-y).flatten().unbind(), list(scaled_x.values())) # mqy test reverse attribution ///
            grad_ = torch.autograd.grad(y.flatten().unbind(), list(scaled_x.values()))
            for j, key in enumerate(grad_keys):
                grad[key] = grad[key] + grad_[j].sum(dim=0, keepdim=True)
                check_abnormal_tensor(grad[key], key + '.grad')
    hidden_states, ys = [torch.cat(ts) if ts[0] is not None else None for ts in [hidden_states, ys]]
    if forward_only: return None, hidden_states0, hidden_states, ys, logits

    attr = {key: (grad[key] / (num_points + 1) * (x[key] if key != 'attn_weights' else 1)
                ).squeeze(0) for key in grad_keys}  # squeeze batch dim
    attr = Attributions(**{key.split('_')[0]: attr.get(key) for key in
        ['attn_weights', 'head_mask', 'neuron_mask', 'mlp_mask', 'embed_mask']})
    return attr, hidden_states0, hidden_states, ys, logits

def logit_attribute(forward_fn, model, post_forward_fn):
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=False)
    with torch.no_grad():
        o = forward_fn(model)
        hs, y, logits = post_forward_fn(model, o)
    assert y.ndim == 1  # (l(n+1))
    return y
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    # assert (y.size(0) - 1) % (H + 1) == 0
    # to_layer = (y.size(0) - 1) // (H + 1) # by solving equation b = 1 + ln + l
    # assert to_layer == (torch.einsum('blni->l', x['head_mask']) > 0).sum().item()
    embed_attr = y[:1].view(1)
    head_attr = y[1: 1 + L * H].view(L, H)  # ln
    mlp_attr = y[1 + L * H:]#.view(L, 1)  # l1
    return Attributions(embed=embed_attr, head=head_attr, mlp=mlp_attr)

def attributions_to(attr, device='cpu'):
    # for field in fields(attr):
    #     name = field.name
    for name in dataclasses.asdict(attr).keys():
        t = getattr(attr, name)
        if t is not None and isinstance(t, torch.Tensor):
            dtype = torch.float32 if name != 'attn' else torch.float16  # to save mem
            setattr(attr, name, t.to(device, dtype=dtype))
    return attr

def attribute_step(data_tuple, model, node, attribute_k=False, 
                staged=True, attr_heads=None, do_logit_attribution=False, device=None):
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    output_layer = node.data.layer
    to_layer = max(output_layer) if isinstance(output_layer, Iterable) else output_layer
    device = device or model.lm_head.weight.device
    text, input_ids, labels, ranges, *_, o = data_tuple
    if getattr(o, 'labels_mask', None) is not None:
        labels = labels.clone()
        labels[o.labels_mask == 0] = -100
    labels = labels.to(device)

    fns = path2fns(node, partial(node2fn, outputs=o, ranges=ranges, labels=labels))
    if len(fns) > 0: labels = None
    elif node.data.label_type == 'argmax_labels':  # for root
        labels = get_argmax_labels(model, o.hidden_states[-2], labels)
    from_layer = math.floor(L / 2.5) if to_layer == L else 0

    if not do_logit_attribution:
        sum_output = o.sum_output if attr_heads is not None else None
        fwd_fn = partial(sum_forward, outputs=o, labels=labels,
                    from_layer=from_layer, output_layer=output_layer,
                    sum_output=sum_output, attr_heads=attr_heads, device=device)
        if not staged: keys = ['attn_weights', 'mlp_mask', 'embed_mask']
        else: keys = ['mlp_mask', 'embed_mask', 'head_mask'] if attr_heads is None else ['attn_weights']
        x = OrderedDict((key, get_x(key, o, attr_heads=attr_heads, to_layer=to_layer, device=device)) for key in keys)
        attr, sum_output = attribute(fwd_fn, model, x, fns, num_points=3+4 if True or attribute_k else 7)[:2]
        print(f"In attribute_step : attr = {attr}") #nrk
        if attr.head is None and attr.attn is not None and attr_heads is None:
            attr.head = torch.einsum('lnij->ln', attr.attn) 
        attr = attributions_to(attr)
        if staged:   # use o to pass sum_output to next call. tricky
            if attr_heads is None: o.sum_output = sum_output  # 1st stage
            else: del o.sum_output  # 2nd stage
        if attr.attn is not None: # associate non-averageable attn attr to current node. tricky
            o.attn_attr[node2key(node)] = attr.attn if attr_heads is None else \
                OrderedDict((lh, attr.attn[i]) for i, lh in enumerate(attr_heads))
                # attr.attn: kij. Keep only topk attn_attr to save memory
        del attr.attn  # avoid being reduced by reduce_objects
    else:
        to_layer = min(output_layer) if isinstance(output_layer, Iterable) else output_layer
        attrs, step = [], 2
        for l in range(from_layer, to_layer, step):  # compute layer-wise to save GPU mem
            fwd_fn = partial(sum_forward, outputs=o, labels=labels,
                from_layer=l, output_layer=min(to_layer, l + step), device=device)
            attrs.append(logit_attribute(fwd_fn, model, fns))
        attrs = torch.cat(attrs).view(to_layer - from_layer, -1)  # l(n+1)
        assert attrs.size(1) == H + 1, str(attrs.size())
        attr = Attributions(head=attrs[:, :H],  # ln
                                mlp=attrs[:, H])  # l
        attr = attributions_to(attr)
        if from_layer > 0:
            attr.head = torch.cat([torch.zeros(from_layer, H), attr.head])
            attr.mlp = torch.cat([torch.zeros(from_layer), attr.mlp])
    return attr

def get_x(key, outputs, attr_heads=None, to_layer=None, device=torch.device('cpu')):
    L = len(outputs.hidden_states)
    H = outputs.attentions[0].size(-3)  # bnij
    qlen = outputs.hidden_states[0].size(1)  # bie
    if to_layer is None: to_layer = L
    # L dim removed when doing per-layer attribution
    if key == 'head_mask': x = torch.ones(1, to_layer, H)#, qlen)
    elif key == 'neuron_mask': x = torch.ones(1, to_layer, H, qlen, outputs.hidden_states[0].size(-1))
    elif key == 'mlp_mask': x = torch.ones(1, to_layer)#, qlen)
    elif key == 'embed_mask': x = torch.ones(1, 1)#, qlen)
    elif key == 'attn_weights': x = gather_by_heads(outputs.attentions, attr_heads).unsqueeze(0)  # kij->1kij
    # if to_layer is not None and x.ndim >= 2 and x.size(1) == L: x[:, to_layer:] = 0

    dtype = torch.float32 if device == torch.device('cpu') else torch.float16
    x = x.to(device, dtype=dtype)
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

def get_ap_scores(aw, attn_labels, ranges_q):
    assert (aw >= 0).all()
    b = attn_labels.sum(1) > 0  # ij->i
    if b.sum() == len(ranges_q):
        scores = (aw[..., b, :] * attn_labels[b, :]).sum(-1)  # ...kj,kj->...kj->...k
    else:
        scores = (aw * attn_labels).sum(-1)  # ...ij,ij->...ij->...i
        scores = torch.cat([scores[..., rq_slice].max(-1, keepdim=True).values if not isinstance(rq_slice, Iterable) else torch.cat([scores[..., _rq_slice].max(-1, keepdim=True).values for _rq_slice in rq_slice], dim=-1).mean(-1, keepdim=True)
            for rq_slice in ranges_q], dim=-1)  # k*[...1]->...k
    return scores#, scores.mean(-1)  # ...k->..., e.g. lnk->ln

def get_head_matching_scores(data_tuple, attn_patterns, layer=None, head=None, node_key=None,
                            averaged=True, device=None, **kwargs):
    text, input_ids, labels, ranges, *args, o = data_tuple
    _attn_patterns = attn_patterns if isinstance(attn_patterns, list) else [attn_patterns]
    if node_key is None or isinstance(o.attn_attr[node_key], torch.Tensor):
        attentions = o.attentions[layer][0, head] if head is not None else \
            list_get(o.attentions, layer, reduce_fn=torch.cat)  # l*[nij]->lnij/nij
        if device is not None: attentions = to(attentions, device)
    matching_scores = {}
    for attn_pattern in _attn_patterns:
        attn_labels, ranges_q = attn_pattern2labels(ranges, attn_pattern, o.attentions[0].size()[-2:], **kwargs)
        if node_key is None:
            if device is not None: attn_labels = to(attn_labels, device)
            # mean log-likelyhood, lhij->lh
            # matching_scores[attn_pattern] = ((attentions + 1e-4).log() * attn_labels).sum((-2, -1)) / attn_labels.sum()
            matching_scores[attn_pattern] = get_ap_scores(attentions, attn_labels, ranges_q).to('cpu', dtype=torch.float32)
            if averaged: matching_scores[attn_pattern] = matching_scores[attn_pattern].mean(-1)
        else:
            attn_attrs = o.attn_attr[node_key]
                        # def get_score(a): return (a * attn_labels).sum((-2, -1)) / ((a * (a > 0)).sum((-2, -1)) + 1e-9)  # lnij->ln
            def get_score(a):
                a = a * (a > 0)
                return (a * attn_labels).sum((-2, -1)) / (a.sum((-2, -1)) + 1e-9)  # lnij->ln
            matching_scores[attn_pattern] = get_score(attn_attrs * attentions) if isinstance(attn_attrs, torch.Tensor) \
                else OrderedDict(((l, h), get_score(aa * o.attentions[l][0, h])) for (l, h), aa in attn_attrs.items()) # l*[1nij]->ij
    return matching_scores if isinstance(attn_patterns, list) else list(matching_scores.values())[0]

def get_sensitivity(data_tuple, l, h, attn_pattern, node_key, **kwargs):
    text, input_ids, labels, ranges, *_, o = data_tuple
    aw = o.attentions[l][0, h]
    aa = o.attn_attr[node_key][l, h].tril().float()  # fp16->fp32
    
    q, k = attn_pattern.split('->')
    if q != 'bos' and q[-1] not in [']', '+', '-']: q = q + ']'  # make q single token
    k = '*' if 'bos' in q and 'ans' in k and 'ans0' not in k else 'example'
    attn_labels, _ = attn_pattern2labels(ranges, f'{q}->{k}', aw.size(), **kwargs)
    if (attn_labels.sum(1) != 0).sum().item() != len(ranges) - kwargs['k_shot']:  # mqy q is not single-token
       print(f"In get_sensitivity: {attn_pattern}, {(attn_labels.sum(1) != 0).sum().item()} != {len(ranges) - kwargs['k_shot']}")  # q is single-token
    b = attn_labels != 0
    _aa = aa * (aa > 0)
    return ((aw[b] + 1e-5).log() * _aa[b]).sum() / _aa[b].sum()

def abbreviate_label_type(label_type):
    label_type = label_type.replace('_labels', '')
    if ':' in label_type:
        label_type, attn_pattern, k_shot = parse_label_type(label_type)
        return label_type + ':' + abbreviate_attn_pattern(attn_pattern)
    return label_type

from operator import lt, ge

def data2str(data, threshold_score=0.3, verbose=True):
    d = data
    step, topi, layer, head, H, label_type, attn_pattern, attribute_k, top_score = \
        d.step, d.topi, d.layer, d.head, d.H, d.label_type, d.attn_pattern, d.attribute_k, d.top_score
    def wrap(head): return head if head < H else 'm'
    if attn_pattern: ap = abbreviate_attn_pattern(attn_pattern)
    if head is None: return ''  # root
    if not isinstance(layer, Iterable):
        s = f'{layer}-{wrap(head)}'
        if verbose:
            if top_score is not None: s += f' {round(top_score * 100)}'
            if head < H:
                if attn_pattern: s += f' {ap}'
                if d.attr_ap_score is not None: s += f' {round(d.attr_ap_score * 100)}'
                if d.ap_score is not None: s += f'/{round(d.ap_score * 100)}'
                if not is_intermediary(attn_pattern):
                    if d.icl_score is not None: s += f'/{round(d.icl_score * 100)}'
                    if d.sensitivity is not None: s += f'/{d.sensitivity:.1f}'
    elif d.dummy:
        def suffix(score): return f' {round(score*100)}' if verbose and \
            top_score is not None and (lt if True or step < 2 else ge)(score, threshold_score) else ''
        s = ','.join([f'{l}-{wrap(h)}{suffix(score)}' for l, h, score in zip(layer, head,
            top_score if top_score is not None else [None] * len(layer))])
        if attn_pattern is not None:
            s += f' {ap} {round(d.attr_ap_score * 100)}'
    else:
        assert d.mixed
        s = ','.join([f'{l}-{wrap(h)}' for l, h in zip(layer, head)]) if len(layer) < 5 else ''
        if verbose: s += f' {ap}'
    if (not d.mixed and verbose or d.mixed and len(layer) >= 5) and topi is not None:
        s = (f'@:{len(topi)} ' if isinstance(topi, Iterable) and len(topi) > 1 and list(topi) == list(range(topi[-1] + 1))
            else f'@{topi}'.replace(' ', '') + ' ') + s
    if label_type is not None and label_type != 'labels':
        if ':' in label_type:  # simplify label_type->str to ease node dedup by data2str(verbose=False)
            _label_type, _attn_pattern, _ = parse_label_type(label_type.split('/')[0])
            if _attn_pattern == attn_pattern:  # e.g. 'attn_labels:bos->ans0,3/ans0s' -> 'attn_labels/ans0s'
                label_type = _label_type + ('/' + label_type.split('/')[1] if '/' in label_type else '')
        s = s + ' ' + abbreviate_label_type(label_type)
    if attribute_k: s = s + ' ' + 'attr_k'
    return s

def get_head_mlp_attr(attr, mask_last_mlps=True):
    if attr is None: return None
    L, H = attr.head.size()
    mask = torch.ones(L, H + 1)
    if mask_last_mlps: mask[-2:, -1] = 0  # mask last two mlp layers
    return torch.cat([attr.head, attr.mlp.unsqueeze(-1)], dim=1) * mask

def get_matched_head_attr(data):
    return data.attr.head * (reduce(torch.maximum, data.scores.values()).exp() if len(getattr(data, 'scores', {})) > 0 else 1)

def update_attr_data(data, head_attr):
    d = data
    if d.topi is None and not isinstance(d.layer, Iterable):
        d.topi = topi_md(head_attr, d.layer, d.head)
    if d.top_score is None:
        top_score = head_attr[d.layer, d.head] / head_attr.max()
        d.top_score = top_score.numpy() if isinstance(d.layer, Iterable) else top_score.item()
    return d

def _add_node(parent, data, verbose=True):
    # node = Node(data2str(data), parent); node.data = data  # prepend=data.dummy
    # if verbose: print('In _add_node: add', node.name)
    node = {data2str(c.data, verbose=False): c for c in parent.children
            if c.data.dummy == data.dummy and c.data.mixed == data.mixed}.get(
            data2str(data, verbose=False))
    if node is None:
        node = Node(data2str(data), parent); node.data = data  # prepend=data.dummy
        if verbose: print('In _add_node: add', node.name)#; plot_tree(node)
    # else:
    #     # update name in case data.attn_pattern or data2str has been updated
    #     node.name = data2str(data if node.data.attn_pattern != data.attn_pattern else node.data)
    return node

def add_node(parent, layer=None, head=None, head_attr_fn=None, topi=None, label_type=None, attn_pattern=None, 
            H=None, step=None, dummy=False, mixed=False, force=False, verbose=True, **kwargs):
    if parent is None:
        si = -1; node = Node(f' {label_type}' if label_type != 'labels' else '')
        node.data = AttrData(step=si, layer=layer, label_type=label_type)
        return node
    
    if H is None: L, H = parent.data.attr.head.size()
    if parent.data.attr is None and not force:
        print('parent has not been attributed yet, replace it instead of adding to it.')
        _id = id(parent); parent = parent.parent
        parent.children = [child for child in parent.children if id(child) != _id]
    head_attr_fn = head_attr_fn or get_head_mlp_attr  # get_matched_head_attr
    head_attr = head_attr_fn(parent.data.attr)
    if layer is None and topi is not None:
        layer, head = topk_md(head_attr, 10, transpose=True)[topi][:2] if type(topi) == int \
            else np.array(topk_md(head_attr, 10)[:2])[:, topi] # list
    si = parent.data.step
    if attn_pattern is None and label_type: _, attn_pattern, _ = parse_label_type(label_type)
    if step is None: step = si + 1
    data = AttrData(step=step, topi=topi, layer=layer, head=head, H=H, label_type=label_type, 
                    attn_pattern=attn_pattern, dummy=dummy, mixed=mixed, **kwargs)
    if head_attr is not None and not dummy and not mixed: update_attr_data(data, head_attr)
    return _add_node(parent, data, verbose=verbose)

attn_patterns_by_step = {
    -1: ['bos->ans0'],
    0: ['bos->ans]^', 'bos->query', 'bos->ans0+', 'bos->tgt', 'bos->bos^', 'bos->rel', 'bos->cls^',
        'bos->sep',  # 11-4
        'bos->sep+',  # 13-11
        'bos->sep-',
        'bos->query-', # 10-11?
        'bos->tgt+',
        'bos->tgt',
        ],
    1: ['ans]->ans0]', 'ans]->ans0+', 'query->tgt','query->tgt+', 'query->ans0', 'tgt->ans0',
        'ans]->query', 'ans]->query+', 'sep->ans0', 'sep+->ans0', 'sep-->ans0',
        'ans0+->ans0', 'sep+->sep', 'query->ans]^','ans]->bos^', 'query->sep+', 'rel->query',
        'query->ntgts', 'query->ntgts+', 'query->ans0+', 'query->nans0s', 'query->nans0s+', 'query->query-', 'query->sep+', 'query-->ans]',
        'sep+->ans+^', 'sep+->ans]^', 'tgt+->ntgts+', 'tgt+->ntgts', 'ans]->ntgts+','ans]->nans0s',
        'cls->ans', 'cls->query',
        ], # mqy add 'ans0+->ans0', 'sep+->sep', 'query->ans]^','query->sep+', 'rel->query'
    2: ['ans0->tgt', 'ans0->tgt+'], 3: ['tgt+->tgt']
}

ioi_attn_patterns_by_step = {-1: ['bos->ans0'],
    0: [ 'bos->s1', 'bos->ans0+', 'bos->s2', 'bos->s2+'],
    1: [ 's2->s1+', 's2->io','s2->s1'],
    2: ['s1+->s1', 's1+->io+','s1+->io','s2->s1'], 3: ['tgt+->tgt']}

mathlogic_patterns_by_step = {  
    -1: ['bos->cls^',],
    0: ['bos->ans]^', 'bos->query]^','bos->query]', 'bos->bos^', 'bos->rel]', 'bos->cls^','bos->sep', 'bos->sep+', 'bos->sep-','cls->bos','bos->ans0]'],
    1: ['bos->ans]','ans]->query]','ans]->query+','ans]->ans]^','cls->cls+^','bos->query+','bos->cls+^','bos->ans+','ans0]->tgt]','ans0]->tgt+',
        'sep+->sep', 'query]->ans]^','ans]->bos^', 'query]->sep+', 'rel]->query]','ans-->rel]','ans-->query-',
        'query->sep+','sep+->ans+^', 'sep+->ans]^', 'cls->ans]', 'cls->query]','cls->cls^','ans]->ans-','ans-->query','ans]->rel]','ans]->rel+','ans]->query]','query+->bos^',
        ], 
    2: ['query]->query]^','query]->query+^','ans]->cls+^','query+->query]','cls+->cls','ans]->ans0]','rel+->rel]','rel]->query]','query+->ans^','query]->ans0]','query]->tgt]']
    # 2: ['query1-->sep','query1-->query0','query1-->query0+','query1-->tgt0-','query1-->ans+^','query1-->ans]^','query1]->tgt0]^','query0+->query1^','query1]->sep-',
    #     'query0-->query1+^','query0+->query0-','query0+->query0','query0->ans]^','query0-->ans]^','query0-->query1]^','query0-->query0]^','query0->ans0]','bos->bos-'], 
    # 3: ['query0->query0+^','query0->tgt0','query0->tgt0+','query0->sep','ans0]->tgt0','ans0]->tgt0+','query1]->bos^','query1+->query1','query1->nans0s'],
    # 4: ['query0->ans0+','query0->query0^','query0->query0-','ans]->bos','ans]->ans]^','ans]->ans+^','ans]->query1]','ans+->query1]','ans+->ans','ans+->bos','ans+->ans+',]
   
}

mathlogic_patterns_by_step_2 = {  
    -1: ['bos->cls^',],
    0: ['bos->ans]^', 'bos->query]^','bos->query]', 'bos->bos^', 'bos->rel', 'bos->cls^','bos->sep', 'bos->sep+', 'bos->sep-','cls->bos',],
    1: ['bos->ans]','ans]->query]','ans]->query+','ans]->ans]^','cls->cls+^','bos->query+','bos->cls+^',
        'sep+->sep', 'query]->ans]^','ans]->bos^', 'query]->sep+', 'rel->query]',
        'query->sep+','sep+->ans+^', 'sep+->ans]^', 'cls->ans]', 'cls->query]','cls->cls^','ans]->ans-','ans-->query',
        ], 
    2: ['query]->query]^','query]->query+^','ans]->cls+^','query+->query]','cls+->cls',]
}

mathlogic_patterns_by_step_3 = {  
    -1: ['bos->tgt0','bos->ans]^','bos->query]','bos->ans0]','bos->query+'],
    0: ['query]->ans0]','query]->tgt0]','query]->ans]^','query+->query]','query+->ans]^','query+->query+^','ans0]->tgt0]','ans0]->tgt0+'], 
    
}

all_attn_patterns = join_lists(mathlogic_patterns_by_step.values())# + ['bos->bos']  #nrk

def parse_attn_pattern(attn_pattern):
    src, tgt = attn_pattern.split('->')
    for s in ['[', ']', '+', '-', '^']: src, tgt = src.replace(s, ''), tgt.replace(s, '')
    return src, tgt

def is_prominent_range(ranges, label):
    if get_range(ranges[0], label) is None: return False
    if label in ['sep', 'sep-']:
        for field in fields(ranges[0]):
            label0 = field.name
            if label0 not in ['tgt', 'ans0']: continue
            r = ranges[0]
            if all(range_contains(get_range(r, label0), get_range(r, label)) for r in ranges) or \
                all(range_contains(get_range(r, label0 + '+'), get_range(r, label)) for r in ranges):
                return False
    return True

def get_possible_attn_patterns(parent, ranges):
    
    d = parent.data
    if parent.parent is None: src = 'bos'  # root
    elif d.label_type is not None and 'attn_labels' in d.label_type and not d.attribute_k:
        src = d.attn_pattern.split('->')[0]
    else: src = d.attn_pattern.split('->')[1].split(':')[0].replace('^', '')
    return [ap for ap in all_attn_patterns if ap.startswith(src + '->') and
        is_prominent_range(ranges, ap.split('->')[1])] + [f'{src}->{src}'] #if not any(s in src for s in ['ans0s', 'ntgts']) else []) # mqy

def point_wise(attn_pattern):
    src, dst = attn_pattern.split('->')
    return src == dst

def get_label_types(parent, d, k_shot=3, icl_score_thld=0.3): # use fixed k_shot #mqy default icl score: 0.4
    def get_normalized_attn_labels(data):
        return 'attn_labels' + ('/ans0s' if data.attn_pattern.startswith('bos->ans0') else '/example')
    normalized_attn_labels = get_normalized_attn_labels(d)
    #不是自己到自己，父节点是自己到自己，走v通路  nrk
    is_predicting_head = not point_wise(d.attn_pattern) and \
        all(point_wise(n.data.attn_pattern) and 
            n.data.label_type == ('labels' if is_root(n.parent) else None)
            for n in node2path(parent))
    #父节点不是根节点，上一步归q，且q不是自己的到自己，父节点和子节点pattern不同 nrk
    is_recurrent_head = not is_root(parent) and parent.data.label_type in ['attn_labels', get_normalized_attn_labels(parent.data)] and \
        not point_wise(d.attn_pattern) and d.attn_pattern == parent.data.attn_pattern  # e.g. bos->ans0 head below bos->ans0 head
    icl_score = d.icl_score or 0
    # if is_predicting_head or is_recurrent_head: label_types = [normalized_attn_labels]
    if is_root(parent) and point_wise(d.attn_pattern):
        label_types = ['labels']  # e.g. bos->bos at step 0
        
    elif icl_score > icl_score_thld or is_predicting_head or is_recurrent_head or d.attn_pattern in ['ans]->query']: #or d.ap_score >0.99: #mqy
        label_types = ['attn_labels', normalized_attn_labels][:1] if not point_wise(d.attn_pattern) else [None]
        activating_attn_labels = f'attn_labels:{parse_attn_pattern(d.attn_pattern)[0]}->~<s>,{min(3, k_shot)}'
        # head_label_types = get_root(parent).data.head_label_types
        # head_label_type = (d.layer, d.head, activating_attn_labels)
        if point_wise(d.attn_pattern): # or icl_score > icl_score_thld:
            label_types += [activating_attn_labels]
            # if head_label_type not in head_label_types: head_label_types.add(head_label_type) 
        if not is_predicting_head and not is_recurrent_head: label_types += [None]
        if not is_root(parent) and is_predicting_head : label_types += [None] #nrk add
        if not is_root(parent) and d.attn_pattern in ['bos->cls^','bos->query]']: label_types += [None] #nrk add
        
    else:
        label_types = [None]
    # if is_predicting_head: label_types += ['attn_labels:attribute_k']
    # if d.attn_pattern in ['ans]->query', 'query->tgt+', 'query->ans0']:label_types += ['attn_labels:attribute_k'] # mqy add k attribution
    # if is_predicting_head and d.attn_pattern.startswith('bos->query'): label_types += ['labels']  # mqy TODO
    # if d.step == 0 and d.attn_pattern in ['bos->query]','bos->rel+']:
    #     label_types +=['labels']
    if (d.layer,d.head) in [(13,15)]: label_types += ['attn_labels'] #nrk add
    return label_types

def get_icl_score(ap_scores, k_shot=3):
    # k_shot = min(3, k_shot)
    if k_shot > 1: k_shot = 3  # use fixed k_shot
    _ap_score = ap_scores[k_shot:].mean() 
    return ((_ap_score - ap_scores[0]) / (_ap_score + 1e-9)).item()  #mqy fix icl_score NaN: add 1e-9

def expand_node(data_tuples, parent, head_attr_fn=None, topk=10, threshold_score=0.3, k_shot=None,
                attribute_k=False, add_dummy_nodes=True, verbose=True):
    pd = parent.data; L, H = pd.attr.head.size()
    head_attr_fn = head_attr_fn or get_head_mlp_attr
    layers, heads, scores = topk_md(head_attr_fn(pd.attr), topk)
    assert scores[0] > 0, f'{parent.name} {scores}'
    scores = scores / scores[0]

    def reduce_fn(scores): return sum(scores) / len(scores)
    def get_score(ap, l, h): return pd.attr_ap_scores[ap][l, h].item() if h < H else float(point_wise(ap))
    ap2score = OrderedDict(sorted([(ap, reduce_fn([get_score(ap, l, h) for l, h in zip(layers, heads)]))
                    for ap in pd.all_attn_patterns], key=lambda x: x[1], reverse=True))

    top_data = []
    for i, (layer, head, score) in enumerate(zip(layers, heads, scores)):
        if score < threshold_score: continue 
        d = AttrData(step=pd.step + 1, topi=i, layer=layer, head=head, H=H, top_score=score)
        d._attn_pattern, d._attr_ap_score = 'unk', 0.  # used for sorting
        if head < H:
            d.attn_pattern, d.attr_ap_score = max([(ap, v[layer, head].item())
                for ap, v in pd.attr_ap_scores.items()], key=lambda x: x[1])#by nrk get most likely pattern and attr_attn_score 
            node_key = node2key(parent)
            kwargs = dict(k_shot=k_shot, attribute_k=attribute_k)
            attr_ap_scores = [get_head_matching_scores(dt, d.attn_pattern,
                node_key=node_key, **kwargs)[layer, head].item() for dt in data_tuples]
            assert all(s >= 0 for s in attr_ap_scores), str(attr_ap_scores)
            # if not all(s > 0 for s in attr_ap_scores):
            #     print('Warning! abnormal attr_ap_scores',
            #         f'step{d.step} {node_key} {layer}-{head} {d.attn_pattern} {attr_ap_scores}')

            d.ap_score = get_root(parent).data.ap_scores[d.attn_pattern][layer, head].item()
            if k_shot > 0: # not ioi task
                d.icl_score = 0 if is_intermediary(d.attn_pattern) or is_relating(d.attn_pattern) else \
                    get_icl_score(mr(get_head_matching_scores)(data_tuples, d.attn_pattern,
                    layer=layer, head=head, averaged=False, k_shot=0), k_shot=k_shot)
                # d.sensitivity = mr(get_sensitivity)(data_tuples, layer, head,
                #     d.attn_pattern, node2key(parent), k_shot=k_shot).item()
        else:  # mlp
            # x->x, convenient for get_possible_attn_patterns and point_wise
            d.attn_pattern, d.ap_score = pd.all_attn_patterns[-1], 1.
        if head == H or ap2score[d.attn_pattern] > list(ap2score.values())[0] / 10:
            d._attn_pattern, d._attr_ap_score = d.attn_pattern, ap2score[d.attn_pattern]
        top_data.append(d)
    top_data = sorted(top_data, key=lambda d: d._attr_ap_score, reverse=True)
    # parent.children = [c for c in parent.children if not c.data.dummy and not c.data.mixed]

    if True: #pd.step < 1 or pd.step == 1 and pd.attn_pattern in ['bos->tgt', 'bos->query']:
        for d in top_data:
            # if d.top_score < threshold_score: continue
            label_types = get_label_types(parent, d, k_shot=k_shot)
            for label_type in label_types:
                _d = deepcopy(d)
                _attribute_k = label_type and label_type.endswith(':attribute_k')
                if label_type: label_type = label_type.replace(':attribute_k', '')
                _d.label_type = label_type
                _d.attribute_k= _attribute_k or attribute_k and label_type and 'attn_labels' in label_type
                _add_node(parent, _d, verbose=verbose)

            # remove settled and unused attn_attr to save memory
            if False and d.head < H and 'attn_labels' not in label_types and d.attr_ap_score > 0.3:
                for *_, o in data_tuples:
                    attn_attr = o.attn_attr[node2key(parent)]
                    if (d.layer, d.head) in attn_attr: del attn_attr[d.layer, d.head]
                # print('In expand_node: remove attn_attr', data2str(d))
                
        if False and pd.step == -1:
            top_attn_pattern = [ap for ap in ap2score if not point_wise(ap)][0] # e.g. 'bos->ans0]' for step -1
            layers, heads, scores = topk_md(pd.ap_scores[top_attn_pattern], 5) #topk // 2)
            label_types = [f'attn_labels:{top_attn_pattern},{k_shot}/ans0s']  # 'attn_labels:bos->ans0],3'
            label_types += ['attn_labels:bos->~<s>,3']

            for layer, head, score in zip(layers, heads, scores):
                if (layer, head) in {(d.layer, d.head) for d in top_data}: continue
                attr_ap_score = None
                if (layer, head) in pd.top_heads:
                    attn_pattern0, attr_ap_score0 = max([(ap, v[layer, head].item())
                        for ap, v in pd.attr_ap_scores.items()], key=lambda x: x[1])
                    if attn_pattern0 == top_attn_pattern: attr_ap_score = attr_ap_score0
                icl_score = get_icl_score(mr(get_head_matching_scores)(data_tuples, top_attn_pattern,
                    layer=layer, head=head, averaged=False, k_shot=0), k_shot=k_shot) if k_shot > 0 else 0
                for label_type in label_types + (['attn_labels:bos->~<s>,3'] if icl_score > 0.4 else []):
                    add_node(parent, layer=layer, head=head, head_attr_fn=head_attr_fn, label_type=label_type,
                        attn_pattern=top_attn_pattern, attr_ap_score=attr_ap_score,
                        ap_score=score, icl_score=icl_score, verbose=verbose,
                        _attn_pattern=top_attn_pattern, _attr_ap_score=ap2score[top_attn_pattern],
                        attribute_k=attribute_k and label_type and 'attn_labels' in label_type)

        # sort again new children and maybe existing old children together by _attr_ap_score
        parent.children = sorted(parent.children, key=lambda c: c.data.topi)
        parent.children = sorted(parent.children, key=lambda c: c.data._attr_ap_score, reverse=True)

    if add_dummy_nodes:
        n_chilren = len(parent.children)
        for ap, data_group in groupby(top_data, key=lambda d: d._attn_pattern):
            data_group = list(data_group)
            add_node(parent, attn_pattern=ap, dummy=True, verbose=verbose,
                attr_ap_score=ap2score.get(ap, 0),  # score 0 for 'unk' not in ap2score
                **{k: [getattr(d, k) for d in data_group] for k in ['layer', 'head', 'topi', 'top_score']})
        parent.children = parent.children[n_chilren:] + parent.children[:n_chilren]  # move dummy nodes to top

def filter_fn(p, c):  
    if c.layer == 0 or c.head == c.H: return False
    pap, ap = abbreviate_attn_pattern(p.attn_pattern or ''), abbreviate_attn_pattern(c.attn_pattern)
    #return (p.step == -1 and c.ap_score > 0.15 and c.top_score > 0.4 and c.icl_score > 0.05 or # and ap.startswith('B->A0') or
    #    p.step == 0 and c.top_score > 0.5 or #and (ap.startswith('B->Q') or ap.startswith('B->A')) or # c.ap_score > 0.2 and c.top_score > 0.5 and c.icl_score > 0.4 or 
    #    p.step == 1 and ((ap.startswith('B->Q') and c.top_score>0.15) or ap.startswith('B->A]') or ap.startswith('Q->A0')) or
    #    p.step == 2 and ap.startswith('B->A]') 
    #)

    # child filter fn
    return (p.step == -1 and c.ap_score > 0.14 and c.top_score > 0.5 and c.icl_score > 0.05 or # and ap.startswith('B->A0') or
        p.step == 0 and c.top_score > 0.9 or #and (ap.startswith('B->Q') or ap.startswith('B->A')) or # c.ap_score > 0.2 and c.top_score > 0.5 and c.icl_score > 0.4 or 
        p.step == 1 and ((ap.startswith('B->Q') and c.top_score>0.55) or ap.startswith('B->A]') or ap.startswith('Q->A0')) or
        p.step == 2 and (ap.startswith('A]->Q]') or ap.startswith('A]->A0]') or ap.startswith('Q->T'))
    )
        # p.step == 0 and pap.startswith('B->A0') and (ap.startswith('B->A0') and c.ap_score > 0.4 or ap == 'B->B' and c.ap_score > 0.6 or ap in ['B->Q', 'B->A]']) or \
    # return  p.step < 1 or (p.step == 1 and ap in ['B->s2'] ) or (p.step == 2 and ap in ['s2->s1+','s2->s1'])  # ioi task

def attribute_tree(data_tuples, model, node, max_step, filter_fn, topk=10, threshold_score=0.3,
                k_shot=None, attribute_k=False, mix=True, device=None, verbose=True):
    blocks = model.transformer.h
    L, H = len(blocks), blocks[0].attn.num_heads
    device = device or model.lm_head.weight.device
    d = node.data; node_key = node2key(node)
    *_, o =  data_tuples[0]
    _topk = topk # round(topk * (d.layer if not isinstance(d.layer, Iterable) else max(d.layer)) / L)
    
    attributed_heads = get_root(node).data.attributed_heads
    attributed_head = (str(d.layer), str(d.head), d.attn_pattern, d.label_type, d.attribute_k)
    if attributed_head in attributed_heads:
        print('In attribute_tree: already attributed, skip', attributed_head)
    if len(node.children) == 0 and attributed_head not in attributed_heads:
        with Timer(f'In attribute_tree: attribute_step {node.name}'):
            d.attr = mr(attribute_step)(data_tuples, model, node)
        d.top_heads = list(zip(*topk_md(get_head_mlp_attr(d.attr), _topk)[:2]))
        d.top_heads = [(l, h) for l, h in d.top_heads if h < H]

        if len(d.top_heads) > 0:
            if True: # with Timer(f'In attribute_tree: attribute_step stage2 {node.name}'):
                model.wos = torch.stack([rearrange(blocks[l].attn.out_proj.weight.data,
                    'e (n d) -> n d e', n=H)[h] for l, h in d.top_heads])  # k*[de]->kde, used by sum_forward
                # resulting attr.attn saved to o.attn_attr
                mr(attribute_step)(data_tuples, model, node, attr_heads=d.top_heads)
                del model.wos

        text, input_ids, labels, ranges, *args, o = data_tuples[0]
        d.all_attn_patterns = get_possible_attn_patterns(node, ranges)
        if len(d.top_heads) > 0:
            kwargs = dict(k_shot=k_shot, attribute_k=attribute_k)
            ap_scores = get_root(node).data.ap_scores
            _attn_patterns = [ap for ap in d.all_attn_patterns if ap not in ap_scores]
            if True: #with Timer('get ap_scores'):
                ap_scores.update(mr(get_head_matching_scores)(
                    data_tuples, _attn_patterns, device=device, **kwargs))
            if True: #with Timer('get attr_ap_scores'):
                d.attr_ap_scores = OrderedDict([(ap, mr(get_head_matching_scores)(
                    data_tuples, ap, node_key=node_key, **kwargs)) for ap in d.all_attn_patterns])
        else:  # all mlps
            d.all_attn_patterns = d.all_attn_patterns[-1:]  # 'x->x'

        if d.step == -1:
            # with Timer(f'In attribute_tree: logit_attribute_step {node.name}'):
            #     d.logit_attr = mr(attribute_step)(data_tuples, model, node, do_logit_attribution=True)
            # pprint(topk_md(get_head_mlp_attr(d.logit_attr), 10))
            if 'bos->ans0' in d.ap_scores: pprint(topk_md(d.ap_scores['bos->ans0'], 10))
        expand_node(data_tuples, node, topk=_topk, threshold_score=threshold_score,
            k_shot=k_shot, attribute_k=attribute_k, add_dummy_nodes=True, verbose=verbose)

        if d.label_type is not None: attributed_heads.add(attributed_head)

    if d.step == max_step: return
    children = [c for c in node.children if not c.data.dummy and not c.data.mixed and filter_fn(d, c.data)]
    if mix:
        _children = []
        keys = ['attn_pattern', 'label_type', 'attribute_k']
        def wrap_none(value): return value if value is not None else '_'  # for label_type is None
        def key_fn(node): return tuple([wrap_none(getattr(node.data, k)) for k in keys])
        for _, node_group in groupby(sorted(children, key=key_fn), key=key_fn):
            node_group = list(node_group); node0 = node_group[0]
            child = add_node(node, mixed=True, verbose=verbose, **{k: getattr(node0.data, k) for k in keys},
                **{k: [getattr(n.data, k) for n in node_group] for k in ['layer', 'head', 'topi']})
            _children.append(child)
        children = _children

    for c in children:
        attribute_tree(data_tuples, model, c, max_step, filter_fn, topk=topk, threshold_score=threshold_score,
            k_shot=k_shot, attribute_k=attribute_k, mix=mix, device=device, verbose=verbose)

def attribute_tree_on(data_tuples, model, node, max_step, filter_fn, device='cpu', **kwargs):
    
    if node is None: node = add_node(None, label_type='labels')
    try:
        if device is not None and device != 'cpu': switch_model_to(model, device)
        with Timer('attribute_tree'):
            attribute_tree(data_tuples, model, node, max_step, filter_fn, **kwargs)
    finally:
        if device is not None and device != 'cpu': switch_model_to(model, 'cpu')
    return node

def node2key(node):
    return ' > '.join(data2str(n.data, verbose=False)
        for n in reversed(node2path(node, stop_at_label=False)))

# depth-first, same as the order of attribute_step (rather than add_node!) in attribute_tree
# when generating the entire tree at once (as opposed to step by step, which is breadth-first)
def traverse_tree(node, fn, include_dummy=False):
    fn(node)
    for child in node.children:
        if include_dummy or not child.data.dummy:
            traverse_tree(child, fn, include_dummy=include_dummy)

def attn_attr2array(tree_nodes, attn_attr, scale_factor=1000.):
    assert len(tree_nodes) == len(attn_attr), f'{len(tree_nodes)} != {len(attn_attr)}'
    a = torch.stack([torch.stack([attn_attr[node2key(node)][lh]
        for lh in node.data.top_heads]) for node in tree_nodes]) # (n_nodes, topk, i, j)
    return (a * scale_factor).to(torch.float16).numpy()

def array2attn_attr(tree_nodes, array, scale_factor=1000.):
    array = torch.from_numpy(array).to(torch.float32) / scale_factor
    assert len(tree_nodes) == array.size(0), f'{len(tree_nodes)} != {array.size(0)}'
    assert all(len(node.data.top_heads) == array.size(1) for node in tree_nodes)
    return OrderedDict((node2key(node), OrderedDict((lh, array[i, j])
        for j, lh in enumerate(node.data.top_heads)))
        for i, node in enumerate(tree_nodes))

def dump_attn_attrs_to_arrays(root, data_tuples):
    nodes = []
    def fn(node):
        if node.data.top_heads is not None: nodes.append(node)
    traverse_tree(root, fn)
    return [attn_attr2array(nodes, dt[-1].attn_attr) for dt in data_tuples]

def load_attn_attrs_from_arrays(root, data_tuples, arrays):
    nodes = []
    def fn(node):
        if node.data.top_heads is not None: nodes.append(node)
    traverse_tree(root, fn)
    for dt, array in zip(data_tuples, arrays):
        dt[-1].attn_attr = array2attn_attr(nodes, array)

results_dir = '/mnt/nvme1/xd/results'

def save_attribution_results(result, res_key, num_attn_attrs=None):
    if num_attn_attrs is None: num_attn_attrs = len(result.data_tuples)
    _root = deepcopy(result.root)
    # asdict is recursive. convert data and data.attr dataclass objs to dict for pickling
    def fn(node): node.data = dataclasses.asdict(node.data)
    traverse_tree(_root, fn, include_dummy=True)
    pickle.dump(_root, gzip.open(f'{results_dir}/{res_key}_tree.pkl.gz', 'wb'))

    fpath = f'{results_dir}/{res_key}_attn_attrs.npz'
    arrays = dump_attn_attrs_to_arrays(result.root, result.data_tuples[:num_attn_attrs])
    print(f'In save_attribution_results: attn_attr_array.shape = {arrays[0].shape}')
    np.savez_compressed(fpath, *arrays)

def load_attribution_results(result, res_key, load_attn_attrs=True, num_attn_attrs=None):
    if num_attn_attrs is None: num_attn_attrs = len(result.data_tuples)
    if result.root is None:
        fpath = f'{results_dir}/{res_key}_tree.pkl.gz'
        if not os.path.isfile(fpath): return
        result.root = pickle.load(gzip.open(fpath, 'rb'))
        def fn(node):
            node.data = AttrData(**node.data)  # dict->dataclass
            # data.attr has also been recursively converted to dict by asdict. convert it back
            if node.data.attr is not None: node.data.attr = Attributions(**node.data.attr)
        traverse_tree(result.root, fn, include_dummy=True)

    if load_attn_attrs:
        fpath = f'{results_dir}/{res_key}_attn_attrs.npz'
        if not os.path.isfile(fpath): return
        arrays = np.load(fpath)
        load_attn_attrs_from_arrays(result.root, result.data_tuples[:num_attn_attrs],
            [arrays[f'arr_{i}'] for i in range(num_attn_attrs)])

def has_attribution_results(res_key):
    return all(os.path.isfile(fpath) for fpath in [f'{results_dir}/{res_key}_tree.pkl.gz',
                                                f'{results_dir}/{res_key}_attn_attrs.npz'])

def map_nodes(nodes, root):
    nodes[node2key(root)] = root
    for child in root.children: map_nodes(nodes, child)
    
def remove_node(nodes, key):
    if key not in nodes: return
    node = nodes[key]

    for child in node.children:
        remove_node(nodes, data2str(child.data, verbose=False))
    
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
            plot_attrs(dict(scores, head_attr=get_head_mlp_attr(data.attr),
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
        attn_pattern=d.attn_pattern, outputs=outputs, attn_attr=outputs.attn_attr.get(node2key(node.parent)),
        hidden_states0=hidden_states0, ranges=ranges, attribute_k=d.attribute_k, scaled=True)

def is_root(node): return node.parent is None

def get_root(node):
    n = node
    while n.parent is not None: n = n.parent
    return n

def node2path(node, stop_at_label=True, except_last=False):
    nodes = []
    while node.parent is not None:
        nodes.append(node)
        if node.data.label_type is None or not stop_at_label: node = node.parent
        else: break
    return nodes if not except_last else nodes[:-1]

def path2fns(node, node2fn, except_last=False):
    return [node2fn(n) for n in node2path(node, except_last=except_last)]

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

def _plot_attn(attn, tokens, ytokens=None, ystart=None, ystop=None, y_pos=None, x_pos=None, topk=None,
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
            a = max(s1 / s2 for s1, s2 in zip(figsize, figsize2))  # min
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
    kwargs = dict(linewidths=0.5 * 2, color='grey')
    if y_pos is not None: ax.hlines(y=y_pos, xmin=0, xmax=attn.size(1)-0.5*use_imshow, **kwargs)  # max-0.5 for imshow
    if x_pos is not None: ax.vlines(x=x_pos, ymin=0, ymax=attn.size(0)-0.5*use_imshow, **kwargs)
    # plt.show()

def plot_attn(data_tuple, tokenizer, l, h, attn_pattern=None, k_shot=0):
    text, input_ids, labels, ranges, *args, o = data_tuple
    a = o.attentions[l][0, h]; aw_size = a.size()
    tokens = [t.replace('Ġ', '').replace('Ċ', '-'*12) 
        for t in tokenizer.convert_ids_to_tokens(input_ids[0])]
    bos_indices = args[1]
    ystart, ystop = bos_indices[k_shot], aw_size[0]
    attn_labels = attn_pattern2labels(ranges, attn_pattern, aw_size)
    y_pos, x_pos = attn_labels.nonzero().T
    _plot_attn(a, tokens, ystart=ystart, ystop=ystop, y_pos=y_pos, x_pos=x_pos,
        fontsize=9, transpose=True, figsize=(15, 15)); plt.show()  # bij->ij

def plot_attn_attr(data_tuple, model, tokenizer, node, l, h, attn_patterns=None, is_relating_head=False,
                k_shot=0, attribute_k=False, plot_attr=True):
    device = model.lm_head.weight.device
    text, input_ids, labels, ranges, *args, o = data_tuple
    labels = labels.to(device)
    tokens = [t.replace(tokenizer.newword_prefix, ' ').replace(tokenizer.newline_token, '-'*12)
        # if not isinstance(tokenizer, LlamaTokenizer) else t
        for t in tokenizer.convert_ids_to_tokens(input_ids[0])]
    aw_size = o.attentions[0][0, 0].size()  # l*[bnij]->ij
    ystart, ystop = 0, aw_size[0]
    if plot_attr and k_shot > 1:  # not ioi task
        bos_indices = args[1]
        if not attribute_k and attn_patterns and (is_relating(restore_attn_pattern(attn_patterns[0])) or is_relating_head):
            ystop = bos_indices[len(ranges) - k_shot] + 3
        else:
            ystart = bos_indices[k_shot]
    if plot_attr:
        H = o.attentions[0].size(1)
        fns = path2fns(node, partial(node2fn, outputs=o, ranges=ranges, labels=labels))
        fwd_fn = partial(sum_forward, outputs=o, output_layer=l)
        label_type = 'labels' if len(fns) == 0 else None
        fn = partial(mixed_forward, layer=l, head=h, labels=labels, label_type=label_type, outputs=o, ranges=ranges)
        keys = ['embed_mask', 'mlp_mask', 'head_mask']
        to_layer = max(l) if isinstance(l, Iterable) else l
        x = OrderedDict((key, get_x(key, o, to_layer=to_layer)) for key in keys)
        *_, ys, logits = attribute(fwd_fn, model, x, [fn] + fns, num_points=3+4, forward_only=True)
        print('scaled_logprobs =', ys)
        if isinstance(logits, (list, tuple)): logits = sum(logits)
        if logits is None: pass
        elif logits.size(-1) == model.lm_head.out_features:  # lm_logits
            # logits = logits + o.logits_mask #* (1e4 if logits.dtype == torch.float16 else 1e9)
            ap_scores, *_ = show_predictions(tokenizer, *args, logits=logits[-1:], labels=labels,
                k_shot=k_shot, topk=4, loss_reduction='none', sep='\t')
        elif logits.size(-1) == input_ids.size(1): # attn_logits, bij
            attn_pattern = node2path(node)[-1].data.attn_pattern or 'bos->ans0'
            attn_labels, ranges_q = attn_pattern2labels(ranges, attn_pattern, aw_size, k_shot=0)
            # y_pos, x_pos = attn_labels.nonzero().T
            aw = logits[-1].softmax(dim=-1)
            ap_scores = get_ap_scores(aw, attn_labels.to(device), ranges_q)
            print('resulting ap_scores =', ap_scores, ap_scores.mean(-1))
            # _plot_attn(aw, tokens, ystart=ystart, ystop=ystop, y_pos=y_pos, x_pos=x_pos,
            #     fontsize=9, transpose=True, figsize=(20-5, 20-5)); plt.show()  # bij->ij
        if isinstance(h, Iterable) or h == H: return ap_scores

    aw = o.attentions[l][0, h]; attns = [aw]
    if plot_attr:
        aa = o.attn_attr[node2key(node)][l, h] * aw if (l, h) in o.attn_attr[node2key(node)] else aw
        attns += [aa]

    y_pos, x_pos = None, None
    if attn_patterns is not None:
        all_attn_labels = [attn_pattern2labels(ranges, ap, aw_size,
            k_shot=0, attribute_k=attribute_k) for ap in attn_patterns]
        if len(all_attn_labels) == 1:
            attn_labels, ranges_q = all_attn_labels[0]
        else:  # steps that have multiple attn patterns, e.g. ['ans->ans0]', 'ans->ans0+'] of relating heads
            attn_labels, ranges_q, _ = max([(al, rq, (al * aa).sum()) for al, rq in all_attn_labels],
                key=lambda x: x[-1])[:2] # find the pattern that best matches aa
        y_pos, x_pos = attn_labels.nonzero().T
        ap_scores = get_ap_scores(aw, attn_labels, ranges_q)
        print('ap_scores =', ap_scores, ap_scores.mean(-1))
    if True: #with Timer():
        _, axs = plt.subplots(1, len(attns), sharex=False, sharey=False,
                    figsize=(20*len(attns), 20/(ystop-ystart)*aw_size[0]))
                    # figsize=(20*len(attns)*(ystop-ystart)/aw_size[0], 20))
        if len(attns) == 1: axs = [axs]
        for ax, a in zip(axs, attns):
            fontsize = 30 if ystop - ystart == aw_size[0] else 10 + 10
            _plot_attn(a, tokens, ax=ax, ystart=ystart, ystop=ystop, y_pos=y_pos, x_pos=x_pos,
                fontsize=fontsize, transpose=True, use_imshow=False)
        plt.show()
    return ap_scores

def plot_attn_attrs(data_tuples, model, tokenizer, node, layer=None, head=None, topi=[0], 
                head_attr_fn=None, attn_patterns=None, mix=False, **kwargs):
    head_attr_fn = head_attr_fn or get_head_mlp_attr  # or get_matched_head_attr
    def wrap(l): return l if isinstance(l, list) else [l]
    heads = np.array(topk_md(head_attr_fn(node.data.attr), 20)[:2])[:, topi] \
        if layer is None else [wrap(layer), wrap(head)]
    heads = zip(*heads) if not mix else [heads]
    for l, h in heads:
        s = f'{l}-{h}' if not mix else ' + '.join([f'{_l}-{_h}' for _l, _h in zip(l, h)])
        print(' -> '.join([s] + [n.name for n in node2path(node)]))
        ap_scores = mr(plot_attn_attr)(data_tuples, model, tokenizer, node, l, h, attn_patterns=attn_patterns, **kwargs)
        print('reduced_ap_scores =', ap_scores, ap_scores.mean())

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
