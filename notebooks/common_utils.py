from __future__ import annotations
from typing import Optional
import dataclasses
from dataclasses import dataclass, is_dataclass, asdict

import operator
from operator import getitem, setitem
from functools import partial
from collections import defaultdict, OrderedDict, Counter
from collections.abc import Iterable
import types
import math
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
# torch.nn.LayerNorm
# from transformers import GPT2Tokenizer

# cache_dir = '/nas/xd/.cache/torch/transformers/'
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

def join_lists(x): return list(chain.from_iterable(x))

def reverse(l): return list(reversed(l))

#保留小数
def numpy(a, decimals=None):
    v = np.array(a) if isinstance(a, list) else a.detach().cpu().numpy()
    if decimals is not None: v = v.round(decimals)
    return v

def show_topk(values, indices, values_fn=lambda x: numpy(x, decimals=3), indices_fn=None):
    if indices_fn is None:
        indices_fn = show_topk.indices_fn if getattr(show_topk, 'indices_fn', None) is not None else lambda x: x
    return dict(OrderedDict(zip(indices_fn(indices), values_fn(values))))

def topk_md(tensor, k, largest=True, transpose=False):
    k = min(tensor.numel(), k)
    if tensor.ndim == 1:
        values, indices = tensor.topk(k, largest=largest)
        return indices.numpy(), values.numpy()
    values, indices = tensor.flatten().topk(k, largest=largest)
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    rows, cols = np.unravel_index(indices.numpy(), tensor.shape)
    return (rows, cols, values.numpy()) if not transpose else list(zip(rows, cols, values.numpy()))

def to_df(*args): return pd.DataFrame(list(zip(*args)))

def norm(tensor, p=2): return tensor.norm(p=p, dim=-1).mean().round().item()

def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

def convert_ids_to_tokens(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    wrapped = False
    if type(tokens) == str: tokens, wrapped = [tokens], True
    out = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    if wrapped: out = out[0]  # unwrap
    return out

def mask_logits(logits, indices, kept_ids):
    mask = torch.ones_like(logits) * (-1e9) # biv
    for i, ids in zip(indices, kept_ids): mask[:, i, ids] = 0
    return logits + mask

def my_isinstance(obj, type_):  # to cope with autoreload
    # return isinstance(obj, type_)  # fail on autorelaod
    return obj.__class__.__name__ == type_.__name__ if not isinstance(type_, tuple) \
        else any(obj.__class__.__name__ == t.__name__ for t in type_)

def equal(a, b):
    assert a.size() == b.size(), f'{a.size()} != {b.size()}'
    return (a - b).abs().mean(), a.abs().mean(), b.abs().mean()

def einsum(
    eq: str,
    *tensors: torch.Tensor,
) -> torch.Tensor:
    """Drop dimensions of size 1 to allow broadcasting.
    from https://github.com/pytorch/pytorch/issues/48420"""
    lhs, rhs = eq.split("->")
    # squeeze
    mod_ops, mod_t = [], []
    for op, t in zip(lhs.split(","), tensors):
        mod_op = ""
        for i, c in reversed(list(enumerate(op))):
            if t.shape[i] == 1:
                t = t.squeeze(dim=i)
            else:
                mod_op = c + mod_op
        mod_ops.append(mod_op)
        mod_t.append(t)
    m_lhs = ",".join(mod_ops)
    r_keep_dims = set("".join(mod_ops))
    m_rhs = "".join(c for c in rhs if c in r_keep_dims)
    m_eq = f"{m_lhs}->{m_rhs}"
    mod_r = torch.einsum(m_eq, *mod_t)
    # unsqueeze
    for i, c in enumerate(rhs):
        if c not in r_keep_dims:
            mod_r = mod_r.unsqueeze(dim=i)
    return mod_r

def binarize(a, fraction=0.33):
    b = torch.zeros_like(a)
    prev_v = None
    for r, c, v in zip(*topk_md(a, 20)):
        if prev_v is not None and abs(v) < abs(prev_v) * fraction: break
        b[r, c] = 1; prev_v = v
    return b

# adapted from: https://dzone.com/articles/python-timer-class-context
# from timeit import default_timer
from datetime import datetime

class Timer(object):
    def __init__(self, msg='', verbose=True):
        self.verbose = verbose
        # self.timer = default_timer
        self.msg = msg
        
    def __enter__(self):
        if self.verbose: print(self.msg, '...', end=' ')
        self.start = datetime.now() # self.timer()
        return self
        
    def __exit__(self, *args):
        end = datetime.now() # self.timer()
        self.elapsed = str(end - self.start)#.split('.')[0]
        # self.elapsed_secs = end - self.start
        # self.elapsed = self.elapsed_secs #* 1000   # millisecs
        if self.verbose: print('done', self.elapsed)
            # print('elapsed time: %d s' % self.elapsed)

# https://stackoverflow.com/questions/2461170/tree-implementation-in-python
# @dataclass
# class Node:
#     data: tuple = ()
#     parent: Optional[Node] = None
#     children: list[Node] = field(default_factory=list)

def traverse(node, fn):
    fn(node)
    for child in node.children: traverse(child, fn)

def reduce_objects(objs, fields=None, reduce_fn='mean'):
    if not isinstance(objs, list): objs = list(objs)
    denorm = len(objs) if reduce_fn == 'mean' else 1
    if isinstance(objs[0], torch.Tensor) or not isinstance(objs[0], dict) and not hasattr(objs[0], '__dict__'):
        # print('In reduce_objects:', [type(o) for o in objs])
        return sum(objs) / denorm
    
    obj, _fields, getter, setter = ({}, objs[0].keys(), getitem, setitem) if isinstance(objs[0], dict) else \
        (type(objs[0])(), objs[0].__dict__.keys(), getattr, setattr)   # dict or (data)class object
    _fields = [f for f in _fields if getter(objs[0], f) is not None]
    fields = set(_fields).intersection(set(fields)) if fields is not None else _fields
    for field in fields: setter(obj, field, sum(getter(o, field) for o in objs) / denorm)
    # obj = dataclasses.replace(objs[0])
    # for field in fields:
    #     if getattr(objs[0], field, None) is None: continue
    #     setattr(obj, field, sum(getattr(o, field) for o in objs) / (len(objs) if reduce_fn == 'mean' else 1))
    return obj

def iterable(item):
    return isinstance(item, Iterable) and not isinstance(item, (tuple, str, torch.Tensor))

def pad(input, dim, to_len, pad_left=False, **kwargs):
    if input.size(dim) == to_len: return input
    padding = [0] * ((input.ndim - dim) * 2)
    padding[-1-int(pad_left)] = to_len - input.size(dim)
    return F.pad(input, tuple(padding), **kwargs)

def sum_except(input, dims):
    if isinstance(dims, int): dims = [dims]
    dims = [d for d in range(input.ndim) if d not in dims]
    return input.sum(dims) if len(dims) > 0 else input

def maybe_map(fn, *iters):
    if iterable(iters[0]):
        res = list(map(fn, *iters))
        return {k: [d[k] for d in res] for k in res[0]} if isinstance(res[0], dict) else res
    return fn(*iters)

def lget(l, i, default=None): return l[i] if len(l) > i else default

import inspect
def get_default_value(fn, name):
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    if isinstance(fn, types.FunctionType): return inspect.signature(fn).parameters[name].default
    assert isinstance(fn, partial); return fn.keywords[name]