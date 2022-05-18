from __future__ import annotations
from typing import Optional
import dataclasses
from dataclasses import dataclass, field


from functools import partial
from collections import defaultdict, OrderedDict, Counter
import types
import math
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

def topk_md(tensor, k, largest=True, zipped=False):
    if tensor.ndim == 1:
        values, indices = tensor.topk(k, largest=largest)
        return indices.numpy(), values.numpy()
    values, indices = tensor.flatten().topk(k, largest=largest)
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    rows, cols = np.unravel_index(indices.numpy(), tensor.shape)
    return (rows, cols, values.numpy()) if not zipped else list(zip(rows, cols, values.numpy()))

def to_df(*args): return pd.DataFrame(list(zip(*args)))

def norm(tensor, p=2): return tensor.norm(p=p, dim=-1).mean().round().item()

def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

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
        answers.append(ans_ids)
    return bos_indices, eos_indices, answers, labels

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

def my_isinstance(obj, type):  # to cope with autoreload
    # return isinstance(obj, type)  # fail on autorelaod
    return obj.__class__.__name__ == type.__name__ 

def equal(a, b):
    assert a.size() == b.size()
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

def reduce_objects(objs, fields, reduce_fn='mean'):
    obj = dataclasses.replace(objs[0])
    for field in fields:
        setattr(obj, field, sum(getattr(o, field) for o in objs) / (len(objs) if reduce_fn == 'mean' else 1))
    return obj