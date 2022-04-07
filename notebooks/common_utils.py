from functools import partial
from collections import defaultdict, OrderedDict, Counter
import types
import math
import numpy as np
from itertools import chain
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

def show_topk(values, indices, values_fn=lambda x: numpy(x, decimals=3), indices_fn=lambda x: x):
    return dict(OrderedDict(zip(indices_fn(indices), values_fn(values))))

def topk_md(tensor, k, largest=True):
    if tensor.ndim == 1:
        values, indices = tensor.topk(k, largest=largest)
        return indices.numpy(), values.numpy()
    values, indices = tensor.flatten().topk(k, largest=largest)
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    rows, cols = np.unravel_index(indices.numpy(), tensor.shape)
    return rows, cols, values.numpy()

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
        eos_indices = bos_indices[1:] + [input_ids.size(1)]
    # labels = torch.ones(input_ids.size(0), input_ids.size(1) - 1).long() * (-100)
    labels = torch.ones_like(input_ids) * (-100)
    answers = []
    # print(bos_indices,eos_indices)
    for bos_i, eos_i in zip(bos_indices, eos_indices):
        # eos_i = bos_i + 2  # show only the first answer token
        ans_ids = input_ids[0, bos_i + 1: eos_i]
        # print(ans_ids)
        labels[0, bos_i: eos_i - 1] = ans_ids
        answers.append(ans_ids)
    return bos_indices, eos_indices, answers, labels

def get_prob_dist(d, topk=5):
    return {k: round(math.exp(v), 3) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk]}

def convert_ids_to_tokens(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return [tokenizer.convert_tokens_to_string([token]) for token in tokens]

def mask_logits(logits, indices, kept_ids):
    mask = torch.ones_like(logits) * (-1e9) # biv
    for i, ids in zip(indices, kept_ids): mask[:, i, ids] = 0
    return logits + mask
    
def show_predictions(text, examples, tokenizer, logits, bos_indices, eos_indices, answers, labels, 
        mask_logits_fn=None, topk=5, loss_reduction='mean', show_range=None, sep='\t', verbose=True):
    # use_openai_api = isinstance(model, types.FunctionType)
    use_openai_api = hasattr(logits, 'token_logprobs')
    if use_openai_api: ans_nlls = []
    if not use_openai_api and mask_logits_fn is not None: logits = mask_logits_fn(logits)
    
    bi = 0
    assert len(bos_indices) == len(examples), '%d != %d' % (len(bos_indices), len(examples))
    if show_range is None: show_range = range(len(examples))
    all_top1_correct = True
    for i, (example, bos_i, eos_i, ans_ids) in enumerate(zip(examples, bos_indices, eos_indices, answers)):
        # eos_i = bos_i + 2  # show only the first answer token
        if i not in show_range: continue
        # print(i)
        if use_openai_api:
            ans_prob_dist = [get_prob_dist(d, topk=topk) for d in logits.top_logprobs[bos_i + 1: eos_i]]
            ans_probs = [math.exp(lp) for lp in logits.token_logprobs[bos_i + 1: eos_i]]
            ans_nlls += [-lp for lp in logits.token_logprobs[bos_i + 1: eos_i]]
        else:
            ans_prob_dist = logits[bi, bos_i: eos_i - 1].softmax(-1)
            # print(ans_prob_dist.shape)
            ans_probs = ans_prob_dist[torch.arange(ans_prob_dist.size(0)), ans_ids]
            # print(ans_probs.shape)
        ans_tokens = convert_ids_to_tokens(ans_ids, tokenizer)
        for ans_id, ans_token, ans_prob, dist in zip(ans_ids, ans_tokens, numpy(ans_probs, decimals=3), ans_prob_dist):
            top1_correct = max(dist.items(), key=lambda x: x[1])[0] == ans_token.replace('Ġ', ' ') \
                if use_openai_api else (dist.argmax() == ans_id).item()
            all_top1_correct = all_top1_correct and top1_correct
            indices_fn = partial(convert_ids_to_tokens, tokenizer=tokenizer)
            if verbose: 
                print(('*' if top1_correct else ' ') + ans_token, ans_prob, dist if use_openai_api 
                    else show_topk(*dist.topk(topk), indices_fn=indices_fn), sep, example) 
            # if verbose: print(sep + example)
    # if verbose: print()
    if use_openai_api:
        loss = ans_nlls if loss_reduction == 'none' else sum(ans_nlls) / len(ans_nlls)
        # print(ans_nlls)
    else:
        loss_fct = nn.CrossEntropyLoss(reduction=loss_reduction)
        # logits = logits[..., :-1, :]
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        # loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1)) if loss_reduction \
        #     else nn.CrossEntropyLoss(reduction='none')(logits.view(-1, logits.size(-1)), labels.view(-1))[labels.view(-1)>=0].tolist()
    return loss, all_top1_correct

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
        self.elapsed = str(end - self.start).split('.')[0]
        # self.elapsed_secs = end - self.start
        # self.elapsed = self.elapsed_secs #* 1000   # millisecs
        if self.verbose: print('done', self.elapsed)
            # print('elapsed time: %d s' % self.elapsed)