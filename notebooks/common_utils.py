from collections import defaultdict, OrderedDict, Counter
import types
import math
import numpy as np
from itertools import chain

import torch
import torch.nn as nn

from transformers import GPT2Tokenizer

cache_dir = '/nas/xd/.cache/torch/transformers/'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

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
    values, indices = tensor.flatten().topk(k, largest=largest)
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    rows, cols = np.unravel_index(indices.numpy(), tensor.shape)
    return rows, cols

def norm(tensor, p=2): return tensor.norm(p=p, dim=-1).mean().round().item()

def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

def locate_answers(input_ids, bos_token='Ġ->', eos_token='Ċ', nrows=None):
    assert input_ids.size(0) == 1  # bsz == 1
    bos_id = tokenizer._convert_token_to_id(bos_token)
    eos_id = tokenizer._convert_token_to_id(eos_token)
    bos_indices = (input_ids[0] == bos_id).nonzero().squeeze(1).tolist()
    if nrows is not None:
        assert nrows == len(bos_indices)
    else:
        nrows = len(bos_indices)
    eos_indices = (input_ids[0] == eos_id).nonzero()[-nrows:].squeeze(1).tolist()
    labels = torch.ones_like(input_ids) * (-100)
    answers = []
    for bos_i, eos_i in zip(bos_indices, eos_indices):
        ans_ids = input_ids[0, bos_i + 1: eos_i]
        labels[0, bos_i: eos_i - 1] = ans_ids
        answers.append(ans_ids)
    return bos_indices, eos_indices, answers, labels

def get_prob_dist(d, topk=5):
    return {k: round(math.exp(v), 3) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk]}

def show_predictions(text, examples, outputs, bos_indices, eos_indices, answers, labels, 
        use_openai_api=False, topk=5, return_reduced_loss=True, show_range=None, sep='\t'):
    # use_openai_api = isinstance(model, types.MethodType)  # openai.Completion.create
    # outputs = model(engine=engine, prompt=text, max_tokens=0, echo=True, logprobs=5).choices[0].logprobs
    if use_openai_api: ans_nlls = []
    else: logits = outputs.logits
    
    bi = 0
    assert len(bos_indices) == len(examples), '%d != %d' % (len(bos_indices), len(examples))
    if show_range is None: show_range = range(len(examples))
    all_top1_correct = True
    for i, (example, bos_i, eos_i, ans_ids) in enumerate(zip(examples, bos_indices, eos_indices, answers)):
        if i not in show_range: continue
        print(' ' + example, end=sep)
        if use_openai_api:
            ans_prob_dist = [get_prob_dist(d, topk=topk) for d in outputs.top_logprobs[bos_i + 1: eos_i]]
            ans_probs = [math.exp(lp) for lp in outputs.token_logprobs[bos_i + 1: eos_i]]
            ans_nlls += [-lp for lp in outputs.token_logprobs[bos_i + 1: eos_i]]
        else:
            ans_prob_dist = logits[bi, bos_i: eos_i - 1].softmax(-1)
            ans_probs = ans_prob_dist[torch.arange(ans_prob_dist.size(0)), ans_ids]
        ans_tokens = tokenizer.convert_ids_to_tokens(ans_ids)
        for ans_id, ans_token, ans_prob, dist in zip(ans_ids, ans_tokens, numpy(ans_probs, decimals=3), ans_prob_dist):
            top1_correct = max(dist.items(), key=lambda x: x[1])[0] == ans_token.replace('Ġ', ' ') \
                if use_openai_api else (dist.argmax() == ans_id).item()
            all_top1_correct = all_top1_correct and top1_correct
            print(('*' if top1_correct else ' ') + ans_token, ans_prob, 
                    dist if use_openai_api else show_topk(*dist.topk(topk), indices_fn=tokenizer.convert_ids_to_tokens)) 
    if use_openai_api:
        loss = ans_nlls if return_reduced_loss else sum(ans_nlls) / len(ans_nlls)
    else:
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1)) if return_reduced_loss \
            else nn.CrossEntropyLoss(reduction='none')(logits.view(-1, logits.size(-1)), labels.view(-1))[labels.view(-1)>=0].tolist()
    return loss, all_top1_correct

def my_isinstance(obj, type):  # to cope with autoreload
    # return isinstance(obj, type)  # fail on autorelaod
    return obj.__class__.__name__ == type.__name__ 

def equal(a, b):
    assert a.size() == b.size()
    return (a - b).abs().mean(), a.abs().mean(), b.abs().mean()