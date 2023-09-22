import sys
sys.path.insert(0, '/nas/xd/projects/transformers/src')
import os
os.environ['HF_HOME'] = '/raid3/xd/.cache/torch'  # deliberately set this wrong path to avoid migrating cache
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="8,7"

from types import MethodType
from tqdm import tqdm
from collections import defaultdict, OrderedDict, Counter
from datetime import datetime
from io import StringIO
from dataclasses import dataclass, fields, asdict
import itertools
from itertools import chain, product
import math
from functools import reduce, partial
from collections.abc import Iterable
from collections import namedtuple 
import traceback
import pickle, gzip

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="task name", type=str)
args = parser.parse_args()
print('args', args)

# from multiprocessing import Pool
# from torch.multiprocessing import Pool
# torch.multiprocessing.set_start_method('spawn', force=True)
from multiprocessing.dummy import Pool

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

# from transformers.data.data_collator import DataCollator, default_data_collator
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer#, pipeline
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, LlamaTokenizer
# from transformers import RobertaForMaskedLM, RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
# from transformers import T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration
# from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed, AdamW
torch.set_grad_enabled(False);

from common_utils import Timer
with Timer('common_utils'): from common_utils import *
with Timer('utils'): from utils import *
with Timer('child_utils'): from child_utils import *
from child_utils import _str, _cxt2str, _item2str, _s, _be
from child_frames import *
with Timer('tasks'): from tasks import *
with Timer('model_utils'): from model_utils import *
from logits_dynamics import *
with Timer('weight_analysis'): from weight_analysis import *

import circuitsvis as cv
import plotly
import plotly.express as px
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load_context(load_mode='gpt-j', checkpoint=0):
    models = {}
    cache_dir = '/nas/xd/.cache/torch/transformers/'  # for models besides t5-3b/11b
    # cache_dir = '/mnt/nvme1/xd/.cache/torch/transformers/'  # for gpt-neox-20b on elderberry
    proxies = {'http': '192.168.50.1:1081'}
    
    
    # curl -x http://192.168.50.1:1081 -L -O [-C -] https://huggingface.co/google/ul2/resolve/main/pytorch_model.bin  # -C for 断点续传
    s2s_model_names = ['google/t5-xl-lm-adapt', 'google/t5-xxl-lm-adapt', 'bigscience/T0p', 'bigscience/T0_3B',
        'allenai/tk-instruct-3b-pos', 'allenai/tk-instruct-3b-def-pos', 'google/ul2']
    gpt_model_names = ['EleutherAI/gpt-j-6B/cpu', 'EleutherAI/gpt-j-6B/int8', 'EleutherAI/gpt-j-6B',
                      ]#, 'EleutherAI/gpt-neox-20b/cpu', #'EleutherAI/gpt-neox-20b', 'gpt2-xl', 'gpt2']
    llama_model_names = ['models/vicuna/vicuna-7b@int8', 'models/vicuna/vicuna-13b@int8',
                         'lmsys/vicuna-13b-v1.3@cpu', 'lmsys/vicuna-13b-v1.3@int8',
                         'lmsys/vicuna-33b-v1.3@cpu', 'lmsys/vicuna-33b-v1.3@int8', 'lmsys/vicuna-33b-v1.3',
                         'decapoda-research/llama-7b-hf', 'decapoda-research/llama-13b-hf', 'decapoda-research/llama-30b-hf'
                        ]
    name2device = {'gpt-j-6B': 1, #'models/vicuna/vicuna-7b': 8, 'models/vicuna/vicuna-13b': 8,
                   'vicuna-7b-v1.3': 0, 'vicuna-13b-v1.3': 0, 'vicuna-33b-v1.3':1, 'llama-7b-hf': 1, 'llama-13b-hf': 1,'llama-30b-hf': 1}
    if load_mode == 'gpt-j':
        finetuned_models = {f'checkpoint-{checkpoint}': 0}
        for fmn, device in finetuned_models.items():
            print(fmn, device)
            if fmn == 'checkpoint-0':
                full_fmn = 'EleutherAI/gpt-j-6B'
            else:
                full_fmn = f'/raid/xd/data/models/train_tasks_28_with_normalsentence_1e5_3to1/{fmn}'
            model = AutoModelForCausalLM.from_pretrained(full_fmn, torch_dtype=torch.float16).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-j-6B', cache_dir=cache_dir)
            unify(model)
            models[fmn] = model, tokenizer
    elif load_mode in ['vicuna', 'llama']:
        for model_name in llama_model_names[-1:] + gpt_model_names[:0]:
        #for model_name in llama_model_names[:0] + gpt_model_names[:1]:
            if model_name in models: continue
            with Timer(model_name):
                model_cls = AutoModelForCausalLM #if any(s in model_name for s in ['gpt', 'fairseq-dense']) else T5ForConditionalGeneration
                _cache_dir = cache_dir# .replace('/nas/', '/nas2/') if 'gpt' not in model_name else cache_dir
                dst = model_name.split('@')[-1] if '@' in model_name else 'cuda'
                model_name = model_name.replace('/cpu', '').replace('/int8', '')
                _model_name = model_name.split('/')[-1]
                kwargs = dict(cache_dir=_cache_dir, proxies=proxies, low_cpu_mem_usage=True)
                if dst == 'cpu':
                    model = model_cls.from_pretrained(model_name, **kwargs)
                else:  # fp16 or int8 on GPU
                    device = name2device[_model_name]
                    device_map = get_device_map(devices=device, **name2mapping[_model_name]) if isinstance(device, Iterable) else None
                    dtype_kwargs = dict(load_in_8bit=True) if dst == 'int8' else dict(torch_dtype=torch.float16)
                    revision_kwargs = dict(revision='float16') if _model_name == 'gpt-j-6B' else {}
                    model = model_cls.from_pretrained(model_name, device_map=device_map, **dtype_kwargs, **revision_kwargs, **kwargs)
                    if device_map is None: model = model.to(device)
                if hasattr(model.config, 'use_cache'): model.config.use_cache = False  # save GPU mem
                # to avoid slow loading of AutoTokenizer->TokenizerFast
                tokenizer_cls = LlamaTokenizer if 'vicuna' in model_name or 'llama' in model_name else GPT2Tokenizer
                tokenizer = tokenizer_cls.from_pretrained(model_name, cache_dir=_cache_dir)
                unify(model)
                models[model_name] = model, tokenizer
        
        #model_name = gpt_model_names[0].replace('/cpu', '')  # engines[4]
        #model_name_gpu = model_name.replace('/cpu', '')
        #model_name = llama_model_names[-1]  # gpt_model_names/llama_model_names/engines
        #assert not model_name.endswith('/int8'), model_name
        #model, tokenizer = models[model_name]
        #model_name_gpu = model_name.replace('/cpu', '/int8') if model_name.endswith('/cpu') else model_name #+ '/int8'
        #model_gpu = models[model_name_gpu][0] if model_name_gpu in models else model  # for prediction rather than attribution
    
    blocks = model.transformer.h
    for i, b in enumerate(blocks): b.layer = i
    ln_f = model.transformer.ln_f
    L, H, embed_dim = len(blocks), blocks[0].attn.num_heads, blocks[0].attn.embed_dim
    
    we = model.transformer.wte.weight.data
    wu = model.lm_head.weight.data
    
    es = [we]
    for b in blocks[:1]: es.append(es[-1] + mlp_forward(b, es[-1])[0])
    return model, tokenizer

def run_task(model, tokenizer, checkpoint=0, key=None):
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    from child_utils import empty_cxt2str
    def filter_fn(p, c):
        if c.layer == 0 or c.head == c.H: return False
        pap, ap = abbreviate_attn_pattern(p.attn_pattern or ''), abbreviate_attn_pattern(c.attn_pattern)
        return c.top_score > 0.3 #and c.ap_score > 0.15 
    def filter_fn_negate_equal(p, c):
        if c.layer == 0 or c.head == c.H: return False
        pap, ap = abbreviate_attn_pattern(p.attn_pattern or ''), abbreviate_attn_pattern(c.attn_pattern)
        return (p.step == -1 and (ap in ['B->A0'] and c.ap_score > 0.15 and c.top_score > 0.4 and c.icl_score > 0.15 or (c.layer, c.head) in [(25,13)]) or
            p.step == 0 and (ap == 'B->A0' and (c.layer, c.head) in [(13,2),(13,11),(13,7), (16,7)]) or 
            p.step == 1 and (ap == 'B->Q' and (c.layer, c.head) in [(12,14), (10,11), (11,12), (11,9), (14,9),(14,12)]) or  # B->Q (9,5),
            p.step == 1 and (ap == 'B->A]^' and (c.layer, c.head) in [(12,10),(8,1), (13,13)]) #or  # B->A]^
        )

    def filter_fn_child(p, c):
        if c.layer == 0 or c.head == c.H: return False
        pap, ap = abbreviate_attn_pattern(p.attn_pattern or ''), abbreviate_attn_pattern(c.attn_pattern)
        return (p.step == -1 and (c.ap_score > 0.15 and c.top_score > 0.4 and c.icl_score > 0.15 or (c.layer, c.head) in [(25,13)]) or
            p.step == 0 and (ap == 'B->A0' and (c.layer, c.head) in [(13,2),(13,11),(13,7), (16,7)]) or 
            p.step == 1 and (ap == 'B->Q' and (c.layer, c.head) in [(12,14), (10,11), (11,12), (11,9), (14,9),(14,12)]) or  # B->Q (9,5),
            p.step == 1 and (ap == 'B->A]^' and (c.layer, c.head) in [(12,10),(8,1), (13,13)]) #or  # B->A]^
        )
    def filter_fn_equal(p, c):
        if c.layer == 0 or c.head == c.H: return False
        pap, ap = abbreviate_attn_pattern(p.attn_pattern or ''), abbreviate_attn_pattern(c.attn_pattern)
        return (p.step == -1 and (ap in ['B->A0'] and c.ap_score > 0.15 and c.top_score > 0.4 and c.icl_score > 0.15 or (c.layer, c.head) in [(24,10)]) or
            p.step == 0 and (ap == 'B->A0' and (c.layer, c.head) in [(13,2),(13,11),(13,7), (16,7)]) or 
            p.step == 1 and (ap == 'B->Q' and (c.layer, c.head) in [(12,14), (10,11), (11,12), (11,9), (14,9),(14,12)]) or  # B->Q (9,5),
            p.step in [1,2] and (ap == 'B->A]^' and (c.layer, c.head) in [(12,10),(8,1), (13,13),(9,14),(15,5)])
        )

    filter_fns = {'MlM_gen[genders_of_persons.TreeSet.neg_equal,types_of_things.TreeSet.equal][cxt_len=3]': filter_fn_negate_equal,
                  'MlM_gen[genders_of_persons.TreeSet.equal,types_of_things.TreeSet.child][cxt_len=3]': filter_fn_child,
                  'MlM_gen[genders_of_persons.TreeSet.equal,types_of_things.TreeSet.equal][cxt_len=3]': filter_fn_equal,
                 }
    if key not in filter_fns: return None, None
    tasks = [
        (lambda: [TreeSet(genders_of_persons).use(['equal', 'child', 'sibling']), TreeSet(types_of_things).use(['child', 'equal', 'sibling'])], MlM_gen,
         partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} has {a_(i[1])}.", f"{_be(the_(i[1]))} {i[0]}'s."]), lambda q, _: f"{the_(q)} has",
        ), # t: 21-5, 15-8, 19. p: 16-7, 18-5, [3-12, 13-7]. p+: 16-7, 16-0. 13-7:induction head qk, thing->type ov
        ]
    results = {};
    topk=10; nrows, k_shot = 12, 3; cxt_len = 3; batch_size = 12;
    save_results = False; verbose = False #not save_results or batch_size <= 8
    rel1_kwargs = {'x_f': None}  # {'x_f': _s, 'y_f': a_, 'skip_inv_f':False}
    for task,        rel0_i, rel1_i, do_swap_qa, do_negate, do_rm_query, rev_item2str, do_g2c in product(
        tasks[0:1], [0],[1,0],[False,],  [True,False],[False],[False],[False]):
    #     tasks[:1],range(3),range(3),[False,],[False,True],[False,True],[False,True],[False,True]):
        seed(42)
        args = dict(cxt_len=cxt_len, rev_item2str=rev_item2str, abstract=False)
        trans_args = dict(rel0_i=rel0_i, rel1_i=rel1_i, rel1_kwargs=rel1_kwargs, do_swap_qa=do_swap_qa, do_negate=do_negate,
                          do_rm_query=do_rm_query, do_g2c=do_g2c)
        task = transform_and_validate_task(task, **trans_args, **args)
        if task is None: continue
        res_key = f'{task2str(task)}[{args2str(args)}]'  # {composed_heads2str(model)}
        print(f'\n== {res_key} == {args2str(trans_args)}')
        if key is not None and res_key != key: continue
        r = results[res_key] if save_results and res_key in results else None
    #     if r is not None: print('duplicate task!'); continue 
        r = generate_and_predict_batch(model, tokenizer, task, nrows, k_shot, batch_size,
                logits_bias=None, custom_forward=True, result=r, verbose=verbose, **args)
        if save_results: results[res_key] = r
        if r.root is None: r.root = add_node(None, layer=L, label_type='labels')
        r.root = attribute_tree_on(r.data_tuples, model, r.root, 2, filter_fns.get(res_key, filter_fn), topk=topk, k_shot=k_shot, mix=True, device=None, verbose=False)
        return r, res_key
    
def single_model_vis(model, tokenizer, task_key, r, checkpoint=0):
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    task_dir = f'data/{task_key}'; os.makedirs(task_dir, exist_ok=True) 
    def save(fig_data, f_name, mode='fig'):
        fn = f'{task_dir}/{f_name}_model-{model.__class__.__name__}_ckpt{checkpoint}.html'
        if mode=='fig':
            fig_data.write_html(fn)
        elif mode=='data':
            torch.save(fig_data, fn.replace('.html', '.pt'))

    with Printer(f'{task_dir}/attribution_tree_model-{model.__class__.__name__}_ckpt{checkpoint}.txt'): print_tree(r.root)
    save([r.mean_loss, r.mean_acc, r.answer_probs], 'loss_acc', mode='data')
    fig, data = scan_global_attention_patterns(model, r.data_tuples)
    save(fig, 'global_attention_patterns', mode='fig')
    save(data, 'global_attention_patterns', mode='data')

    topk= model.transformer.wte.weight.data.shape[0]
    fig_heatmap, fig_rank = batch_logits_lens(r.data_tuples, model, tokenizer, tgt=None, metric='target', heatmap=True, case_sensitive=False, topk=topk)
    save(fig_heatmap, 'logits_heatmap', mode='fig')
    save(fig_rank, 'logits_rank_diff', mode='fig')

    attn_colormap= {'root': 'blue', 'B->A0':'green', 'B->Q':'orange', 'Q->A0':'purple'}
    ranges = r.data_tuples[0][3]
    fig = plot_attr_circuit(r.root, (L, H), ranges, depth=0, attn_colormap=attn_colormap, selectors=[], patterns=None)
    save(fig, 'attribution_circuit', mode='fig')
    return 

def cross_model_vis(task_key, checkpoints):
    # draw loss
    task_dir = f'data/{task_key}'
    #from glob import glob
    #losses = torch.stack([torch.tensor(torch.load(f'data/{task_key}/loss_acc_model-GPTJForCausalLM_ckpt{ckpt}.pt')) for ckpt in checkpoints])
    losses = []
    for ckpt in checkpoints:
        data = torch.load(f'data/{task_key}/loss_acc_model-GPTJForCausalLM_ckpt{ckpt}.pt')
        losses.append([data[0], data[1], data[2][3:].mean()])
    losses = torch.tensor(losses)
    fig = go.Figure()
    for name, data in zip(['loss', 'accuray', 'ans_prob'], losses.T):
        fig.add_trace(go.Scatter(x=checkpoints, y=data, name=name, mode="lines+markers"))
    fig.update_layout(xaxis_title="model checkpoint", yaxis_title="loss/acc")
    fig.write_html(f'data/{task_key}/aggregate_loss_acc_model-GPTJForCausalLM.html')
     
    



def main():
    #model = load_context(load_mode='vicuna')
    #model = load_context(load_mode='gpt-j', checkpoint=0)
    print('start processing')
    from logits_dynamics import get_cross_layer_similarity, show_pca_captured_var
    device = torch.device("cuda:0")
    #device = model.device
    if args.task == 'pca-sim':
        model, _ = load_context(load_mode='vicuna')
        with Timer('get_cross_layer_similarity'): get_cross_layer_similarity(model, save=True, full_layer='cat', device=device, rand_ratio=0.25, modes=['qk','ov','w1','w2','wg'])
        #with Timer('get_cross_layer_similarity'): get_cross_layer_similarity(model, save=True, full_layer=False, device=device, by_head=True, rand_ratio=0.1, modes=['ov'])
        #with Timer('get_cross_layer_similarity'): get_cross_layer_similarity(model, save=True, layer_pairs=[(6,4)], full_layer=False, device=device, by_head=True, rand_ratio=1, modes=['qk'])
    elif args.task == 'pca-auc':
        model, _ = load_context(load_mode='llama')
        L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
        with Timer('show_pca_captured_var window1'): show_pca_captured_var(model, save=True, device=device, window_size=1, rand_ratio=0.8, modes=['qk', 'ov', 'w1', 'w2', 'wg'], full_layer='cat', step=1, layers=list(range(L)),heads=list(range(1)))
        #with Timer('show_pca_captured_var window2'): show_pca_captured_var(model, save=True, device=device, window_size=2, rand_ratio=0.25, modes=['qk', 'ov', 'w1', 'w2', 'wg'], full_layer='cat', step=1, layers=list(range(L)),heads=list(range(1)))
        #with Timer('show_pca_captured_var window2'): show_pca_captured_var(model, save=True, device=device, window_size=2, rand_ratio=0.25, modes=['qk', 'ov', 'w1', 'w2', 'wg'], full_layer='cat', step=1, layers=list(range(L)),heads=list(range(1)))
    elif args.task == 'gpt-j-cross-ckpt':
        #for checkpoint in [0,100,300,500,700,900]:
        task_keys = ['MlM_gen[genders_of_persons.TreeSet.neg_equal,types_of_things.TreeSet.equal][cxt_len=3]',
                     'MlM_gen[genders_of_persons.TreeSet.equal,types_of_things.TreeSet.child][cxt_len=3]',
                     'MlM_gen[genders_of_persons.TreeSet.equal,types_of_things.TreeSet.equal][cxt_len=3]',]
        checkpoints = [0,100,200,300,400,500,700,900]
        for key in task_keys:
            cross_model_vis(key, checkpoints)
        #for checkpoint in tqdm(checkpoints, desc='cross models'):
        #    try:
        #        model, tokenizer = load_context(load_mode='gpt-j', checkpoint=checkpoint)
        #        for key in task_keys:
        #            r, task_key = run_task(model, tokenizer, checkpoint=checkpoint, key=key)
        #            if r is None:
        #                print('skip', key)
        #                continue
        #            single_model_vis(model, tokenizer, task_key, r, checkpoint=checkpoint)
        #    except Exception as e:
        #        print('model error!', str(e))
        #    finally:
        #        del model, tokenizer, r
    elif args.task == 'download_llama':
        model, _ = load_context(load_mode='llama')


if __name__ == '__main__':
    main()
