from random import sample
from collections import OrderedDict
# from typing import Iterable
from collections.abc import Iterable
import numpy as np
import math
from functools import reduce
from itertools import chain, product, combinations, cycle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F 

import einops
from einops import rearrange

from common_utils import iterable, show_topk, topk_md


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
    mag = (x**2 + y**2).sqrt()
    if plot:
        if start_i is None: start_i = 0
        if end_i is None: end_i = len(w)
        start_i = 0
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x[start_i: end_i], y[start_i: end_i], '.', alpha=alpha); plt.show()
        start_i = 1
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x[start_i: end_i], y[start_i: end_i], '.', alpha=alpha); plt.show()
        start_i = 0
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x[start_i: end_i]/mag, y[start_i: end_i]/mag, '.', alpha=alpha); plt.show()
        start_i = 0
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x[start_i: end_i]/mag * mag.log(), y[start_i: end_i]/mag* mag.log(), '.', alpha=alpha); plt.show() # https://transformer-circuits.pub/2021/framework/index.html#copying-matrix
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
    wn2i = OrderedDict(zip('qkvo', range(4)))  # q:0, k:1, v:2, o:3
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

def get_positional_score(model, forward_fn, qlen=128, offset=-1, substract_null_attn=True):
    attentions = forward_fn(model, torch.randint(100, 25000, size=(1, qlen)).to(model.device)).attentions
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

def get_matrix(model, layer, head, qk=True, compute_eigv=True):
    blocks = model.transformer.h
    wu = model.lm_head.weight.data
    ln_f = model.transformer.ln_f
    wq, wk, wv, wo = get_head_weights(model, layer, head, transpose=True)
    e = blocks[layer].ln_1(model.es[1])
    if qk:
        A, B = e @ wq, e @ wk
        m = A @ B.T
    else:
        A, B = wu @ wo.T, e @ wv  #  wu, ln_f(e @ wv @ wo)  # slow for eig
        m = wu @ ln_f(B @ wo).T  # a little better than A @ B.T
    if compute_eigv: print(plot_eigv((B.T @ A).eig()[0], plot=False))
    return m

def sample_all_top_entries(tokenizer, m, b, transpose=False, n_samples=50):
    if transpose: m, b = m.T, b.T # transpose=False: q->k, o->i; transpose=True: k->q, i->o
    indices = b.max(1).values.nonzero()[:, 0].tolist()
    for i in sample(indices, n_samples):
        row_token = tokenizer.convert_ids_to_tokens(i)
        col_tokens = tokenizer.convert_ids_to_tokens(b[i].nonzero()[:, 0])
        values = m[i][b[i]].numpy().astype(int)
        print('  --', row_token, '--', sorted(zip(col_tokens, values), key=lambda x: x[1], reverse=True))

def lookup_top_entries(tokenizer, m, keyword, topk=20):
    if not keyword.startswith(' '): keyword = ' ' + keyword
    ids = tokenizer.encode(keyword)
    if len(ids) > 1:
        # print(tokenizer.tokenize(keyword))
        return None
    i = ids[0]
    indices_fn = tokenizer.convert_ids_to_tokens
    return show_topk(*m[i].round().int().topk(topk), indices_fn=indices_fn)

def interpret_circuit(model, tokenizer, task, l, h, qk):
    print(l, h, 'qk' if qk else 'io')
    m = get_matrix(model, l, h, qk=qk, compute_eigv=True)
    if not qk: m = m.T
    vocab_fn = task[0]
    hop = 1; rel = vocab_fn()[hop].relations[0]
    for x in rel.dom():
        if len(tokenizer.tokenize(' ' + x)) == 1:
            print(f'{x}->{rel.f(x)} {lookup_top_entries(tokenizer, m, x, topk=5)}')

def get_head2scores(node):
    head2scores = []
    while 'root' not in node.name:
        scores = node.parent.data.scores
        score, *the_other_scores = list(scores.values())  # should use OrderedDict for scores
        head2score = list(zip(node.data.layer, node.data.head, score.numpy().round(3)[node.data.layer, node.data.head]))
        head2score = [(l, h, s) for l, h, s in head2score if (l, h) != (1, 7) and
            all(s > _score[l, h] for _score in the_other_scores)]
        head2scores.append(head2score)
        node = node.parent
    return head2scores

def head_chain_to_str(head_chain): return ' '.join([f'{l}-{h}' for l, h in head_chain])

def analyze_head_chains(model, head2scores, chain_len=None, plot=True):
    if chain_len is None: chain_len = len(head2scores)
    print(head2scores)
    head_chain_results = []
    for head_chain_with_scores in product(*head2scores[:chain_len]):
        head_chain, scores = zip(*[((l, h), s) for l, h, s in head_chain_with_scores])
        if any(h0[0] >= h1[0] for h0, h1 in zip(head_chain, head_chain[1:])):
            continue  # layers should be increasing
        reduced_score = scores[-1] - scores[0]
        eigv_pos = plot_eigv(weightprod(model, list(head_chain), 'e vo vo qk e',
            weBTA=model.weBTAs[0]), plot=False)[0]
        head_chain_results.append((head_chain, eigv_pos, reduced_score))
    if plot:
        head_chains, ys, xs = zip(*head_chain_results)
        plt.scatter(xs, ys)
        for i, head_chain in enumerate(head_chains):
            plt.annotate(head_chain_to_str(head_chain), (xs[i], ys[i]))
        plt.show()
    return head_chain_results

def get_transformations(model, f, e=None, qk=None, out=False, compose=False): # f: (N, 4096) --> (N, 4096)
    blocks = model.transformer.h;ln_f = model.transformer.ln_f;wu = model.lm_head.weight.data
    if qk:
        A, B = e @ wq, e @ wk
        m = A @ B.T
        return m 
    if e is None: e = model.transformer.wte.weight.data
    e1 = e + mlp_forward(blocks[0], e)[0] #if e is not None else we + mlp_forward(b, es[-1])[0]
    if compose:
        e_out = e1 +0
        for _f in f:
            e_out = _f(e_out)
    else:
        e_out = f(e1)         # transform
    e_ln = ln_f(e_out)
    if out:return e_ln @ wu.T, e_out
    return e_ln @ wu.T
    
def get_head_qk(model, layer=0,head=0,weight=False):
    blocks = model.transformer.h;we = model.transformer.wte.weight.data; e1=we + mlp_forward(blocks[0], we)[0]
    wq, wk, wv, wo = get_head_weights(model, layer, head, transpose=True)
    e = blocks[layer].ln_1(e1)
    A, B = e @ wq, e @ wk
    m = A @ B.T
    if weight:
        return blocks[layer].ln_1, wq @ wk.T
    return m 

def mlp_ov_fn(model, layer=0, act_vec=None, norm=False):
    blocks = model.transformer.h
    def fn(e):
        e = blocks[layer].ln_1(e) 
        w2, b2 = blocks[layer].mlp.fc_out.weight.data, blocks[layer].mlp.fc_out.bias.data
        act_vec_normed = act_vec if not norm else act_vec / act_vec.norm(dim=1) * e.norm(dim=1).mean()
        ov = (w2 @ act_vec_normed.T).T + b2.unsqueeze(0)
        return e + ov
    return fn

def head_ov_fn(model, layer=0, head=0, weight=False):
    blocks = model.transformer.h
    wq, wk, wv, wo = get_head_weights(model, layer, head, transpose=True)
    def fn(e):
        e = blocks[layer].ln_1(e)
        return e @ wv @ wo      
    if weight: return blocks[layer].ln_1, wv @ wo
    return fn
