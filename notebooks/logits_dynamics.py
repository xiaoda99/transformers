from model_utils import *
from weight_analysis import get_positional_score

import circuitsvis as cv
import plotly.express as px
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from einops import rearrange, einsum
from tqdm import tqdm

def get_tokens_from_ids(iids, tokenizer, replace=True):
    return [t if not replace else t.replace('Ġ',' ').replace('Ċ', '\n').replace('<0x0A>', ' \\n') for t in tokenizer.convert_ids_to_tokens(iids)] 

def show_interactive_heads(outputs, input_ids, tokenizer, attr=False, layer=0, head=None, token_idx=None):
    attentions = outputs.attentions; str_tokens = get_tokens_from_ids(input_ids, tokenizer)  # attentions (b,n,i,j)
    B, heads_num, i, j = attentions[0].shape
    print(f'layer:{layer}, head:{head}, token:{str_tokens[token_idx] if token_idx is not None else None}')
    if attr:
        attn, node_keys = [], []
        for k, v in outputs.attn_attr.items():
            for k2, v2 in v.items():
                if k2 == (layer,head):
                    attn.append(v2*attentions[layer][0,head]); node_keys.append(k)
        if attn:
            attn = torch.stack(attn) 
            attn[attn<0] = 0 # ingore negative attr
            attn = attn / attn.amax(dim=(1,2), keepdim=True) # normalize
            attn = torch.cat([attn, attentions[layer][0,head:head+1]])
            print('\n'.join(f'head{i}-nodekey:{k}' for i,k in enumerate(node_keys)))
        else:
            attn = attentions[layer][0,head:head+1]
    elif layer == 'all': # show all layer head at token with idx, (layer='all', token_idx=-3)
        assert len(attentions) <= len(str_tokens)
        attn = torch.zeros(len(attentions), i, j)
        attn[:,:heads_num] = torch.cat([l[:1,:,token_idx,:] for l in attentions])
    elif head is None: # show all heads, (layer=0)
        attn = attentions[layer][0]
    else: # show a head, (layer=0, head=0)
        attn = attentions[layer][0,head:head+1]
    return cv.attention.attention_patterns(tokens=str_tokens, attention=attn) # attention [num_heads x dest_tokens x src_tokens]

def show_hidden_traj(data_tuples, tokenizer, hid_type='hidden_states', layer_slice=None, head_slice=None, layer_heads=None, selector=None, perplexity=30, metric='euclidean'):
    hs, h_labels, colors = [], [], []
    for data_tuple in data_tuples:
        o = data_tuple[-1]; L = len(o.attn_outputs); _slice=slice(0,L) if hid_type != 'hidden_states' else slice(1,L+1)
        if layer_slice is None: layer_slice = slice(0, L)
        str_tokens = get_tokens_from_ids(data_tuple[1][0], tokenizer)
        if hid_type in ['hidden_states', 'attn_outputs', 'mlp_outputs']:
            hs.append(torch.cat([hid[0] for hid in getattr(o, hid_type)[_slice][layer_slice]]))
            h_labels.extend([f'{t}_i{i:03d}_l{l:02d}_{hid_type}' for l in range(L)[layer_slice] for i,t in enumerate(str_tokens)])
            colors.extend([l for l in range(L)[layer_slice] for i,t in enumerate(str_tokens)])
        elif hid_type in ['head_outputs']:
            H = o.head_outputs[0].shape[1]
            if head_slice is None: head_slice = slice(0, H)
            if layer_heads is None: layer_heads = [(l, h) for l in range(L) for h in range(H)]
            hs.append(torch.cat([ head_hid for l, hid in list(enumerate(getattr(o, hid_type)))[layer_slice] for head, head_hid in list(enumerate(hid[0]))[head_slice] if (l,head) in layer_heads]))  # (head, token, dim) --> (head*token, dim)
            #hs.append(torch.cat([hid[0][head_slice].view(-1, hid[0].shape[-1]) for hid in getattr(o, hid_type)[layer_slice]] for h, head_hid in enumerate(hid[0])))  # (head, token, dim) --> (head*token, dim)
            h_labels.extend([f'{t}_i{i:03d}_l{l:02d}_h{head:02d}' for l in range(L)[layer_slice] for head in range(H)[head_slice] for i,t in enumerate(str_tokens) if (l,head) in layer_heads])
            colors.extend([l*H + head for l in range(L)[layer_slice] for head in range(H)[head_slice] for i,t in enumerate(str_tokens) if (l,head) in layer_heads])
    hs = torch.cat(hs)
    print(hs.shape, len(h_labels), hid_type, L, len(str_tokens))
    tsne = TSNE(n_components=2, verbose=1, random_state=0, metric=metric, perplexity=perplexity)
    zs = tsne.fit_transform(hs)
    def _contains(t, l): return any([i in t for i in l])
    fig = px.scatter(zs, x=0, y=1, hover_name=h_labels,color=colors, text=[s.split('_')[0] if _contains(s, selector) else '' for s in h_labels])
    return fig

def batch_logits_lens(data_tuples, model, tokenizer, task_name='', tgt=None, scan_layer=[], orders=[1,2], heatmap = True, case_sensitive=False, **kwargs):
    tuples = []; demo = ['', ''];
    for dt in data_tuples:
        text, input_ids, labels, ranges, *_, o = dt
        input_ids = input_ids[0]
        token_idxs = ranges[-1].bos; ans_idxs = ranges[-1].ans; query_idxs = ranges[-1].query
        #print('tgt idxs', [input_ids[ans_idxs[0]].tolist()]+ input_ids[query_idxs[0]:query_idxs[1]].tolist())
        if tgt is not None and len(data_tuples)==1:
            pass
        else:
            tgt = get_tokens_from_ids([input_ids[ans_idxs[0]].tolist()]+ input_ids[query_idxs[0]:query_idxs[1]].tolist(), tokenizer)
            if len(tgt)>2: tgt = [ tgt[0], ''.join(tgt[1:])]
            if not my_isinstance(tokenizer, LlamaTokenizer):
                tgt = [t.lower() if not case_sensitive else t for t in tgt]
                tgt = [' '+t if not t.startswith(' ') else t for t in tgt]
            if (token_idxs[1] - token_idxs[0])>1 or (query_idxs[1] - query_idxs[0])>1: demo = tgt #print('two tokens in query or target', tgt, dt[0])
        tuples.append((o, token_idxs[1]-1, tgt, input_ids))
    task_sample = data_tuples[0][0].split('\n')[-2]
    if not task_name: task_name = f'{demo[1]}->{demo[0]}' 
    if heatmap:
        logits_dict, fig_heatmap = logits_lens(tuples[0], model, tokenizer, task_name=task_name, scan_layer=scan_layer, heatmap=True, **kwargs)
    if len(tuples) >1 :
        logits_dict = mr(logits_lens)(tuples, model, tokenizer, task_name=task_name, scan_layer=scan_layer, heatmap=False, **kwargs)
    layer_idxs = np.arange(0,len(model.transformer.h)+1); head_idxs= np.arange(0, model.transformer.h[0].attn.num_heads)
    fig = px.scatter(x=[0], y=[0],labels={'x':'layer','y':'rank(logits_diff)'}, title=f'Logits difference and ranks for Model:{model.__class__.__name__}, Task:{task_name}')
    color_dict = {'layer_ranks': 'yellow', 'attn_ranks': 'red','mlp_ranks': 'green','layer_diff':'gold', 'attn_diff':'Crimson', 'mlp_diff':'ForestGreen', 'layer_diff2': 'Khaki','attn_diff2': 'Tomato', 'mlp_diff2': 'LimeGreen'}
    for k, color in color_dict.items():
        eidx, k_order = (-1, 2) if 'diff2' in k else (len(layer_idxs), 1) 
        #eidx =  if 'diff2' not in k else -1 
        #k_order = 2 if 'diff2' in k else 1 
        if k_order not in orders: continue
        fig.add_scatter(x=layer_idxs[:eidx], y=logits_dict[k][:,1], mode='markers+lines', name=k, marker=dict(color=color))
    if scan_layer:
        for k, color in [('head_diff2', 'Silver'), ('head_ranks', 'grey')]:
            fig.add_scatter(x=head_idxs, y=logits_dict[k][:,1], mode='markers+lines', name=k, marker=dict(color=color))
    fig.update_layout(annotations=[
        #go.layout.Annotation(text=f'Aggregation:{len(data_tuples)}, scan_layer:{scan_layer}',align='left',showarrow=False,xref='paper',yref='paper',x=0.0,y=1.10,font=dict(size=14)),
        go.layout.Annotation(text=f'Task sample:{task_sample}',align='left',showarrow=False,xref='paper',yref='paper',x=0.0,y=1.05,font=dict(size=14))])
    #fig.show() if not save_fig else fig.write_html(f'{save_dir}/batch_logits_lens_model-{model.__class__.__name__}_ckpt{checkpoint}_task-{task_name}.html')
    if heatmap:
        return fig_heatmap, fig
    return fig

def logits_lens(arg_tuples, model, tokenizer, residual_types=['all', 'attn', 'mlp'], task_name='', topk=10000, topk_out=10, scan_layer=[], verbose=False, heatmap=False, metric='target_diff'):
    outputs, token_idx, tgt, input_ids = arg_tuples
    print(f'tgt:{tgt}, metric:{metric}')
    def _show_topk_rank(raw_logits, tokenizer, topk=10000, tgt=' Mary', head_idx='all', l_idx=None, token_idx=-1, topk_out=5):
        logits, pred_ids = torch.topk(raw_logits, topk, dim=-1)
        tgts_idx = torch.tensor([tokenizer.encode(_tgt)[0] for _tgt in tgt]) if not my_isinstance(tokenizer, LlamaTokenizer) else torch.tensor([tokenizer.convert_tokens_to_ids(tgt)])[0]
        #print('tgt in rank', tgt, tgts_idx,  torch.tensor(tokenizer.encode(tgt)))
        tgts_idx = tgts_idx.to(pred_ids.device)
        tgt_idx = torch.where(pred_ids[:1,token_idx] == tgts_idx.unsqueeze(1))[1]
        token_logits = raw_logits[0,token_idx]
        logits_diff = logits[0, token_idx, tgt_idx[0]] - logits[0, token_idx, tgt_idx[1]]
        #print(l_idx, tgt_idx[0])
        rank = float(1/(tgt_idx[0]+1));
        pred_tokens = get_tokens_from_ids(pred_ids[0,token_idx][:topk_out], tokenizer, replace=False)
        lens_key = f'{l_idx}-{residual_type}-{head_idx}'
        if verbose: print(f'{l_idx}-{residual_type}-{head_idx}', f'rank:{rank:.4f}', f'logit_diff:{logits_diff:.1f}', f'logits{logits[0,token_idx,tgt_idx[0]]:.1f}', '_'.join([f'{t}-{logit:.1f}' for t, logit in zip(pred_tokens[:topk_out],logits[0,token_idx][:topk_out])]))
        core_logits[lens_key] = token_logits[tgts_idx[0]] - token_logits[tgts_idx] * 0  #TODO # logits, logits_diff (token_logits[tgts_idx], 
        ranks.append((lidx-1, residual_type, head_idx, rank))
        if heatmap:
            if metric == 'target_diff':
                confid = raw_logits[0,:,tgts_idx[0]] - raw_logits[0,:,tgts_idx[1]]
            elif metric == 'confidence':
                confid = logits[0,:,0] - logits[0,:,1]
            elif metric == 'target':
                confid = raw_logits[0,:,tgts_idx[0]]
            elif metric == 'target_prob':
                probs = torch.softmax(raw_logits,dim=-1)
                confid = probs[0,:,tgts_idx[0]]
            pred_tokens = [(get_tokens_from_ids(pred_id[:topk_out], tokenizer, replace=False), _logits[:topk_out]) for pred_id, _logits in zip(pred_ids[0], logits[0])]
            heatd.append(confid); heat_texts.append(pred_tokens); ylabels.append(lens_key)

    def _logits_lens(hid, model):
        hid = hid.to(model.device).half()
        return model.transformer.ln_f(hid) @ model.lm_head.weight.data.T
    o = outputs; core_logits, ranks, heatd, heat_texts, ylabels = {}, [], [], [],[]; L = len(model.transformer.h)
    _residual_types = ['all', 'attn', 'mlp']
    head_outputs = o.head_outputs if o.head_outputs is not None else [0] * len(o.attn_outputs)
    for lidx, (h, h_attn, h_mlp, h_heads) in enumerate(zip(o.hidden_states, list(o.attn_outputs)+[0], list(o.mlp_outputs)+[0], list(head_outputs)+[0])):
        for residual_type in _residual_types:
            if residual_type in ['attn', 'mlp']:
                logits = _logits_lens(h+h_attn if residual_type =='attn' else h+h_mlp, model)
            else:
                logits = _logits_lens(h, model)
            _show_topk_rank(logits, tokenizer, token_idx=token_idx, topk=topk, tgt=tgt, l_idx=lidx, topk_out=topk_out)
            if lidx not in scan_layer or residual_type != 'attn': continue
            for h_idx in range(h_heads.shape[1]):
                logits = _logits_lens(h+h_heads[:,h_idx], model) 
                _show_topk_rank(logits, tokenizer, token_idx=token_idx, head_idx=h_idx, topk=topk, tgt=tgt, l_idx=lidx, topk_out=topk_out)
    layer_diff2, attn_diff2, mlp_diff2, head_diff2, layer_diff, mlp_diff, attn_diff = [], [], [], [], [], [], []
    for l in range(len(o.attn_outputs)+1):
        layer_diff.append(core_logits[f'{l}-all-all'])
        mlp_diff.append(core_logits[f'{l}-mlp-all'])
        attn_diff.append(core_logits[f'{l}-attn-all'])
        if l>0: layer_diff2.append(core_logits[f'{l}-all-all'] - core_logits[f'{l-1}-all-all'])
        if l == len(o.attn_outputs): continue
        attn_diff2.append(core_logits[f'{l}-attn-all'] - core_logits[f'{l}-all-all']) # contribution of attn to logits diff in layer l
        mlp_diff2.append(core_logits[f'{l}-mlp-all'] - core_logits[f'{l}-all-all']) # contribution of mlp to logits diff in layer l
        if o.head_outputs is None: continue
        for h_idx in range(o.head_outputs[0].shape[1]):
            if f'{l}-attn-{h_idx}' not in core_logits: continue
            head_diff2.append(core_logits[f'{l}-attn-{h_idx}'] - core_logits[f'{l}-all-all']) # contribution of head l-hidx to logits diff in layer l
    layer_ranks = torch.tensor([(r[0], r[-1]) for r in ranks if r[1] == 'all'])
    mlp_ranks = torch.tensor([(r[0], r[-1]) for r in ranks if r[1] == 'mlp'])
    attn_ranks = torch.tensor([(r[0], r[-1]) for r in ranks if r[1] == 'attn' and r[2] == 'all'])
    head_ranks = torch.tensor([(r[2], r[-1]) for r in ranks if r[2] != 'all' and r[1] =='attn'])
    layer_diff2, attn_diff2, mlp_diff2, layer_diff, mlp_diff, attn_diff = map(torch.stack, [layer_diff2, attn_diff2, mlp_diff2, layer_diff, mlp_diff, attn_diff])
    head_diff2 = torch.stack(head_diff2) if head_diff2 else torch.tensor([0]) 
    logits_dict = dict(zip(['layer_diff2', 'attn_diff2', 'mlp_diff2', 'head_diff2', 'layer_ranks', 'head_ranks', 'mlp_ranks', 'attn_ranks', 'layer_diff', 'mlp_diff','attn_diff'],[layer_diff2, attn_diff2, mlp_diff2, head_diff2, layer_ranks, head_ranks, mlp_ranks, attn_ranks, layer_diff, mlp_diff, attn_diff]))
    for k, v in logits_dict.items():
        if isinstance(v, torch.Tensor):
            logits_dict[k] = v.detach().cpu().float()
    if verbose:
        for k, v in logits_dict.items(): print(k, v.shape if isinstance(v, torch.Tensor) else v)
    if heatmap:
        heatd = torch.stack(heatd).detach().cpu().float() # (L * 3, n) 
        font_size=9; 
        if abs((heatd[:2].max() - heatd[:2].max()) / heatd[:2].max()) < 0.1: heatd[:2] = 0  # filter first emb and attn layer 
        if len(residual_types)==1: residual_idx = _residual_types.index(residual_types[0]); heatd=heatd[residual_idx::3]; heat_texts=heat_texts[residual_idx::3];ylabels=ylabels[residual_idx::3]
        input_tokens = get_tokens_from_ids(input_ids, tokenizer, replace=False)
        text = [[col[0][0][:10] for j, col in enumerate(row)] for i,row in enumerate(heat_texts)] 
        hovertext = [['<br>'.join([f'{_idx:02d}:{_t[:10]}_{_logit:.1f}' for _idx, (_t, _logit) in enumerate(zip(col[0], col[1]))]) for j, col in enumerate(row)] for i,row in enumerate(heat_texts)] 
        fig = go.Figure(data=go.Heatmap(z=heatd, text=text, hovertext=hovertext, texttemplate="%{text}",textfont={"size":font_size}, xgap=0,ygap=0, hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>%{hovertext}<br>', hoverlabel=dict(font=dict(family='Arial', size=12, color='white')) ))
        fig.update_xaxes(title='Input tokens', ticktext=input_tokens,tickfont=dict(size=font_size), tickvals=np.arange(len(input_tokens)))
        fig.update_yaxes(title='Layer module', ticktext=ylabels,tickfont=dict(size=font_size), tickvals=np.arange(len(ylabels)))
        fig.update_layout(height= 11 * heatd.shape[0],width=50*len(input_tokens), title=f'Logits difference heatmap of {tgt[0]}-{tgt[1]} for model-{model.__class__.__name__}_task-{task_name}')
        #fig.show() if not save_fig else fig.write_html(f'{save_dir}/logits_diff_heatmap_model-{model.__class__.__name__}__ckpt{checkpoint}_task-{task_name}_sample-{tgt[0]}-{tgt[1]}.html')
    if heatmap:
        return logits_dict, fig
    return logits_dict 

def draw_attn_labels(data_tuple, tokenizer, ap='bos->example'):
    text, input_ids, labels, ranges, *_, o = data_tuple
    ts = tokenizer.convert_ids_to_tokens(input_ids[0])
    attn_size = (input_ids.shape[-1],input_ids.shape[-1])
    mask, _ = attn_pattern2labels(ranges, ap, attn_size, k_shot=2)
    fig = go.Figure(data=go.Heatmap(z=mask))
    fig.update_xaxes(title='Input tokens', ticktext=ts,tickfont=dict(size=5), tickvals=np.arange(len(ts)))
    fig.update_yaxes(title='Layer module', ticktext=ts,tickfont=dict(size=5), tickvals=np.arange(len(ts)))
    fig.update_layout(height=5*len(ts),width=5*len(ts))
    fig.show()
    return

def head_forward_fn(model, layer=9,head=14):
    #def _rotate_half(x):
    #    """Rotates half the hidden dims of the input."""
    #    x1 = x[..., : x.shape[-1] // 2]
    #    x2 = x[..., x.shape[-1] // 2 :]
    #    return torch.cat((-x2, x1), dim=-1)
    #def _apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    #    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    #    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    #    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    #    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    #    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    #    q_embed = (q * cos) + (_rotate_half(q) * sin)
    #    k_embed = (k * cos) + (_rotate_half(k) * sin)
    #    return q_embed, k_embed
    def fn(emb):
        blocks = model.transformer.h; attn = blocks[layer].attn
        wq, wk, wv, wo = get_head_weights(model, layer, head, transpose=True)
        emb = blocks[layer].ln_1(emb)
        query = emb @ wq; key = emb @ wk
        query = query.unsqueeze(1);key = key.unsqueeze(1)
        if isinstance(attn,GPTJAttention):
            rotary_dim = attn.rotary_dim
            k_rot = key[:, :, :, :rotary_dim] # Batch, head, tokens, emb
            k_pass = key[:, :, :, rotary_dim :]
            q_rot = query[:, :, :, :rotary_dim]
            q_pass = query[:, :, :, rotary_dim:]
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, rotary_emb=None)
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        elif isinstance(attn,LlamaAttention):
            #seq_length = query.size(-2)
            #position_ids = torch.arange(0, seq_length, dtype=torch.long, device=query.device)
            #position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            #cos, sin = attn.rotary_emb(query, seq_len=key.shape[-2])
            #query, key = _apply_rotary_pos_emb(query, key, cos, sin, position_ids)
            query, key = apply_rotary_pos_emb(query, key, rotary_emb=attn.rotary_emb)
        logits = query[0,0] @ key[0,0].T
        logits = logits / (query.size(-1) ** 0.5)
        logits[torch.ones_like(logits).bool().triu(diagonal=1)] = -1e4
        attention = torch.softmax(logits, dim=-1)#.tril()
        ov_out = emb @ wv @ wo
        out = attention @ ov_out
        return out, attention, logits, ov_out
    return fn

def construct_circuits(data_tuple, model, tokenizer, layers=None, heads=None, from_scratch=False, circuits=[], device=None, draw=True, attn_pattern='ans]->query]'):
    if device is None: device=model.device
    def _get_attention_score(data_tuple, attentions, attn_pattern='bos->query'):
        text, input_ids, labels, ranges, *_, o = data_tuple
        attn_labels, ranges_q = attn_pattern2labels(ranges, attn_pattern, attentions.size()[-2:], k_shot=2)
        return get_ap_scores(attentions, attn_labels.to(device), ranges_q).to('cpu', dtype=torch.float32).mean(-1)
    blocks = model.transformer.h; L, H, embed_dim = len(blocks), blocks[0].attn.num_heads, blocks[0].attn.embed_dim
    head_dim = embed_dim // H
    if layers is None: layers = range(L)
    if heads is None: heads = range(H+3)
    text, input_ids, labels, ranges, *_, o = data_tuple
    token_idx = ranges[-1].bos[1]-1
    #token_idx = ranges[-1].ans0[0]
    attentions = []
    tids = input_ids[0]; ts = get_tokens_from_ids(tids, tokenizer)
    print('bos token', ts[token_idx], input_ids.shape, ranges[-1].bos, text, ts)
    for hd_layer in layers:
        for head_idx in heads:
            if from_scratch:
                emb0 = model.transformer.wte(tids)
                emb0 = emb0.unsqueeze(0)
                emb1 = emb0 + mlp_forward(blocks[0], emb0) 
            else:
                if head_idx == H+1:
                    emb1 = o.hidden_states[hd_layer] + o.attn_outputs[hd_layer]
                elif head_idx == H+2:
                    emb1 = o.hidden_states[hd_layer] + o.mlp_outputs[hd_layer] + o.attn_outputs[hd_layer]
                elif head_idx == H:
                    emb1 = o.hidden_states[hd_layer] + o.mlp_outputs[hd_layer]
                else:
                    emb1 = o.hidden_states[hd_layer].to(device)
                    if head_idx is None:
                        head_out = 0
                    else: 
                        #head_out = o.head_outputs[hd_layer][:,head_idx] if head_idx is not None else 0
                        head_out, attention, _, _ = head_forward_fn(model, layer=hd_layer,head=head_idx)(emb1)
                    emb1 = emb1 + head_out 
            emb1 = emb1.to(device)
            #heads in first three layers: (0,7), (0,13), (1,3),(1,7),(1,10),(1,11),(2,8),(2,3),(2,11); after three layers:(3,5), (3,6),(4,6),(6,2),(7,6),(8,1),(9,14),(11,12)
            for _layer,_head in circuits:
                out, attention, logits, ov_out = head_forward_fn(model, layer=_layer,head=_head)(emb1)
                emb1 = emb1 + out
            #qk_logits, top_idx = torch.topk(logits[token_idx], logits.shape[1])
            #ov_logits = model.transformer.ln_f(out)[0] @ model.lm_head.weight.data.T
            #print('ov', model.transformer.ln_f(out).shape, ov_logits.shape)
            #ov_logits, ov_idxs = torch.topk(ov_logits[token_idx], 8)
            #ov_tokens = tokenizer.convert_ids_to_tokens(ov_idxs)
            ##attention = torch.softmax(logits  / head_dim ** 0.5, dim=-1).tril()
            #print(f'layer:{hd_layer},head:{head_idx}', 'token logits', logits[token_idx].max(), logits[token_idx].max()-logits[token_idx].min(),)# logits[token_idx].long())
            #print('qk----','_'.join([f'{_t}{_logit:.2f}' for _t,_logit in zip([ts[_i] for _i in top_idx[:8]], qk_logits[:8])]))
            #print('ov----','_'.join([f'{_t}{_logit:.2f}' for _t,_logit in zip(ov_tokens, ov_logits)]))
            # print('token attn', attention[token_idx].sum(), (attention[token_idx].round(decimals=3)*1000).long())
            attentions.append(attention)
    for i, attn in enumerate(attentions):
        ap_score = torch.tensor([_get_attention_score(dt, attn.unsqueeze(0).to(device),attn_pattern=attn_pattern) for dt in [data_tuple]]).mean() * 100
        print(f'{i}-attn_pattern:{attn_pattern}', ap_score)
    html = cv.attention.attention_patterns(tokens=ts, attention=torch.stack(attentions)) if draw else None
    return html, attentions

#base_nodes = ['A0]', 'A0+', 'Q]', 'B^', 'A]', 'S',  'S+', 'T', 'T+', 'A0', 'Q-', 'Q', 'Q+', 'B:s', 'B:t']
#label_nodes, nodes, xs, ys, link_labels, links = [], [], [], [], [], {}
## set base nodes
#for layer_idx in range(1):
#    nodes.extend(base_nodes)
#    xs.extend(list(range(1,len(base_nodes)+1)))
#    ys.extend([L]*len(base_nodes))
#    
#def scan_top_layer_nodes(parent, mode='top'):
#    ns = {}
#    for n in parent.children:
#        if n.data.dummy: continue
#        if isinstance(n.data.layer, int) and n.data.layer>57:continue
#        if n.data.head == n.data.H:
#            s, t = 'B', 'B'
#            ap = 'mlp'
#            continue
#        else:
#            ap = abbreviate_attn_pattern(n.data.attn_pattern)
#            s, t = ap.split('->')
#        l = n.data.layer if not isinstance(n.data.layer,list) else max(n.data.layer)
#        t = t.replace('^', '')
#        if mode == 'aggregate':
#            for _s in [s]:
#                if _s not in ns:
#                    ns[_s] = [l]
#                elif l not in ns[s]:
#                    ns[_s].append(l)
#        elif mode == 'top':
#            if ns.get(s, 0)<l:  ns[s] = l
##         if ns.get(t, L)>l:  ns[t] = l
#    return ns
#
#def add_nodes(ns, nodes, xs, ys):
#    for s, l in ns.items():
#        key = f'{s}:{l}'
#        if key not in nodes:
#            if s=='B': s='B:s'
#            _l = l if not isinstance(l,list) else max(l)
#            nodes.append(key); xs.append(base_nodes.index(s)+1); ys.append(L-_l)
#    
#def plot_node(attr_node, depth=0, depths=[0,1,2,3,4], root_label_key='loss', inter_nodes={}, mode='attn_agg'):
#    if depth not in depths: return
#    def _node2key(n):
#        return node2key(n).split(' > ')[-1].split(' ')[0] #+ '_' + abbreviate_attn_pattern(n.data.attn_pattern)
##     print('node', node2key(attr_node), len(attr_node.children))
#    for n in attr_node.children:
#        if n.data.dummy: continue            
##         if isinstance(n.data.layer, int) and n.data.layer>57: continue
#        if n.data.head == n.data.H:
#            s, t, ap = 'B', 'B', 'mlp'
#        else:
#            ap = abbreviate_attn_pattern(n.data.attn_pattern); s, t= ap.split('->')
#        t = t.replace('^', '')
#        if s in inter_nodes: s = f'{s}:{inter_nodes[s]}'
#        if t in inter_nodes: t = f'{t}:{inter_nodes[t]}'
#        if s == 'B': s= 'B:s'
#        if t == 'B': t= 'B:s'
#        si, ti, v = nodes.index(s), nodes.index(t), n.data.top_score
#        if s!=t and s.startswith('B:') and 'attn' in root_label_key: si = nodes.index(root_label_key)
#        v_str = f'{v:.2f}' if isinstance(v, float) else str(v)
##         link_label = f'{n.data.layer}-{n.data.head}_{ap}'#_{v_str}'
#        link_label = n.name
#        if v is not None and v <=0.3:
#            continue
##         print(node2key(n), ap, n.data.layer, s, t, si, ti, v)
#        if n.data.label_type is not None and 'attn' in n.data.label_type:
#            color = 'red' 
#        elif ap == 'mlp':
#            color = 'green'
#        elif s == t or (s=='B:t' and t == 'B:s'):
#            color = 'yellow'
#        else:
#            color = 'grey'
#        link_k = '_'.join(n.name.split(' ')[:4])
#        if link_k not in links: 
#            links[link_k] = (si, ti, v, link_label, color)
#        if n.children:
#            node_key = _node2key(n)
#            label_node_key = root_label_key
#            if n.data.label_type is not None and 'attn' in n.data.label_type: 
#                if mode == 'attn_agg':
#                    label_node_key = ap + '_attn'
#                    if label_node_key not in nodes:
#                        # create label node 
#                        label_nodes.append(label_node_key)
#                        nodes.append(label_node_key); xs.append(len(base_nodes) + len(label_nodes)); ys.append(L/2)
#                        nodes.append(f'B-{n.data.layer}');xs.append(len(base_nodes));ys.append(L-n.data.layer if isinstance(n.data.layer, int) else L-max(n.data.layer))
#                elif mode == 'attn_sep':
#                    label_node_key = f'{ap}_attn_{n.data.layer}-{n.data.head}-{n.data.label_type}'
#                    if label_node_key not in nodes:
#                        # create label node 
#                        label_nodes.append(label_node_key)
#                        _l = n.data.layer if isinstance(n.data.layer, int) else max(n.data.layer)
#                        nodes.append(label_node_key); xs.append(len(base_nodes) + len(label_nodes)); ys.append(L-_l)
##                         nodes.append(f'B-{n.data.layer}');xs.append(len(base_nodes));ys.append(L-_l)
#            # create inter-nodes 
#            inter_nodes = scan_top_layer_nodes(n)
#            add_nodes(inter_nodes, nodes, xs, ys)
##             print('label_node_key', label_node_key)
#            plot_node(n, depth=depth+1, depths=depths, root_label_key=label_node_key, inter_nodes=inter_nodes)
## #               links.append((si, nodes.index(_node2key(attr_node)), v, l_label,'grey'))
##                 links.append((nodes.index(_node2key(attr_node)), ti, v, l_label,'grey'))
#        
#plot_node(r.root, depths=[0,1,2])
#links = links.values()
## links = list(set(links)) # filter same links
#links = [l for l in links if l[0]!=l[1]]  # filter B->B  
#for i, n in enumerate(nodes): links.append((i,i,0.1,'null', 'grey')) # add base node links
## update node label
#labels = [f'{n}' for n,h,l in zip(nodes,xs,ys)]
#customdata = []
#for i, n in enumerate(nodes):
#    texts = [f'0root-{n}']
#    for link in links: 
#        if i == link[0]:
#            texts.append(f'{link[3]}')
#    texts.sort()
#    customdata.append('<br>'.join(texts))
## rearange nodes and positions    
#xs = np.array(xs)/(len(base_nodes)+len(label_nodes)) * 0.9
#ys = np.array(ys)/L - 0.1; ys[base_nodes.index('B:s')] = 0.1
#def rearange(xs, ys):
#    for y in set(ys.tolist()):
#        if y <=0.1 or y>=0.9:continue
#        idxs = ys == y
#        xs[idxs] = 0.1 + np.arange(0, len(xs[idxs])) / len(xs[idxs]) * 0.9
#    return xs, ys
## xs, ys = rearange(xs, ys)
## for i, (x, y, n) in enumerate(zip(xs, ys, nodes)): 
##     print(n, x, y)
## print('xs', xs.max(), xs)
## print('ys', ys.max(), ys)
#print('labels', labels)
## print('links', links)
## Create the Sankey diagram
#fig = go.Figure(data=[go.Sankey(
#        node=dict(label=labels,x=ys,y=xs,customdata=customdata,hovertemplate='%{customdata}<br>'),
#        link=dict(
#            arrowlen=15,
#            source=[l[0] for l in links],
#            target=[l[1] for l in links],
#            value=[l[2] for l in links],
#            label=[l[3] for l in links],
#            color=[l[4] for l in links],),
#        arrangement='freeform',
#        orientation = "v",)])
## Customize the layout
#fig.update_layout(title='Tree Structure (Sankey Diagram)',font=dict(size=12),height=1000,width=1000,)
#

def treeloop(node, ns=None, parents=False):
    if ns is None: ns = []
    for n in node.children:
        if parents:
            if n.children: 
                ns.append(n)
            #else:
            #    print('skip', n.name)
        else:
            ns.append(n)
        treeloop(n, ns=ns, parents=parents)
    return ns
    
def get_slice(r, name):
    try:
        if name == '*': b, e = 0, attn_size[0]
        elif name.startswith('['): b, _ = getattr(r, name[1:]); e = b + 1
        elif name.endswith(']'): _, e = getattr(r, name[:-1]); b = e - 1
        elif name.endswith('+'): _, e = getattr(r, name[:-1]); b, e = e, e + 1
        elif name.endswith('-'): b, _ = getattr(r, name[:-1]); b, e = b - 1, b
        else: b, e = getattr(r, name)
    except Exception as e:
#         print(f'In attn_pattern2labels: name = {name} not in {[k for k, v in r.__dict__.items() if v]}?')
        raise e
    return [slice(b, e)] if not isinstance(b, Iterable) else [slice(_b, _e) for _b, _e in zip(b, e)]

def get_x_symbols(root, ranges):
    aps = [n.data.attn_pattern for n in treeloop(root)]
    symbols = []
    for ap in aps:
        symbols.extend(ap.split('->'))
    symbols = list(set(symbols))
    print('symbols', symbols)
    symbols = [(s,slc.start) for s in symbols if not any([_s in s for _s in ['^', ']', 'unk']]) for slc in get_slice(ranges, s)]
    symbols = sorted(symbols, key=lambda x:x[1])
    symbols = [abbreviate_attn_pattern(s[0]) for s in symbols]
    symbols = [s+']' for s in symbols] + ['A]'] + symbols
    return symbols

def compute_lines(root, x_labels, grid_shape, attn_colormap=None, color_list=[], step=0, x_interval=10, cc_ap=None, all_cc_aps=[], lines=None, depth=0, selector=[], root_bias=None):
    sub_interval = 2
    L, H = grid_shape
    def cal_idx(s, bias=0):
        return x_labels.index(s) * x_interval + bias
    def in_mixed_node(node):
        if isinstance(node.data.layer, list):
            #print('skip vertical line for node', node2key(node))
            return True 
        #print('debug', node2key(node), 'children', node.children)
        return any([(node.data.layer, node.data.head) in list(zip(n.data.layer, n.data.head)) for n in root.children if n.children and isinstance(n.data.layer, list)])
    #def _has_attn(n):
    #    return any([ f'{n.data.layer}-{n.data.head} {abbreviate_attn_pattern(n.data.attn_pattern)} attn' in _n.name for _n in root.children])
    if lines is None: lines = {}
    #if depth >=4: return
    #print('raw_root_ap', f'step{step}', root.data.attn_pattern, root.data.label_type, node2key(root))
    root_name = 'root' if depth==0 else node2key(root)
    raw_root_ap = 'B->B' if depth==0 else abbreviate_attn_pattern(root.data.attn_pattern)
    root_s, root_t = raw_root_ap.split('->')
    #root_s, root_t = ("B", "B") if depth==0 else raw_root_ap.split('->')
    root_t = root_t.replace('^', ']').replace(']]',']')
    if root_s.endswith(']') and root_t.endswith('+'): root_t += ']'
    #print('root layer head', root.data.layer, root.data.head, node2key(root))
    root_layers = [root.data.layer] if not isinstance(root.data.layer, list) else root.data.layer
    root_heads = [root.data.head] if not isinstance(root.data.head, list) else root.data.head
    if cc_ap is None:  # parent circuits attn pattern
        #if step ==0 and ((raw_root_ap is None) or 'attn' not in raw_root_ap): # raw root or 59-4 B->B
        if step ==0 and 'attn' not in str(root.data.label_type): # raw root or 59-4 B->B
            cc_ap ='root' 
        else:
            cc_ap = raw_root_ap 
    #if step == 0 and abbreviate_attn_pattern(root.data.attn_pattern) : cc_ap
    if cc_ap not in attn_colormap:
        print(f'{cc_ap} not in attn_colormap')
        attn_colormap[cc_ap] = color_list[0]
        color_list.pop(0)
    cc_color = attn_colormap[cc_ap] 
    if root_bias is None: 
        root_xbias, root_ybias = 0, 0 #depth * sub_interval, depth * sub_interval
    else:
        root_xbias, root_ybias = root_bias
    for n in root.children:
        if n.data.dummy: continue  
        #if n.data.top_score is not None and n.data.top_score < 0.5: continue
        if n.data.head == n.data.H:
            s, t, ap = root_s, root_s, 'mlp'
        else:
            ap = abbreviate_attn_pattern(n.data.attn_pattern); s, t= ap.split('->')
        t = t.replace('^', ']').replace(']]',']')
        #if s.endswith(']') and t.endswith('+'): t += ']'
        if s.endswith(']') and not t.endswith(']'): t += ']'
        new_cc_ap = None 
        ls, hs = (n.data.layer , n.data.head) if isinstance(n.data.layer, list) else ([n.data.layer], [n.data.head]) 
        for mixed_idx, (l, h) in enumerate(zip(ls, hs)):
            l_name = node2key(n) 
            l_name = l_name.replace('attn:B->~<s>', 'attn').replace('attn/example', 'attn').replace('attn/ans0s','attn')# merge attn, example and ~<s>
#             print('lname', l_name)
            if not n.children and n.data.label_type is not None and 'attn' in n.data.label_type:
                l_name += '_attr0'
            l_name += f'_depth{depth}_{l}-{h}_'+data2str(n.data)
            if l_name in lines: 
                print('skip', node2key(n))
                continue
            if n.children and n.data.label_type is not None and 'attn' in n.data.label_type:
                color = 'red'
                xbias, ybias = root_xbias+sub_interval, root_ybias+sub_interval
                new_cc_ap = ap
            else:
                color = cc_color 
                xbias, ybias = (root_xbias, root_ybias)
                #if ap == 'mlp': xbias += 1 #= x_interval-1
            x = [cal_idx(s, xbias), cal_idx(t, xbias)]
            #if x[0] < x[1]:
            #    print(s, t, ap, ls, hs)
            y = [l*H+h-ybias] * len(x)
            vertical = False
            width = n.data.top_score if n.data.top_score is not None else 1
            #if n.data.top_score is None: print('top score none', node2key(n))
            lines[l_name] = [x,y, color, depth, vertical, width, cc_ap]
            # add vertical lines for nodes without children
            if not n.children and not in_mixed_node(n) and ap != 'mlp':
                x = [cal_idx(t, xbias), cal_idx(t, xbias)]
                y = [l*H+h-ybias, 0] 
                lines[l_name+'-to-bottom'] = [x,y, color, depth, True, width, cc_ap]
            # add vertical lines
            for root_layer, root_head in zip(root_layers, root_heads):
                if n.children and n.data.label_type is not None and 'attn' in n.data.label_type:
                    continue
                if ap == 'mlp':continue
                if depth==0:
                    x = [cal_idx(root_s), cal_idx(root_t)]
                    y = [L*H-ybias, l*H+h-ybias]
                else:
                    if root.data.label_type is not None and 'attn' in root.data.label_type: # B->A0 attn
                        if n.children:  #continue 
                            mixed = len(ls)>1
                            if not mixed or mixed_idx != len(ls)-1: continue # do not draw vertical line for single attn node who has children
                            if n.data.label_type is None or 'attn' not in n.data.label_type: continue
                            color = 'red' # add mixed vertical line eg. A0 -> A0
                            x = [cal_idx(t, xbias), cal_idx(t, xbias)]
                            y = [ls[0]*H+hs[0]-ybias, ls[-1]*H+hs[-1]-ybias]
                        else:
                            x = [cal_idx(root_s, root_xbias), cal_idx(s,root_xbias)] # B
                            color = cc_color
                            y = [root_layer*H + root_head - ybias, l*H+h-ybias]
                    else: # B->Q or B->A]^ or B->B
                        x = [cal_idx(root_t, xbias), cal_idx(root_t, xbias)] # Q or A]^
                        color = cc_color 
                        y = [root_layer*H + root_head - ybias, l*H+h-ybias]
                    #if color == 'red':
                    #    print(n.data.label_type, x, y, ap, l_name, root_xbias, root_ybias, root_layer, root_head)
                lines[f'{root_name}>>{l_name}-{root_layer}-{root_head}'] = [x, y, color, depth, True, width, cc_ap]
        if n.children:
            #print('has children', n.name)
            if selector and node2key(n) not in selector: continue
            _cc_ap = new_cc_ap or cc_ap
            #if new_cc_ap is not None and _cc_ap in all_cc_aps: 
            #    print('skip because of recurrent cc_ap', _cc_ap, node2key(n))
                #continue
            if _cc_ap not in all_cc_aps: all_cc_aps.append(_cc_ap)
            compute_lines(n, x_labels, grid_shape, step=step+1, attn_colormap=attn_colormap, cc_ap=_cc_ap, color_list=color_list, x_interval=x_interval, lines=lines, depth=depth+1, selector=selector, root_bias=(xbias, ybias))   
    return lines

def scan_attn_patterns(node, attn_nodes=None, patterns=None, ap='B->A0'):
    if patterns is None: return []
    if attn_nodes is None: attn_nodes = []
    for n in node.children:
        if node2key(n).startswith(patterns) and (ap is None or ap == abbreviate_attn_pattern(n.data.attn_pattern)):
            attn_nodes.append(node2key(n))
        if n.children:
            scan_attn_patterns(n, attn_nodes, patterns=patterns, ap=ap)
    return attn_nodes

def plot_attr_heatmap(root):
    ns = treeloop(root, parents=True)
    ns = [root] + ns
    for n in ns:
        print(node2key(n))
        attr = n.data.attr.head/n.data.attr.head.max()
        fig = px.imshow(attr)
        fig.update_layout(title=node2key(n))
        fig.show()

# plot circuits
def plot_attr_circuit(root, grid_shape, ranges, depth=0, attn_colormap=None,patterns=None, selectors=[], selector_ap=None, x_interval=10): # patterns=['28-8 attn'], ap='B->A0'
#     x_labels = ['A0]', 'A0+', 'Q]', 'B^', 'A]', 'S',  'S+', 'T', 'T+', 'A0', 'Q-', 'Q', 'Q+', 'B']
    depth_selectors = ['all', 'depth0', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5', 'depth6']
    selector = scan_attn_patterns(root, patterns=patterns, ap=selector_ap)
    print('selector', selector)
    #text, input_ids, labels, ranges, *_, o = r.data_tuples[0]
    x_labels = get_x_symbols(root, ranges[0])
    L, H = grid_shape
    #L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    if attn_colormap is None: attn_colormap={'root': 'blue', 'B->A0':'green', 'B->Q':'orange', 'Q->A0':'purple'}
    lines ={};color_list = ['grey', 'yellow', 'goldenrod', 'black', 'firebrick', 'firebrick', 'brown', 'lightgray', 'olive', 'silver']
    lines = compute_lines(root, x_labels, (L,H), step=0, color_list=color_list, attn_colormap=attn_colormap, lines=lines, depth=depth, x_interval=x_interval, selector=selector)
    y_axis_range = [-H, (L+2) * H]; y_tickvals = list(range(0, (L+1)*H, H)); y_ticktext = [f'{int(val/H):02d}L' for val in y_tickvals] 
    fig = go.Figure()
#     for i, label in enumerate(x_labels):
#         fig.add_trace(go.Scatter(x=[i*x_interval], y=[L*H], mode="markers+text", text=[label], textposition="top center"))
    attn_traces = dict((k,[]) for k in selectors)
    depth_traces = dict((k,[]) for k in depth_selectors)
    circuits_traces = dict((k,[]) for k in attn_colormap)
    points = {}
    for idx, (name, line) in enumerate(lines.items()):      
        x, y, color, depth, vertical, width, cc_ap = line
        for k in selectors:
            attn_traces[k].append(name.startswith(k))
        for k in depth_selectors:
            selected = True if k == 'all' or f'depth{depth}' <= k else False
            depth_traces[k].append(selected)
        for k in circuits_traces:
            circuits_traces[k].append(k==cc_ap)
        if len(x) == 2 and x[0] == x[1] and y[0]==y[1]:
            color = "black" if '-m' in name else 'red'
            mode = "markers+text"; tcolor = 'black'; marker_size = 8;symbol = 'square'
        else:
            tcolor = 'grey' if x[0] == x[1] else 'black'
            mode = 'lines+markers' if vertical else "lines+markers+text"
            marker_size = 4; symbol = 'circle'
        text = '' if (color == 'red' and x[0]!=x[1]) or vertical else name.split('_')[-2] 
        #text = '' if vertical else name.split('_')[-2] 
        #hover_text = name#.split('_')[-2:]
        hover_text = name.split(' > ')[-1]
        opacity = width 
        for _x, _y in zip(x, y):
            points[(_x, _y)] = points.get((_x, _y), '') + '<br>'+hover_text
        customdata = ['<br>'.join(sorted(list(set(points[(_x, _y)].split('<br>'))))) for _x, _y in zip(x, y)] # filter repetive node text
        #customdata = ["", hover_text]
        fig.add_trace(go.Scatter(x=x, y=y, opacity=opacity, mode="lines+markers+text", customdata=customdata, text=['',text], name=name.split(' > ')[-1][:10], textposition="top right",  line=dict(color=color, dash='solid', width=2),textfont=dict(size=12,color=tcolor), marker=dict(size=marker_size, symbol=symbol)))
#     fig.update_traces(visible=False, selector=dict(name='depth1'))  
    fig.update_traces(hovertemplate="<br>".join(["%{customdata}","X: %{x}","Y: %{y}"]))
    fig.update_layout(
        title="LLM Attribution Circuit Diagram",
        showlegend=False,height=1000,width=1000,
        xaxis=dict(range=[-1, len(x_labels)* x_interval + x_interval],zeroline=True,showgrid=True,dtick=x_interval,tickvals=list(range(0, len(x_labels)*x_interval, x_interval)),ticktext=x_labels),
        yaxis=dict(range=y_axis_range, zeroline=True, showgrid=True,dtick=H,tickvals=y_tickvals, ticktext=y_ticktext),
        updatemenus=[{'buttons': [{'args': [{'visible': attn_traces[sk]}],'label': sk,'method': 'update'}  for sk in selectors]+[{'args': [{'visible': circuits_traces[sk]}],'label': sk,'method': 'update'}  for sk in circuits_traces],
                    'type':'buttons', 'direction': 'right','showactive': True,'x': 0.0,'xanchor': 'left','y': -0.07,'yanchor': 'top'},
                     {'buttons': [{
                    'args': [{'visible': depth_traces[sk]}],
                    'label': sk,'method': 'update'}  for sk in depth_selectors],
                    'type':'buttons', 'direction': 'left','showactive': True,'x': 0.0,'xanchor': 'left','y': -0.12,'yanchor': 'top'}
                    ])
    return fig

def scan_global_attention_patterns(model, data_tuples, attn_patterns=None):
    def _get_attention_score(data_tuple, layer, head, attn_pattern='bos->query'):
        text, input_ids, labels, ranges, *_, o = data_tuple
        attentions = o.attentions[layer][0, head]
        attn_labels, ranges_q = attn_pattern2labels(ranges, attn_pattern, attentions.size()[-2:], k_shot=3)
        return get_ap_scores(attentions, attn_labels, ranges_q).to('cpu', dtype=torch.float32).mean(-1)
    if attn_patterns is None: attn_patterns=all_attn_patterns
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    scores = torch.zeros((len(attn_patterns), L,H)) - 0.1
    for i, ap in enumerate(attn_patterns):
        print(ap)
        try:
            for l in range (L):
                for h in range(H):
                    scores[i,l,h] = torch.tensor([_get_attention_score(dt, l, h,attn_pattern=ap) for dt in data_tuples]).mean() * 100
        except Exception as e:
            print('error', ap)
    fig = px.imshow(scores, facet_col_wrap=min(10,len(attn_patterns)), facet_col=0,zmax=100)
    for i, annot in enumerate(fig.layout.annotations):
        fig.layout.annotations[i].update(text=attn_patterns[int(annot['text'].split('=')[-1])])
    fig.update_layout(height=1000,width=1600)
    return fig, [scores, attn_patterns]

def get_cum_var_ratio(samples, mean=True, compute_uv=False, device='cpu', method='svd'): # method: svd, eigvals
    samples = samples.to(device)
    if mean:
        samples = samples - samples.mean(dim=0, keepdim=True)
    if compute_uv:
        #u, s, v = torch.svd(samples)
        u, s, v = torch.linalg.svd(samples, driver='gesvdj',full_matrices=True)
        explained_variance = (s**2) / (samples.shape[0] - 1)
    else:
        #u, s, v = torch.linalg.svd(samples, driver='gesvdj',full_matrices=False)
        #u, s, v = torch.svd_lowrank(samples, q=samples.shape[-1])
        if method == 'svd':
            s = torch.linalg.svdvals(samples) # driver='gesvdj'
            explained_variance = (s**2) / (samples.shape[0] - 1)
        else:
            cor_matrix = samples.T @ samples / (samples.shape[0] - 1)
            eigvals = torch.linalg.eigvals(cor_matrix)
            eigvals = (eigvals.real ** 2 + eigvals.imag **2) ** 0.5
            eigvals, _ = torch.topk(eigvals, eigvals.shape[0])
            explained_variance = eigvals 
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    size = (explained_variance_ratio.shape[0],explained_variance_ratio.shape[0])
    cumulated_var_ratio = torch.broadcast_to(explained_variance_ratio.unsqueeze(0), size).tril().sum(dim=1)
    if compute_uv: return cumulated_var_ratio.cpu(), u, s, v
    return cumulated_var_ratio.cpu()

def get_weights(model, l, selector=None, full_layer='cat', window_size=1, device='cpu'): # full_layer: cat, add, False
    arrs = []; L = len(model.transformer.h)
    for l in range(l,l+window_size):
        if l >= L: continue
        wq, wk, wv, wo = get_head_weights(model, l)
        wq, wk, wv, wo = [x.to(device) for x in[wq, wk, wv, wo]]
        if full_layer == 'add':
            (qkv_pattern, o_pattern) = ('n e d -> 1 (n d) e', 'n d e -> 1 (n d) e')
            wq, wk, wv = [rearrange(w, qkv_pattern) for w in [wq, wk, wv]]
            wo = rearrange(wo, o_pattern)
        else:
            wq, wk, wv = [w.transpose(1,2) for w in [wq, wk, wv]]
        if selector is not None: wq, wk, wv, wo = [ w[selector.to(device)] for w in [wq, wk, wv, wo]]
        qk = einsum(wq, wk, 'n d eq, n d ek -> n eq ek')
        ov = einsum(wv, wo, 'n d ev, n d eo -> n ev eo')
        if my_isinstance(model, LlamaForCausalLM):
            mlp_block = model.model.layers[l].mlp
            w1, w2, wg = mlp_block.up_proj.weight.data, mlp_block.down_proj.weight.data.T, mlp_block.gate_proj.weight.data
        else:
            mlp_block = model.transformer.h[l].mlp
            w1, w2, wg = mlp_block.fc_in.weight.data, mlp_block.fc_out.weight.data.T, mlp_block.fc_out.weight.data  # fake wg
        w1, w2, wg = w1.unsqueeze(0), w2.unsqueeze(0), wg.unsqueeze(0)
        wq, wk, wv, wo = None, None, None, None 
        arrs.append((wq, wk, wv, wo, qk, ov, w1, w2, wg))
    if len(arrs)>1:
        wq, wk, wv, wo, qk, ov, w1, w2, wg = [torch.cat([arr[i] for arr in arrs], dim=1) if arrs[0][i] is not None else None for i in range(len(arrs[0]))]
    if full_layer == 'cat':
        qk = rearrange(qk, 'n eh e -> 1 (n eh) e')
        ov = rearrange(ov, 'n eh e -> 1 (n eh) e')
    if not my_isinstance(model, LlamaForCausalLM): wg = None
    return wq, wk, wv, wo, qk, ov, w1, w2, wg

def show_pca_captured_var(model, save=False, heads_filter=None, window_size=1, step=1, layers=[], heads=[], modes=['q'], full_layer='cat',rand_ratio=1, device=None): # full_layer: add/cat/False
    H = model.transformer.h[0].attn.num_heads
    if device is None: device = model.device
    if heads_filter is not None:
        assert window_size == 1 and full_layer == 'cat'
        heads_score = get_positional_score(model, forward0)
        heads_selected = [torch.where(s >= 0.1)[0] for s in heads_score] if heads_filter == 'short' else [torch.where(s < 0.1)[0] for s in heads_score]
    fig = go.Figure()
    features = {}
    mean_ratios = dict((k,{}) for k in modes)
    layers = layers[::step] + layers[-1:] if step > 1 else layers
    for l in tqdm(layers):
        if l + window_size > layers[-1]: continue
        if full_layer in ['cat', 'add']: assert len(heads) == 1 and heads[0] == 0
        selector = None if heads_filter is None else heads_selected[l]
        wq, wk, wv, wo, qk, ov, w1, w2, wg = get_weights(model, l, selector=selector, full_layer=full_layer, window_size=window_size, device=device)
        features = dict([('q', wq), ('k', wk),('v', wv), ('o', wo), ('qk', qk), ('ov', ov), ('w1', w1), ('w2', w2),('wg', wg)])
        for h in heads:
            for f in modes:
                #if features[f] is None: continue # wg in GPT-J
                samples = features[f][h].float()
               # print(l, h, f, samples.shape)
                assert samples.shape[1] == model.transformer.h[0].attn.embed_dim
                if rand_ratio != 1 and f in ['qk', 'ov']: samples = samples[torch.randperm(samples.shape[0])[:int(rand_ratio*samples.shape[0])]]
#              with Timer('get_cum_var_ratio'):
                cumulated_var_ratio = get_cum_var_ratio(samples, device=device)
                #heads_filter_weight = 1 if heads_filter is None else heads_selected[l].shape[0] / H
                #cumulated_var_ratio *= heads_filter_weight   
#                 print(l, h, cumulated_var_ratio.shape)
                name = f'{l}-{h}-{f}'
                mean_ratios[f][l] = mean_ratios[f].get(l, 0) + cumulated_var_ratio
                if f not in ['qk', 'ov', 'w1', 'w2', 'wg']: continue
                if l % 5 == 0:
                    fig.add_trace(go.Scatter(x=np.arange(cumulated_var_ratio.shape[0]), y=cumulated_var_ratio, opacity=0.3, name=name, line=dict(color='grey'), mode="lines+text"))
    for f in modes:
        if l ==  layers[-1]:
            name_mean = f'{f}-mean'
            length = list(mean_ratios[f].values())[0].shape[0]
            fig.add_trace(go.Scatter(x=np.arange(length), y=torch.stack([r for r in mean_ratios[f].values()]).mean(dim=0), opacity=1, text=[name_mean if _idx == int(length/10) else '' for _idx in range(length)], name=name_mean, mode="lines+text"))
    fig.update_xaxes(title='PCA components'); fig.update_yaxes(title='Captured variance')
    fig.update_layout(title=f'Model:{model.__class__.__name__}, mode: {modes}, window_size:{window_size}, full_layer:{full_layer}, rand_ratio:{rand_ratio}, heads:{heads_filter}')
    fig2 = go.Figure()
    for k, _ in mean_ratios.items():
        fig2.add_trace(go.Scatter(x=[l for l, v in mean_ratios[k].items()], y=[ v.mean() for l, v in mean_ratios[k].items()], name=k, mode="lines+text"))
    fig2.update_layout(title=f'Model:{model.__class__.__name__}, mode: {modes}, window_size:{window_size}, full_layer:{full_layer}, rand_ratio:{rand_ratio}, heads:{heads_filter}')
    fig2.update_xaxes(title='layer index'); fig2.update_yaxes(title='area under the curve')
    if save:
        fig.write_html(f'visualization/pca/pca_window{window_size}_{full_layer}_captured_var_heads_filter-{heads_filter}_Model-cpt500-{model.__class__.__name__}.html')
        fig2.write_html(f'visualization/pca/pca_window{window_size}_{full_layer}_auc_heads_filter-{heads_filter}_Model-cpt500-{model.__class__.__name__}.html')
    return fig, fig2

def plot_cross_layer_similarity():
    sim_heads_qk = torch.load('data/cross_layer_similarity_full_layerFalse_modes-qk_Model-GPTJForCausalLM_byhead.pt')
    sim_heads_ov = torch.load('data/cross_layer_similarity_full_layerFalse_modes-ov_Model-GPTJForCausalLM_byhead.pt')
    L,_, _,H = sim_heads_qk['qk'].shape
    for mode, sim_heads in [('qk', sim_heads_qk), ('ov', sim_heads_ov)]:
        head_data = rearrange(sim_heads[mode], 'l1 l2 h1 h2 -> (l1 h1) (l2 h2)')
        head_data = head_data.tril()+ head_data.tril(diagonal=-1).T
        hovertext=[]
        for l1 in range(L):
            for h1 in range(H):
                hovertext.append([f'{l1}-{h1}->{l2}-{h2}' for l2 in range(L) for h2 in range(H)])
        fig = go.Figure(data=go.Heatmap(z=head_data.flip(dims=(0,)), hovertext=hovertext[::-1],textfont={"size":9}, hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>%{hovertext}<br>', hoverlabel=dict(font=dict(family='Arial', size=12, color='white')) ))
        fig.update_layout(height= 1000,width=1000, title=f'cross-layer auc similarity of {mode} heads in GPTJForCausalLM')
        fig.write_html(f'visualization/pca/cross_layer_similarity_byhead_cat_modes-{mode}_Model-GPTJForCausalLM.html')

def get_cross_layer_similarity(model, save=False, layer_pairs=None, full_layer='cat', by_head=False, rand_ratio=1, modes=['qk'], device='cpu'):
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    sim = dict([(k, torch.zeros((L,L))) for k in modes])
    #sim_heads = dict([(k, torch.zeros((L,L,H*2,H*2))) for k in modes]) if by_head else None
    sim_heads = dict([(k, torch.zeros((L,L,H,H))) for k in modes]) if by_head else None
    assert not by_head or (by_head and not full_layer) # by_head=True and full_layer=False
    def _subsample(_samples):
        return _samples[torch.randperm(_samples.shape[0])[:int(rand_ratio*_samples.shape[0])]]
    def _get_head_sim(hdata1, hdata2): # (n 2) e1 e2
        #num_heads = hdata.shape[0] # heads in two layers
        num_heads = hdata1.shape[0] # heads in two layers
        data = torch.zeros((num_heads,num_heads))
        for h1 in range(num_heads):
            for h2 in range(num_heads):
            #for h2 in range(h1+1):
                _samples = torch.cat([_subsample(hdata1[h1]), _subsample(hdata2[h2])], dim=0)
                data[h1, h2] = get_cum_var_ratio(_samples.float()).mean()
                #data[h2, h1] = data[h1, h2]
        return data
    for k in tqdm(range(L)):
        for l in range(k+1):
            if layer_pairs is not None and ((k, l) not in layer_pairs and (l, k) not in layer_pairs): continue
            _, _, _, _, qk1, ov1, w1_1, w2_1, wg1 = get_weights(model, k, full_layer=full_layer, device=device)
            _, _, _, _, qk2, ov2, w1_2, w2_2, wg2 = get_weights(model, l, full_layer=full_layer, device=device)
            features = dict([('qk', [qk1, qk2]), ('ov', [ov1, ov2]),('w1', [w1_1, w1_2]), ('w2', [w2_1, w2_2]), ('wg', [wg1, wg2])])
            for f in modes:
                t0 = time.time()
                if by_head:
                    #data = _get_head_sim(torch.cat([features[f][0], features[f][1]], dim=0))
                    data = _get_head_sim(features[f][0], features[f][1])
                    sim_heads[f][k,l] = data
                    sim_heads[f][l,k] = data
                else:
                    samples = torch.cat(features[f], dim=1).squeeze(0)
                    if f in ['qk', 'ov']: 
                        samples = torch.cat([_subsample(features[f][0][0]), _subsample(features[f][1][0])], dim=0)
                    else:
                        samples = torch.cat(features[f], dim=1).squeeze(0)
                    sim[f][k,l] = get_cum_var_ratio(samples.float()).mean()
                    sim[f][l,k] = sim[f][k,l]
                if k ==0:
                    print('time consumed for mode', f, time.time()- t0)
    if save:
        mode_name = '-'.join(modes)
        if by_head:
            torch.save(sim_heads, f'data/cross_layer_similarity_full_layer{full_layer}_layer_pairs-{layer_pairs}_modes-{mode_name}_Model-{model.__class__.__name__}_byhead.pt')
        else:
            torch.save(sim, f'data/cross_layer_similarity_full_layer{full_layer}_modes-{mode_name}_Model-{model.__class__.__name__}.pt')
    return sim, sim_heads
