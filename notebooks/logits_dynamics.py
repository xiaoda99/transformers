from model_utils import *

import circuitsvis as cv
import plotly.express as px
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def get_tokens_from_ids(iids, tokenizer, replace=True):
    return [t if not replace else t.replace('Ġ',' ').replace('Ċ', '___________') for t in tokenizer.convert_ids_to_tokens(iids)] 

def show_interactive_heads(outputs, input_ids, tokenizer, layer=0, head=None, token_idx=None):
    attentions = outputs.attentions; str_tokens = get_tokens_from_ids(input_ids, tokenizer)  # attentions (b,n,i,j)
    B, heads_num, i, j = attentions[0].shape
    print(f'layer:{layer}, head:{head}, token:{str_tokens[token_idx] if token_idx is not None else None}')
    if layer == 'all': # show all layer head at token with idx, (layer='all', token_idx=-3)
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


def batch_logits_lens(data_tuples, model, tokenizer, task_name='', tgt=None, scan_layer=[], orders=[2], case_sensitive=False, save_fig=False, **kwargs):
    tuples = []; demo = ['', '']; heatmap = True if len(data_tuples)==1 else False
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
            #print('ranges', dt[3])
            if (token_idxs[1] - token_idxs[0])>1 or (query_idxs[1] - query_idxs[0])>1: demo = tgt #print('two tokens in query or target', tgt, dt[0])
        tuples.append((o, token_idxs[1]-1, tgt, input_ids))
    task_sample = data_tuples[0][0].split('\n')[-2]
    if not task_name: task_name = f'{demo[1]}->{demo[0]}' 
    _save_fig = True if save_fig and heatmap else False 
    logits_dict = mr(logits_lens)(tuples, model, tokenizer, plot_ranks=False, task_name=task_name, scan_layer=scan_layer, heatmap=heatmap, save_fig=False, **kwargs)
    if save_fig: mr(logits_lens)(tuples[:1], model, tokenizer, plot_ranks=False, task_name=task_name, scan_layer=scan_layer, heatmap=True, save_fig=save_fig, **kwargs)
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
    fig.show() if not save_fig else fig.write_html(f'results_logits_lens/batch_logits_lens_model-{model.__class__.__name__}_task-{task_name}.html')
    return 

def logits_lens(arg_tuples, model, tokenizer, residual_types=['all', 'attn', 'mlp'], task_name='', topk=10000, topk_out=10, plot_ranks=True, scan_layer=[], verbose=False, heatmap=False, save_fig=False, metric='target_diff'):
    outputs, token_idx, tgt, input_ids = arg_tuples
    print(f'tgt{tgt}')
    def _show_topk_rank(raw_logits, tokenizer, topk=10000, tgt=' Mary', head_idx='all', l_idx=None, token_idx=-1, topk_out=5):
        logits, pred_ids = torch.topk(raw_logits, topk, dim=-1)
        tgts_idx = torch.tensor(tokenizer.encode(tgt))[:2] if not my_isinstance(tokenizer, LlamaTokenizer) else torch.tensor([tokenizer.convert_tokens_to_ids(tgt)])[0]
        tgt_idx = torch.where(pred_ids[:1,token_idx] == tgts_idx.unsqueeze(1))[1]
        token_logits = raw_logits[0,token_idx]
        logits_diff = logits[0, token_idx, tgt_idx[0]] - logits[0, token_idx, tgt_idx[1]]
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
            pred_tokens = [(get_tokens_from_ids(pred_id[:topk_out], tokenizer, replace=False), _logits[:topk_out]) for pred_id, _logits in zip(pred_ids[0], logits[0])]
            heatd.append(confid); heat_texts.append(pred_tokens); ylabels.append(lens_key)

    def _logits_lens(hid, model):
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
    if verbose:
        for k, v in logits_dict.items(): print(k, v.shape if isinstance(v, torch.Tensor) else v)
    if heatmap:
        heatd = torch.stack(heatd) # (L * 3, n) 
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
        fig.show() if not save_fig else fig.write_html(f'results_logits_lens/logits_diff_heatmap_model-{model.__class__.__name__}_task-{task_name}_sample-{tgt[0]}-{tgt[1]}.html')
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
    def _rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    def _apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed
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
        logits  = query[0,0] @ key[0,0].T
        attention = torch.softmax(logits  / (query.size(-1) ** 0.5) , dim=-1).tril()
        ov_out = emb @ wv @ wo
        out = attention @ ov_out
        return out, attention, logits, ov_out
    return fn

def construct_circuits(data_tuple, model, tokenizer, layers=None, heads=None, from_scratch=False, circuits=[]):
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
                    head_out = o.head_outputs[hd_layer][:,head_idx] if head_idx is not None else 0
                    emb1 = o.hidden_states[hd_layer] + head_out 
            #heads in first three layers: (0,7), (0,13), (1,3),(1,7),(1,10),(1,11),(2,8),(2,3),(2,11); after three layers:(3,5), (3,6),(4,6),(6,2),(7,6),(8,1),(9,14),(11,12)
            for _layer,_head in circuits:
                out, attention, logits, ov_out = head_forward_fn(model, layer=_layer,head=_head)(emb1)
                emb1 = emb1 + out
        #     print('qk norm', torch.norm(query[0,0],dim=-1).max(),  torch.norm(key[0,0],dim=-1).max())
            #logits  = query[0,0] @ key[0,0].T #/ (torch.norm(query[0,0],dim=-1).unsqueeze(1) * torch.norm(key[0,0],dim=-1).unsqueeze(0))
        #     print(emb1.shape, query.shape, key.shape, logits.shape)
            qk_logits, top_idx = torch.topk(logits[token_idx], logits.shape[1])
            ov_logits = model.transformer.ln_f(out)[0] @ model.lm_head.weight.data.T
            print('ov', model.transformer.ln_f(out).shape, ov_logits.shape)
            ov_logits, ov_idxs = torch.topk(ov_logits[token_idx], 8)
            ov_tokens = tokenizer.convert_ids_to_tokens(ov_idxs)
            #attention = torch.softmax(logits  / head_dim ** 0.5, dim=-1).tril()
            print(f'layer:{hd_layer},head:{head_idx}', 'token logits', logits[token_idx].max(), logits[token_idx].max()-logits[token_idx].min(),)# logits[token_idx].long())
            print('qk----','_'.join([f'{_t}{_logit:.2f}' for _t,_logit in zip([ts[_i] for _i in top_idx[:8]], qk_logits[:8])]))
            print('ov----','_'.join([f'{_t}{_logit:.2f}' for _t,_logit in zip(ov_tokens, ov_logits)]))
            # print('token attn', attention[token_idx].sum(), (attention[token_idx].round(decimals=3)*1000).long())
            attentions.append(attention)
    html = cv.attention.attention_patterns(tokens=ts, attention=torch.stack(attentions))
    return html
