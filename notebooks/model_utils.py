import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import einops
from einops import rearrange

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJModel
from transformers.models.xglm.modeling_xglm import XGLMAttention, XGLMDecoderLayer

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

from common_utils import numpy, einsum, my_isinstance

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

def fill_list(e, length, i, default_e=None): # fill e to ith position of a list of default_es
    if type(e) == list: assert len(e) == l, f'{len(e)} != {l}'; return e
    l = [default_e for _ in range(length)]
    if i is not None: l[i] = e
    return l

def default_get_hqkv(h): return h, h, h  # h is used for query, key and value
def get_hqkv_k(h, h0): return h0, h, h0  # h is only used for key

def embed_forward(transformer, inputs, output_embeds=True): # gptneo
    self = transformer
    input_ids = inputs.input_ids
    input_shape = input_ids.size()
    device = input_ids.device
    position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids) \
        if not my_isinstance(transformer, GPTJModel) else inputs_embeds * 0.
    hidden_states = inputs_embeds + position_embeds
    hidden_states = self.drop(hidden_states)
    return (hidden_states, inputs_embeds, position_embeds) if output_embeds else hidden_states

def unify(model):
    if not hasattr(model, 'transformer'):
        assert hasattr(model, 'model') # xglm
        model.transformer = model.model
        model.model.h = model.model.layers
        model.model.ln_f = model.model.layer_norm

    for i, block in enumerate(model.transformer.h):
        if my_isinstance(block, XGLMDecoderLayer):
            block.attn = block.self_attn
            block.ln_1 = block.self_attn_layer_norm
            block.ln_2 = block.final_layer_norm
        if hasattr(block.attn, 'attention'): # gptneo
            block.attn = block.attn.attention
        if my_isinstance(block, GPTJBlock): block.ln_2 = block.ln_1

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
            self.attn_dropout = nn.Dropout(self.dropout)
            self.resid_dropout = nn.Dropout(block.dropout)
        elif my_isinstance(self, GPTJAttention):
            self.num_heads = self.num_attention_heads

def _split_heads(tensor, num_heads, attn_head_size, rotary=False):
    '''b i (n d) -> b n i d'''
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    if rotary: return tensor
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

def _merge_heads(tensor, num_heads, attn_head_size):
    '''b n i d -> b i (n d)'''
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)

def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).float()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(2, 3), sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

def _attn(self, query, key, value, attention_mask=None):
    # bias and masked_bias copied from GPTNeoSelfAttention.__init__
    max_positions = 1024  # config.max_position_embeddings
    bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
        1, 1, max_positions, max_positions)
    if isinstance(self, GPTNeoSelfAttention):
        assert self.attention_type in ['global', 'local']
        if self.attention_type == 'local': # GPTNeoSelfAttention
            bias = torch.bitwise_xor(bias, torch.tril(bias, -256)) # config.window_size
    _attn.bias = bias
    _attn.masked_bias = torch.tensor(-1e9)
    
    if isinstance(self, GPTNeoSelfAttention) or isinstance(self, GPTJAttention):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query, key = query.to(torch.float32), key.to(torch.float32)
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) # bnid,bnjd->bnij

    # turns out gptneo fold_scaling_into_initializer, and uses float32_logits. 
    # see https://crfm.stanford.edu/2021/08/26/mistral.html (Diagnosing Numerical Instability, Eureka!)
    # and https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/attention.py#L517&66
    if not isinstance(self, GPTNeoSelfAttention):
        attn_weights = attn_weights / (value.size(-1) ** 0.5)

    # mask handling copied from GPTNeoSelfAttention._attn
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = _attn.bias[:, :, key_length - query_length : key_length, :key_length].bool()
    attn_weights = torch.where(causal_mask, attn_weights, _attn.masked_bias.to(attn_weights.dtype))
    if attention_mask is not None: attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value) # bnij,bnjd->bnid
    return attn_output, attn_weights

def attn_forward(block, hq, hk, hv, attention_mask=None, by_head=False,
                head_mask=None, attn_weights=None): # gptneo
    # hq, hk, hv = block.ln_1(hq), block.ln_1(hk), block.ln_1(hv)
    self = block.attn  # block.attn.attention already renamed
    query = self.q_proj(hq)
    key = self.k_proj(hk)
    value = self.v_proj(hv)

    rotary = my_isinstance(self, GPTJAttention)
    query = _split_heads(query, self.num_heads, self.head_dim, rotary=rotary)
    key = _split_heads(key, self.num_heads, self.head_dim, rotary=rotary)
    value = _split_heads(value, self.num_heads, self.head_dim)

    if rotary:
        seq_len = key.shape[1]
        offset = 0
        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]
        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]

        sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
        k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
        q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

    attn_output, attn_weights = _attn(self, query, key, value, attention_mask) \
        if attn_weights is None else (attn_weights @ value, attn_weights)  # XD
    # print('attn_output.shape',attn_output.shape)
    # print('head_mask.shape',head_mask.shape)
    if head_mask is not None: attn_output = einsum('bnid,bni->bnid', attn_output, head_mask)

    head_input, head_output = None, None
    if by_head:
        w_o = self.out_proj.weight
        # w_o = w_o.view(self.embed_dim, self.num_heads, -1).permute(1, 2, 0)
        w_o = rearrange(w_o, 'e (n d) -> n d e', n=self.num_heads) # d=d_head, e=d_model
        head_input, head_output = value @ w_o, attn_output @ w_o  # bnid,nde->bnie
        head_output = self.resid_dropout(head_output)
        head_input = self.resid_dropout(head_input)

    attn_output = _merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    # if output_by_head:
    #     print(equal(head_output.sum(1) + self.resid_dropout(self.out_proj.bias), attn_output))
    return attn_output, attn_weights, head_input, head_output

def mlp_forward(block, hidden_states, output_intermediate=False): # gptneo
    return block.mlp(block.ln_2(hidden_states), output_intermediate=output_intermediate)

def compute_loss(logits, labels, reduction='none'):
    if labels.size(0) < logits.size(0): # logits has been scaled
        labels = einops.repeat(labels, '1 i -> b i', b=logits.size(0))
    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    # print(labels)
    loss_fct = nn.CrossEntropyLoss(reduction='none' if reduction == 'per_example_mean' else reduction)
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # print(f'logits.size = {logits.size()}, labels.size = {labels.size()}')
    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    # print(loss.shape)
    if reduction != 'mean':
        loss = loss.view(labels.size(0), -1) #4,16
        # print(loss)
        # print(torch.einsum('bi->b', loss),torch.einsum('bi->b', labels != -100))
        if reduction == 'per_example_mean':
            # loss = einops.reduce(loss, 'b i -> b', 'sum') / \
            #     einops.reduce(labels != -100, 'b i -> b', 'sum')
            loss = torch.einsum('bi->b', loss) / torch.einsum('bi->b', labels != -100)
            # print(loss)
    return loss

def scaled_input(input, num_points, baseline=None, requires_grad=True):
    # shape of input: (bsz, num_head, seq_len, seq_len)
    assert input.size(0) == 1
    if baseline is None: baseline = torch.zeros_like(input)   
    step = (input - baseline) / num_points
    # res = torch.cat([baseline + step * i for i in range(num_points)], dim=0)
    res = torch.cat([baseline + step * (i + 1) for i in range(num_points)], dim=0)  # XD
    # alphas = list(0.5 * (1 + np.polynomial.legendre.leggauss(num_points)[0])) # copied from captum
    # res = torch.cat([baseline + alpha * (input - baseline) for alpha in alphas], dim=0)
    if requires_grad: res.requires_grad_(True)
    return res #, step




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

def plot_attn(attn, tokens, annot=False, figsize=(10, 10), ax=None):
    if ax is None: plt.figure(figsize=figsize)
    res = sns.heatmap(numpy(attn), square=True, cbar=False, annot=annot, fmt='d', linewidths=0.1, linecolor='grey', 
                      xticklabels=tokens, yticklabels=tokens, ax=ax)
    _ = res.set_xticklabels(res.get_xmajorticklabels(), fontsize=8, rotation=90)
    _ = res.set_yticklabels(res.get_ymajorticklabels(), fontsize=8, rotation=0)
    # _ = plt.xlabel('%d-%d    %.4f' % (layer, head, v), fontsize=14)
    # res.tick_params(top=True, right=True, labeltop=True, labelright=True)
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

def get_head_weights(layer, head):
    m = get_attn_module(blocks[layer])
    wq = m.q_proj.weight.view(H, -1, hidden_size)[head]
    wk = m.k_proj.weight.view(H, -1, hidden_size)[head]
    wv = m.v_proj.weight.view(H, -1, hidden_size)[head]
    wo = m.out_proj.weight.view(hidden_size, H, -1)[:, head]
#     return wq, wk, wv, wo
    return wq.t(), wk, wv.t(), wo.t()

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


def test(hidden, query, key=None, logits=None, always_show=False):
    if logits is None:
        if key is None:
            key = self.k_proj(hidden)
            key = self._split_heads(key, self.num_heads, self.head_dim)[0, head2]
        logits = (query * key).sum(dim=-1)
    else:
        always_show = True
    cand_pos = torch.LongTensor(cand_positions).view(-1, n_candidates)
    is_extremal = [logits[p] == logits[cand_pos[i]].max() for i, p in enumerate(tgt_positions)]
    if always_show or sum(is_extremal[1:]) / len(tgt_positions[1:]) > 0.9:
        logits[0] = logits[1]
        plot(logits)
        _ = plt.xticks(range(len(logits)), tokens)
        for p, b in zip(tgt_positions, is_extremal): plt.axvline(x=p, color='gray' if b else 'r')
        plt.show()
        probs = logits[cand_positions].view(-1, n_candidates).softmax(-1)[cand_is_tgt]
        print(numpy(probs), '\n', probs.mean())
        return True
    return False 


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

combined_weights = {}

def get_combined_w(layer, head, qk=False):
    if (layer, head, qk) in combined_weights: return combined_weights[(layer, head, qk)]
    wq, wk, wv, wo = get_head_weights(layer, head)
    w = torch.matmul(wq, wk) if qk else torch.matmul(wv, wo)
    combined_weights[(layer, head, qk)] = w
    return w


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