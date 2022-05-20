# from ast import pattern
from collections import OrderedDict
from difflib import restore
# from typing import Iterable
from collections.abc import Iterable
from matplotlib import scale  # same as typing.Iterable?
import numpy as np
import math
from dataclasses import dataclass
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

from common_utils import numpy, einsum, my_isinstance, convert_ids_to_tokens, show_topk, topk_md, equal, join_lists

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
    topk: int = None
    layer: int = None
    head: int = None
    label_type: str = None
    attribute_k: bool = False
    attr: Attributions = None

def fill_list(e, length, i, default_e=None): # fill e to ith position of a list of default_es
    if type(e) == list: assert len(e) == length, f'{len(e)} != {length}'; return e
    l = [default_e for _ in range(length)]
    if i is not None: l[i] = e
    return l

def default_get_hqkv(h): return h, h, h  # h is used for query, key and value
def get_hqkv_k(h, h0): return h0, h, h0  # h is only used for key

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

def scaled_ln(ln, x, scaled=True):
    if not scaled: return ln(x)
    self = ln
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    std = std[-1:].detach()  # use the std of the last original (unscaled) example in the batch
    y = (x - mean) * self.weight / std + self.bias
    return y

def scaled_ln_wrapper(ln): return lambda x: scaled_ln(ln, x)

def embed_forward(transformer, inputs, output_embeds=True): # gptneo
    self = transformer
    input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs.input_ids
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
        # attn_weights = attn_weights / (value.size(-1) ** 0.5) # vale may be None
        attn_weights = attn_weights / (query.size(-1) ** 0.5)

    # mask handling copied from GPTNeoSelfAttention._attn
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = _attn.bias[:, :, key_length - query_length : key_length, :key_length].bool()
    attn_weights = torch.where(causal_mask, attn_weights, _attn.masked_bias.to(attn_weights.dtype))
    if attention_mask is not None: attn_weights = attn_weights + attention_mask
    if value is None: return None, attn_weights

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value) # bnij,bnjd->bnid
    return attn_output, attn_weights

def backup_heads(self):
    self.backup_names = ['num_heads', 'q_proj', 'k_proj', 'v_proj', 'out_proj']
    for name in self.backup_names: setattr(self, name + '0', getattr(self, name))

def restore_heads(self):
    for name in self.backup_names:
        setattr(self, name , getattr(self, name + '0'))
        # delattr(self, name + '0')

def trim_heads(self, kept_heads):
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        # assert getattr(self, proj_name).bias is None #lxy，报错
        weight = getattr(self, proj_name).weight
        if proj_name == 'out_proj':
            patterns = ['e (n d) -> n d e', 'n d e -> e (n d)']
            size = (self.head_dim * len(kept_heads), self.embed_dim)
        else:
            patterns = ['(n d) e -> n d e', 'n d e -> (n d) e']
            size = (self.embed_dim, self.head_dim * len(kept_heads))
        weight = rearrange(weight, patterns[0], n=self.num_heads)[kept_heads] #[kept_heads,head_dim,embed_dim]
        weight = rearrange(weight, patterns[1])
        proj = nn.Linear(*size, bias=False)
        proj.weight = nn.Parameter(weight)
        setattr(self, proj_name, proj)
    self.num_heads = len(kept_heads)

def attn_forward(block, hq, hk, hv, attention_mask=None, by_head=True, compute_head_input=True,
                head_mask=None, attn_weights=None): # gptneo
    self = block.attn  # block.attn.attention already renamed
    if hq is not None and hk is not None and attn_weights is None:
        rotary = my_isinstance(self, GPTJAttention)
        if True: #head is None:
            key = self.k_proj(hk)
            key = _split_heads(key, self.num_heads, self.head_dim, rotary=rotary)
            if hq.ndim == 3:  # bie
                query = self.q_proj(hq)
                query = _split_heads(query, self.num_heads, self.head_dim, rotary=rotary)
            else:
                assert hq.ndim == 4  # bnid, computed in cat_attn_forward
                if rotary: hq = rearrange(hq, 'b n i d -> b i n d')
                assert hq.size()[1:] == key.size()[1:], f'{hq.size()} != {key.size()}'
                query = hq
        # else:
        #     assert self.q_proj.bias is None and self.k_proj.bias is None
        #     w_q = rearrange(self.q_proj.weight, '(n d) e -> n d e', n=self.num_heads)[head]
        #     w_k = rearrange(self.k_proj.weight, '(n d) e -> n d e', n=self.num_heads)[head]
        #     query, key = hq @ w_q.T, hk @ w_k.T  # bie,ed->bid
        #     pattern = 'b i d -> b i 1 d' if rotary else 'b i d -> b 1 i d'
        #     query, key = rearrange(query, pattern), rearrange(key, pattern)

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

    value = _split_heads(self.v_proj(hv), self.num_heads, self.head_dim) if hv is not None else None

    attn_output, attn_weights = _attn(self, query, key, value, attention_mask) \
        if attn_weights is None else (attn_weights @ value, attn_weights)  # XD
    if value is None: return None, attn_weights, None, None

    if head_mask is not None: attn_output = einsum('bnid,bni->bnid', attn_output, head_mask)

    head_input, head_output = None, None
    if by_head:
        w_o = self.out_proj.weight
        # w_o = w_o.view(self.embed_dim, self.num_heads, -1).permute(1, 2, 0)
        w_o = rearrange(w_o, 'e (n d) -> n d e', n=self.num_heads) # d=d_head, e=d_model
        head_output = attn_output @ w_o  # bnid,nde->bnie
        head_output = self.resid_dropout(head_output)
        if self.num_heads < getattr(self, 'num_heads0', 0): compute_head_input = False
        head_input = self.resid_dropout(value @ w_o) if compute_head_input else None

    attn_output = _merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    # if output_by_head:
    #     print(equal(head_output.sum(1) + self.resid_dropout(self.out_proj.bias), attn_output))
    return attn_output, attn_weights, head_input, head_output

def head_forward(model, hidden_states, layer, head, labels=None, loss_reduction=None,
            attn_weights=None, attn_labels=None, hidden_states0=None, attribute_k=False, trim=True, scaled=True):
    if isinstance(layer, Iterable):  # tuple, list or np.ndarray
        assert labels is None and attn_labels is not None
        assert isinstance(attn_labels, (list, tuple))
        assert isinstance(hidden_states0, (list, tuple))
        if not isinstance(hidden_states, (list, tuple)): hidden_states = [hidden_states] * len(layer)
        assert len(attn_labels) == len(hidden_states) == len(hidden_states0) == len(layer) == len(head)
        outputs = [head_forward(model, hs, l, h, loss_reduction=loss_reduction,
            attn_labels=al, hidden_states0=hk, trim=trim, scaled=scaled)
            for hs, l, h, al, hk in zip(hidden_states, layer, head, attn_labels, hidden_states0)]
        return Outputs(hidden_states=[o.hidden_states for o in outputs],
            logits=[o.logits for o in outputs], loss=sum(o.loss for o in outputs))

    block = model.transformer.h[layer]
    # only hq and hv can be scaled, not hk
    hk = block.ln_1(hidden_states0 if hidden_states0 is not None else hidden_states)
    h = scaled_ln(block.ln_1, hidden_states, scaled=scaled) #if scaled else block.ln_1(hidden_states)
    if attn_weights is not None:
        hq, hv = None, h
    elif attn_labels is not None:
        hq, hv = h, None # return attn_logits instead of attn_weights by passing None hv 
        if attribute_k: hq, hk = hk, hq

    if trim:
        self = block.attn
        backup_heads(self)
        try:
            trim_heads(self, [head])
            if attn_weights is not None: attn_weights = attn_weights[:, [head]] # bnij->b1ij
            _, attn_logits, _, head_output = attn_forward(block, hq, hk, hv, attn_weights=attn_weights)#
            assert head_output is None or head_output.size(1) == 1, str(head_output.size())
            head = 0  # used to index head_output and attn_logits
            restore_heads(self)
        except Exception:
            restore_heads(self)
            raise  # print(traceback.format_exc())
    else:
        _, attn_logits, _, head_output = attn_forward(block, hq, hk, hv, attn_weights=attn_weights)
    if head_output is not None: head_output = head_output[:, head]
    logits, loss = None, None
    if labels is not None:
        logits, loss = lm_head_forward(model, head_output, labels=labels, loss_reduction=loss_reduction, scaled=scaled)
    elif attn_labels is not None:
        # may do some attn_logits masking here
        logits = attn_logits[:, head]  # bnij->bij
        # print('in head_forward, logits =', torch.einsum('bij->b', logits * torch.ones_like(logits).tril()))
        if not attribute_k: logits = logits.log_softmax(-1)
        loss = -torch.einsum('bij->b', logits * attn_labels) # per_example_sum. per_example_mean is hard to define when using unormalized attn attr  # bug fix
    return Outputs(hidden_states=(head_output,) if head_output is not None else (), logits=logits, loss=loss)

def mlp_forward(block, hidden_states, layer=None, output_intermediate=False, 
                labels=None, loss_reduction=None, scaled=False):
    if layer is not None:
        model = block
        block = model.transformer.h[layer]
    hidden_states = scaled_ln(block.ln_2, hidden_states, scaled=scaled)
    if layer is None: return block.mlp(hidden_states, output_intermediate=output_intermediate)
    hidden_states = block.mlp(hidden_states, output_intermediate=False)
    logits, loss = lm_head_forward(model, hidden_states, labels=labels, loss_reduction=loss_reduction, scaled=scaled) \
        if labels is not None else (None, None)
    return Outputs(hidden_states=(hidden_states,), logits=logits, loss=loss)

def lm_head_forward(model, hidden_states, labels=None, loss_reduction=None, compact=True, scaled=False):
    if compact and labels is not None:
        if labels.size(0) != hidden_states.size(0): labels = einops.repeat(labels, '1 i -> b i', b=hidden_states.size(0))
        valid_flags = labels != -100
        n_valid = torch.einsum('bi->b', valid_flags)[0].item()
        hidden_states = rearrange(hidden_states[valid_flags], '(b i) e -> b i e', b=hidden_states.size(0), i=n_valid)
        labels = rearrange(labels[valid_flags], '(b i) -> b i', b=labels.size(0), i=n_valid)
    logits = model.lm_head(scaled_ln(model.transformer.ln_f, hidden_states, scaled=scaled))
    loss = compute_loss(logits, labels, reduction=loss_reduction) if labels is not None else None
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
            # print('in compute_loss, before loss =', loss)
            loss = torch.einsum('bi->b', loss) / torch.einsum('bi->b', labels != -100)
            # print('in compute_loss, after loss =', loss)
    return loss

def get_argmax_labels(model, hidden_states, labels):
    logits = model.lm_head(model.transformer.ln_f(hidden_states))
    argmax_labels = labels.clone()
    argmax_labels[labels != -100] = logits.argmax(-1)[labels != -100]
    return argmax_labels

def get_prob_dist(d, topk=5):
    return {k: round(math.exp(v), 3) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk]}

def show_predictions(text, examples, tokenizer, logits, bos_indices, eos_indices, answers, labels, 
        mask_logits_fn=None, topk=5, loss_reduction='mean', show_range=None, sep='\t', verbose=True):
    # use_openai_api = isinstance(model, types.FunctionType)
    use_openai_api = hasattr(logits, 'token_logprobs')
    if use_openai_api: ans_nlls = []
    if not use_openai_api and mask_logits_fn is not None: logits = mask_logits_fn(logits)
    
    bi = 0
    assert len(bos_indices) == len(examples), '%d != %d' % (len(bos_indices), len(examples))
    if show_range is None: show_range = range(len(examples))
    top1_corrects = []  # True
    for i, (example, bos_i, eos_i, ans_ids) in enumerate(zip(examples, bos_indices, eos_indices, answers)):
        # eos_i = bos_i + 2  # show only the first answer token
        if i not in show_range: continue
        if use_openai_api:
            ans_prob_dist = [get_prob_dist(d, topk=topk) for d in logits.top_logprobs[bos_i + 1: eos_i]]
            ans_probs = [math.exp(lp) for lp in logits.token_logprobs[bos_i + 1: eos_i]]
            ans_nlls += [-lp for lp in logits.token_logprobs[bos_i + 1: eos_i]]
        else:
            ans_prob_dist = logits[bi, bos_i: eos_i - 1].softmax(-1)
            ans_probs = ans_prob_dist[torch.arange(ans_prob_dist.size(0)), ans_ids]
        ans_tokens = convert_ids_to_tokens(ans_ids, tokenizer)
        for ans_id, ans_token, ans_prob, dist in zip(ans_ids, ans_tokens, numpy(ans_probs, decimals=3), ans_prob_dist):
            top1_correct = max(dist.items(), key=lambda x: x[1])[0] == ans_token.replace('Ġ', ' ') \
                if use_openai_api else (dist.argmax() == ans_id).item()
            top1_corrects.append(top1_correct)
            indices_fn = partial(convert_ids_to_tokens, tokenizer=tokenizer)
            if verbose: 
                print(('*' if top1_correct else ' ') + ans_token, ans_prob, dist if use_openai_api 
                    else show_topk(*dist.topk(topk), indices_fn=indices_fn), sep, example) 
    if use_openai_api:
        loss = ans_nlls if loss_reduction == 'none' else sum(ans_nlls) / len(ans_nlls)
    else:
        loss = compute_loss(logits, labels, reduction=loss_reduction)
    # all_top1_correct = sum(not correct for correct in all_top1_correct) < 1
    return loss, top1_corrects

def sum_forward(model, outputs, labels=None, loss_reduction='per_example_mean', 
        embed_mask=None, mlp_mask=None, head_mask=None, neuron_mask=None, attn_weights=None, 
        reduce_fn=sum, truncate_layers=False, scaled=True, reshape=False, output_layer=None):
    embed_output = outputs.hidden_states[0]
    if embed_mask is not None: embed_output = einsum('bie,bi->bie', embed_output, embed_mask)

    head_outputs = rearrange(list(outputs.head_outputs), 'l 1 n i e -> 1 l n i e')
    if reduce_fn == torch.cat and head_mask is None and attn_weights is not None:
        head_mask = einops.reduce(attn_weights, '1 l n i j -> 1 l n i', 'sum')
        attn_weights = None
    if head_mask is not None:
        head_outputs = einsum('blni,blnie->blnie', head_mask, head_outputs)
    elif neuron_mask is not None:
        head_outputs = einsum('blnie,blnie->blnie', neuron_mask, head_outputs)
    elif attn_weights is not None:
        head_inputs = rearrange(list(outputs.head_inputs), 'l 1 n i e -> 1 l n i e')
        head_outputs = einsum('blnij,blnje->blnie', attn_weights, head_inputs)

    mlp_outputs = rearrange(list(outputs.mlp_outputs), 'l 1 i e -> 1 l i e')
    if mlp_mask is not None:
        # print('in sm_forward, mlp_outputs.size(), mlp_mask.size() =', mlp_outputs.size(), mlp_mask.size())
        mlp_outputs = einsum('blie,bli->blie', mlp_outputs, mlp_mask)
    if reshape:  # for head amplication, obsolete
        assert reduce_fn == torch.cat
        L = (torch.einsum('bli->l', mlp_mask) > 0).sum().item() \
            if mlp_mask is not None and truncate_layers else len(outputs.mlp_outputs)
        attn_outputs = rearrange(head_outputs[:, :L], '1 l n i e -> l n i e')
        mlp_outputs = rearrange(mlp_outputs[:, :L], '1 l i e -> l 1 i e')
        padded_embed_output = embed_output.new(*((L,) + embed_output.size())).zero_() # l1ie
        padded_embed_output[0] = embed_output
        output = torch.cat([attn_outputs, mlp_outputs, padded_embed_output], dim=1) # lnie,l1ie,l1ie->l(n+2)ie
    elif output_layer is not None:  # default
        assert reduce_fn == sum
        if isinstance(output_layer, Iterable):
            # output = []
            # for l in output_layer:
            #     attn_outputs = torch.einsum('blnie->bie', head_outputs[:, :l])
            #     _mlp_outputs = torch.einsum('blie->bie', mlp_outputs[:, :l])
            #     output.append(reduce_fn([embed_output, attn_outputs, _mlp_outputs])) # bie for sum
            output = [reduce_fn([embed_output,
                                torch.einsum('blnie->bie', head_outputs[:, :l]), 
                                torch.einsum('blie->bie', mlp_outputs[:, :l])])
                    for l in output_layer]
        else:
            attn_outputs = torch.einsum('blnie->bie', head_outputs[:, :output_layer])
            mlp_outputs = torch.einsum('blie->bie', mlp_outputs[:, :output_layer])
            output = reduce_fn([embed_output, attn_outputs, mlp_outputs]) # bie for sum
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

    logits, loss = None, None
    if labels is not None:
        ln_output = scaled_ln(model.transformer.ln_f, output, scaled=scaled)
        logits = model.lm_head(ln_output)
        loss = compute_loss(logits, labels, reduction=loss_reduction)
    return Outputs(hidden_states=(output,), logits=logits, loss=loss)

def cat_attn_forward(block, cat_hidden_states, sum_hidden_states, mask=None, attn_labels=None, scaled=True):
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

def scaled_input(input, num_points, baseline=None, requires_grad=True):
    assert input.size(0) == 1
    if baseline is None: baseline = torch.zeros_like(input)
    if num_points == 3:
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
        return -outputs.loss, outputs.logits
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

def attribute(forward_fn, model, x, post_forward_fn=[], num_points=10, batch_size=11):
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
                # if i == 0: grad[key + '0'], grad[key + '1'] = grad_[j][0:1] * num_points, grad_[j][1:2] * num_points
                # print(key, 'grad', grad_[j].reshape(grad_[j].size(0), -1).sum(1)) # debug
    attr = {key: (grad[key] / num_points * x[key]).squeeze(0) for key in x}
    # attr.update({key + '0': (grad[key + '0'] / num_points * x[key]).squeeze(0) for key in x})
    # attr.update({key + '1': (grad[key + '1'] / num_points * x[key]).squeeze(0) for key in x})

    attn_attr = attr.get('attn_weights')
    neuron_attr = attr.get('neuron_mask')
    head_attr = torch.einsum('lnij->ln', attn_attr) if attn_attr is not None \
        else torch.einsum('lnie->ln', neuron_attr)
    mlp_attr = attr['mlp_mask'].sum(-1, keepdim=False) # li->l
    emb_attr = attr['embed_mask'].sum(-1, keepdim=True) # i->1
    attr = Attributions(attn=attn_attr, head=head_attr, neuron=neuron_attr, mlp=mlp_attr, embed = emb_attr)
    return attr, torch.cat(ys), logits

def attribute2(forward_fn, model, x, post_forward_fn):
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=False)
    with torch.no_grad():
        o = forward_fn(model, **x)
        y, logits = post_forward_fn(model, o)
    assert y.ndim == 1
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    # assert (y.size(0) - 1) % (H + 1) == 0
    # to_layer = (y.size(0) - 1) // (H + 1) # by solving equation b = 1 + ln + l
    # assert to_layer == (torch.einsum('blni->l', x['head_mask']) > 0).sum().item()
    embed_attr = y[:1].view(1)
    head_attr = y[1: 1 + L * H].view(L, H)  # ln
    mlp_attr = y[1 + L * H:]#.view(L, 1)  # l1
    return Attributions(embed=embed_attr, head=head_attr, mlp=mlp_attr)

def attribute22(forward_fn, model, x, post_forward_fn):
    if isinstance(post_forward_fn, (list, tuple)):
        post_forward_fn = compose_forward_fns(post_forward_fn, scaled=False)
    with torch.no_grad():
        o = forward_fn(model, **x, truncate_layers=True)
        y, logits = post_forward_fn(model, o)
    assert y.ndim == 1
    L, H = len(model.transformer.h), model.transformer.h[0].attn.num_heads
    head_attr, mlp_attr = torch.zeros(L, H), torch.zeros(L)
    assert (y.size(0) - 1) % (H + 1) == 0
    to_layer = (y.size(0) - 1) // (H + 1) # by solving equation b = 1 + ln + l
    assert to_layer == (torch.einsum('bli->l', x['mlp_mask']) > 0).sum().item()
    L = to_layer
    embed_attr = y[:1]#.view(1, 1)
    head_attr[:], mlp_attr[:] = embed_attr.item(), embed_attr.item()
    head_attr[:L] = y[1: 1 + L * H].view(L, H)  # ln
    mlp_attr[:L] = y[1 + L * H:]#.view(L, 1)  # l1
    return Attributions(embed=embed_attr, head=head_attr, mlp=mlp_attr)

def get_x(key, outputs, to_layer=None):
    L, H = len(outputs.attentions), outputs.attentions[0].size(1)
    # L dim removed when doing per-layer attribution
    if key == 'head_mask': x = torch.ones(1, L, H, outputs.hidden_states[0].size(1))
    elif key == 'neuron_mask': x = torch.ones(1, L, H, outputs.hidden_states[0].size(1), outputs.hidden_states[0].size(-1))
    elif key == 'mlp_mask': x = torch.ones(1, L, outputs.hidden_states[0].size(1))
    elif key == 'embed_mask': x = torch.ones(1, outputs.hidden_states[0].size(1))
    elif key == 'attn_weights': x = rearrange(list(outputs.attentions), 'l 1 n i j -> 1 l n i j')
    if to_layer is not None and x.size(1) == L: x[:, to_layer:] = 0
    return x

def plot_attr(attr, attr2):
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 4))
    def concat_attrs(head_attr, mlp_attr, embed_attr): return torch.cat([head_attr, einops.repeat(mlp_attr, 'l -> l 3'), einops.repeat(embed_attr, '1 -> l 3', l = head_attr.size(0))], dim=1)
    for ax, a in zip(axs, [concat_attrs(attr.head, attr.mlp, attr.embed), concat_attrs(attr2.head, attr2.mlp, attr2.embed)]):
        res = sns.heatmap(a, cbar=True, ax=ax)
        _ = res.set_yticklabels(res.get_ymajorticklabels(), rotation=0)
        res.tick_params(top=False, right=True, labeltop=False, labelright=True)
    plt.show()

def get_head_rank(head_attr, layer, head, topk=20):
    if head is not None:
        head2rank = {k: v for k, v in zip(zip(*topk_md(head_attr, topk)[:2]), range(topk))}
        return head2rank.get((layer, head), None)
    else: # mlp
        head2rank = {k: v for k, v in zip(zip(*topk_md(head_attr, topk)[:1]), range(topk))}
        return head2rank.get((layer,), None)

def get_head_weights(model, layer, head=None, transpose=False):
    m = model.transformer.h[layer].attn
    H = m.num_heads
    # wq = m.q_proj.weight.view(H, -1, embed_dim)[head]
    # wk = m.k_proj.weight.view(H, -1, embed_dim)[head]
    # wv = m.v_proj.weight.view(H, -1, embed_dim)[head]
    # wo = m.out_proj.weight.view(embed_dim, H, -1)[:, head]
    if head is None: head = range(H)
    (qkv_pattern, o_pattern) = ('(n d) e -> n d e', 'e (n d) -> n e d') \
        if not transpose else ('(n d) e -> n e d', 'e (n d) -> n d e')
    wq, wk, wv = [rearrange(getattr(m, name).weight, qkv_pattern, n=H)[head]
                for name in ['q_proj', 'k_proj', 'v_proj']]
    wo = rearrange(getattr(m, 'out_proj').weight, o_pattern, n=H)[head]
    # if transpose: wq, wk, wv, wo = wq.transpose(-2, -1), wk.transpose(-2, -1), wv.transpose(-2, -1), wo.transpose(-2, -1)
    return wq.data, wk.data, wv.data, wo.data

def combine_weights(weights, qk=True, with_embedding=False, BA=False):
    wq, wk, wv, wo = weights
    wqt = wq.t()
    if with_embedding:
        wqt, wk = we.t().mm(wqt), wk.mm(we)
        wo, wv = wu.mm(wo), wv.mm(we)
    if BA: return wk.mm(wqt) if qk else wv.mm(wo)
    return wqt.mm(wk) if qk else wo.mm(wv)

def plot_eigv(w, start_i=0, end_i=None, alpha=0.1, plot=True):
    # w = w.detach()#.numpy()
    x, y = w[:, 0], w[:, 1]
    eigv_positivity = x.sum() / (x**2 + y**2).sqrt().sum()
    if plot:
        if start_i is None: start_i = 0
        if end_i is None: end_i = len(w)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x[start_i: end_i], y[start_i: end_i], '.', alpha=alpha); plt.show()
    return eigv_positivity.item()

def get_eigv_pos(m): return plot_eigv(m.eig()[0], plot=False)

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
            q, k = (we.T @ wq), (we.T @ wk).T
            # print('in compute_eigv_positivity', k.size(), q.size(), (k @ q).size())
            eig_qk0 = get_eigv_pos(k @ q)
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
            ln_1 = blocks[heads_1[0][0]].ln_1
            wvo_1 = sum(torch.matmul(*get_head_weights(model, l, h, transpose=True)[2:]) for l, h in heads_1)
            hq0 = wvo_1 @ hq0
        if use_ln: 
            ln0 = blocks[l0].ln_1
            ln1 = ln0 = Id
            # hq = ln1((ln0(_e) @ wv0 @ wo0)) @ wv1 @ wo1 if heads_1 is None else \
            #     ln1((ln0(ln_1(_e) @ wvo_1) @ wv0 @ wo0)) @ wv1 @ wo1
            hq = ln1((ln0(_e) @ wv0 @ wo0)) @ wv1 @ wo1 if isinstance(_e, torch.Tensor) else \
                rearrange([ln1((ln0(e) @ wv0 @ wo0)) @ wv1 @ wo1 for e in _e[: l0 + 1]], 'l v e -> l v e')
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

def get_conductivity(eigv_positivity012, l1, h1, plot=False, figsize=(5, 2)):
    x = eigv_positivity012.get((l1, h1))
    if x is None: return 0.
    if plot: plt.figure(figsize=figsize); plt.hist(x, 20); _ = plt.title(f'{l1}-{h1}'); plt.show()
    return np.abs(np.array(x)).mean()

def plot_k_comp(heads, k_compositions, pos_heads2val):
    ls, hs = zip(*heads)
    _ = plt.figure(figsize=(20, len(heads) * 0.5))
    _ = sns.heatmap(torch.cat([k_compositions[list(ls), list(hs)], torch.Tensor(list(pos_heads2val.values())).unsqueeze(0)]), cbar=False,
        xticklabels=[f'{l}-{h}' for l, h in pos_heads2val.keys()], yticklabels=[f'{l}-{h}' for l, h in heads])

def add_attr(head_tuples, attr_dicts):
    if not isinstance(attr_dicts, (tuple, list)): attr_dicts = [attr_dicts]
    return [[l, h, *v] + [attr_dict[l, h] for attr_dict in attr_dicts] for l, h, *v in head_tuples]

# losses2 = []
# sum_output = head_outputs * 0
# # hq, hk, hv = block.ln_1(sum_output), block.ln_1(o.hidden_states[layer[0]]), None
# # with torch.no_grad(): attn_logits = attn_forward(block, hq, hk, hv)[1]
# # logits = attn_logits[:, head[0]]  # bnij->bij
# # loss = -torch.einsum('bij->b', logits.log_softmax(-1) * attn_labels / attn_labels.sum())
# # print(-1, -1, 0, loss.item())
# # losses2.append(loss.item())
# for l, h, v in list(zip(*topk_md(head_attr2[:layer[0]], head_attr2.numel())))[:]:
#     sum_output = sum_output + o.head_outputs[l][:, h]
#     hq, hk, hv = block.ln_1(sum_output), block.ln_1(o.hidden_states[layer[0]]), None
#     with torch.no_grad(): attn_logits = attn_forward(block, hq, hk, hv)[1]
#     logits = attn_logits[:, head[0]]  # bnij->bij
#     loss = -torch.einsum('bij->b', logits.log_softmax(-1) * attn_labels / attn_labels.sum())
#     print(l, h, v, loss.item())
#     losses2.append(loss.item())


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

def plot_attn(attn, tokens, annot=False, figsize=(15, 15), ax=None):
    if ax is None: plt.figure(figsize=figsize)
    res = sns.heatmap(numpy(attn), square=True, cbar=False, annot=annot, fmt='d', linewidths=0.1, linecolor='grey', 
                      xticklabels=tokens, yticklabels=tokens, ax=ax)
    _ = res.set_xticklabels(res.get_xmajorticklabels(), fontsize=10, rotation=90)
    _ = res.set_yticklabels(res.get_ymajorticklabels(), fontsize=10, rotation=0)
    # _ = plt.xlabel('%d-%d    %.4f' % (layer, head, v), fontsize=14)
    res.tick_params(top=True, right=True, labeltop=True, labelright=True)
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

# def get_head_weights(layer, head):
#     m = get_attn_module(blocks[layer])
#     wq = m.q_proj.weight.view(H, -1, hidden_size)[head]
#     wk = m.k_proj.weight.view(H, -1, hidden_size)[head]
#     wv = m.v_proj.weight.view(H, -1, hidden_size)[head]
#     wo = m.out_proj.weight.view(hidden_size, H, -1)[:, head]
# #     return wq, wk, wv, wo
#     return wq.t(), wk, wv.t(), wo.t()

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


# def test(hidden, query, key=None, logits=None, always_show=False):
#     if logits is None:
#         if key is None:
#             key = self.k_proj(hidden)
#             key = self._split_heads(key, self.num_heads, self.head_dim)[0, head2]
#         logits = (query * key).sum(dim=-1)
#     else:
#         always_show = True
#     cand_pos = torch.LongTensor(cand_positions).view(-1, n_candidates)
#     is_extremal = [logits[p] == logits[cand_pos[i]].max() for i, p in enumerate(tgt_positions)]
#     if always_show or sum(is_extremal[1:]) / len(tgt_positions[1:]) > 0.9:
#         logits[0] = logits[1]
#         plot(logits)
#         _ = plt.xticks(range(len(logits)), tokens)
#         for p, b in zip(tgt_positions, is_extremal): plt.axvline(x=p, color='gray' if b else 'r')
#         plt.show()
#         probs = logits[cand_positions].view(-1, n_candidates).softmax(-1)[cand_is_tgt]
#         print(numpy(probs), '\n', probs.mean())
#         return True
#     return False 


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

# combined_weights = {}

# def get_combined_w(layer, head, qk=False):
#     if (layer, head, qk) in combined_weights: return combined_weights[(layer, head, qk)]
#     wq, wk, wv, wo = get_head_weights(layer, head)
#     w = torch.matmul(wq, wk) if qk else torch.matmul(wv, wo)
#     combined_weights[(layer, head, qk)] = w
#     return w


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