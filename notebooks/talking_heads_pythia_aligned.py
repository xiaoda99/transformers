import transformers
import json
import numpy as np
import time
import torch.nn.functional as F
from transformers.models.gpt_neox.modeling_gpt_neox import *
from transformers.configuration_utils import PretrainedConfig
from typing import Dict

from einops import rearrange, repeat

import torch.jit as jit

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_mask_to_logits(logits, mask):
    #min_value = get_large_negative_number(logits.dtype)
    min_value = (-0.7) * torch.finfo(logits.dtype).max#.to(logits.device)
    #print('mask logits', mask.device, logits.device)
    return torch.where((mask >= min_value * 0.5), logits, min_value)



def apply_rotary_pos_emb_llama(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class GPTNeoXTalkingConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTNeoXModel`]. It is used to instantiate an
    GPTNeoX model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GPTNeoX
    [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the GPTNeoX model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTNeoXModel`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 44):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        rotary_pct (`float`, *optional*, defaults to 0.25):
            percentage of hidden dimensions to allocate to rotary embeddings
        rotary_emb_base (`int`, *optional*, defaults to 10000)
            base for computing rotary embeddings frequency
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio probability of the attention score.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio of (1) the word embeddings, (2) the post-attention hidden states, and (3) the post-mlp
            hidden states.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing token classification, used in the model [`GPTNeoXForTokenClassification`].

            The dropout ratio for the hidden layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 1e-5):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_parallel_residual (`bool`, *optional*, defaults to `True`):
            Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
            speedup at large scales (e.g. 20B).
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.

        Example:

    ```python
    >>> from transformers import GPTNeoXConfig, GPTNeoXModel

    >>> # Initializing a GPTNeoX gpt-neox-20b style configuration
    >>> configuration = GPTNeoXConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-20b style configuration
    >>> model = GPTNeoXModel(configuration)  # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config  # doctest: +SKIP
    ```"""
    model_type = "gpt_neox"

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=6144,
        num_hidden_layers=44,
        num_attention_heads=64,
        intermediate_size=24576,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them!"
            )

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")


class GPTNeoXForCausalLMTalking(GPTNeoXForCausalLM):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super(GPTNeoXForCausalLM, self).__init__(config)

        self.gpt_neox = GPTNeoXModelTalking(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class GPTNeoXModelTalking(GPTNeoXModel):
    def __init__(self, config):
        super(GPTNeoXModel, self).__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNeoXLayerTalking(config, layer_idx=lidx) for lidx, _ in enumerate(range(config.num_hidden_layers))])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

def rms(arr):
    #return ((arr**2).mean() ** 0.5).cpu().item()
    return {'var': (((arr - arr.mean())**2).mean() ** 0.5).cpu().item(), 'rms': ((arr**2).mean() ** 0.5).cpu().item()}

class GPTNeoXLayerTalking(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPTNeoXAttentionTalking(config, layer_idx=layer_idx)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        #print('layer input', self.layer_idx, rms(hidden_states))
        if self.layer_idx ==0: self._layer_input = hidden_states
        normed_hid = self.input_layernorm(hidden_states)
        if self.layer_idx ==0: self._normed_hid = normed_hid
        #print('after attn layernorm', self.layer_idx, rms(normed_hid))
        attention_layer_outputs = self.attention(
            normed_hid,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
            #print('layer attn_output', self.layer_idx, rms(attn_output))
            #print('layer mlp_output', self.layer_idx, rms(mlp_output))
            #print('layer output', self.layer_idx, rms(hidden_states))
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs

class DynamicWeightProjection(nn.Module):

    def __init__(self, layer_idx, num_heads=32, num_groups=1, residual=True, squeeze_ratio=16, query_input_dim=4096, dynamic_squeeze_ratio=16, dynamic_w_hidden_dim=128):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads 
        self.num_groups = num_groups 
        self.query_input_dim = query_input_dim 
        self.dynamic_w_init = True 
        self.dynamic_d_init = True 
        self.dynamic_squeeze_ratio = dynamic_squeeze_ratio# mqy
        self.dynamic_w_hidden_dim = dynamic_w_hidden_dim # mqy
        self.merge_dynamic_w_hidden = False
        self.merge_projection = True
        self.dw_hidden_activation = nn.GELU()
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.dw_activation = nn.Tanh()
        self.dw1_norm = RMSnormNoscale(dim=-1)

        if self.dynamic_w_init is not None:
            dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio 
            if self.dynamic_w_hidden_dim:
                self.dw1 = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, 4, self.dynamic_w_hidden_dim)) #(4096, 1, 4, 128)
                G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
                I = dynamic_hidden_dim * (1 if self.merge_dynamic_w_hidden else 2)
                self.qkw = nn.parameter.Parameter(torch.zeros([G, 4, K, I, M])) # (1, 4, 128, 4, 32)
        if self.dynamic_d_init is not None:
            self.dd = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * 4)) #  (4096, 1, 128)
        
    def forward(self,query_vec,key_vec):  
        def unbind(ary, n, dim=0):
            return [torch.squeeze(a, dim=dim) for a in torch.split(ary, ary.shape[dim] // n, dim=dim)]
        if self.merge_projection:
            pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = None, None, None, None, None, None
            post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = None, None, None, None, None, None
        if self.dynamic_w_init is not None:
            if self.dynamic_w_hidden_dim and not self.merge_dynamic_w_hidden:
                if self.merge_projection:
                    dw_hidden = torch.einsum('BTD,DGCK->BTGCK', query_vec, self.dw1)  # C=4 [pre,post]*[query,key]
                    dw_hidden = self.dw_hidden_activation(dw_hidden)
                    w1, w2 = torch.split(torch.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), self.qkw.shape[-2]//2, dim=-2) #BTGC(2I)M -> [BTGCIM] * 2
                    if hasattr(self, 'dw1_norm'): w1 = self.dw1_norm(w1) # BTGCIM
                    pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, dim=3) # BT4GIM->[BTGIM]*4
                    pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, dim=3)
        if self.dynamic_d_init is not None:
            dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd)
            if hasattr(self, 'dw_activation'): dd = self.dw_activation(dd)
            if not self.merge_projection: qdd, kdd = torch.split(dd,  dd.shape[-1] // 2, dim=-1)
            else: pre_qdd, pre_kdd, post_qdd, post_kdd = torch.split(dd, dd.shape[-1] // 4, dim=-1)
        return (qw1, qw2, kw1, kw2, qdd, kdd) if not self.merge_projection else \
          ((pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd),
          (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd))

class GPTNeoXAttentionTalking(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.window_size = 256 if layer_idx % 2==0 else None # config.window_size if layer_idx % 2==0 else None #256 
        self.query_chunk_size = 128#config.query_chunk_size  # 128
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.transpose_logits = False
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self._init_bias(config.max_position_embeddings)

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        self._init_rope()

        self.q_norm = RMSnorm(hid_dim=config.hidden_size // config.num_attention_heads)
        self.k_norm = RMSnorm(hid_dim=config.hidden_size // config.num_attention_heads)

        self.dyn_w_proj = DynamicWeightProjection(layer_idx, num_heads=self.num_attention_heads, query_input_dim=config.hidden_size, dynamic_squeeze_ratio=self.num_attention_heads//2, dynamic_w_hidden_dim=self.num_attention_heads*4)

        self.pre_proj = CrossHeadProjectionV2(layer_idx, 'pre', num_heads=self.num_attention_heads, squeeze_ratio=None) # mqy
        self.post_proj = CrossHeadProjectionV2(layer_idx, 'post', num_heads=self.num_attention_heads, squeeze_ratio=None) # mqy

        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _init_bias(self, max_positions, device=None):
        if self.window_size is None:
            bias_mask = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions)
        else:
            t = max_positions
            col_idx = torch.tile(torch.arange(t).unsqueeze(0), [t, 1])
            row_idx = torch.tile(torch.arange(t).unsqueeze(1), [1, t])
            bias_mask = (col_idx + self.window_size >= row_idx).tril().view(1, 1, max_positions, max_positions)
        self.register_buffer(
            "bias",
            bias_mask,
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = GPTNeoXRotaryEmbedding(
                self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base
            )
            #self.rotary_emb = LlamaRotaryEmbedding(self.head_size, max_position_embeddings=self.config.max_position_embeddings, base=self.config.rotary_emb_base)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ): #tag
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states) # BTE -> BT(n*3*h)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        #print('qkv', self.layer_idx, rms(query), rms(key), rms(value))
        if self.layer_idx==0: self._query = query
        query, key = self.q_norm(query), self.k_norm(key)
        if self.layer_idx==0: self._normed_query = query

        b, n, t, h = query.shape

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        #print('debug q', query.dtype, cos.dtype, query_rot.dtype)
        #query, key = apply_rotary_pos_emb_llama(query, key, cos, sin, position_ids)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query, key = query.to(dtype=hidden_states.dtype), key.to(dtype=hidden_states.dtype)
        #print('debug q', query.dtype, cos.dtype, query_rot.dtype)

        if self.layer_idx==0: self._roped_query = query
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # scale query
        #r_softplus_0 = 1.442695041
        #scale = r_softplus_0 / (self.head_size ** 0.5)#), dtype=query.dtype, device=query.device)
        #scale *= F.softplus(torch.tensor(0).float().to(query.device))
        query = query * self.head_size ** -0.5 #.088382527 

        #print('qkv_after_rope', self.layer_idx, rms(query), rms(key), rms(value))
        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        pre_proj_dw_args, post_proj_dw_args = self.dyn_w_proj(hidden_states, hidden_states)
        _qw1, _qw2, _kw1, _kw2, _qdd, _kdd = pre_proj_dw_args

        if self.layer_idx==0: self._qw1 = _qw1
        if self.layer_idx==0: self._kw1 = _kw1
        #print('qw1_kw2_kdd_after_rope', self.layer_idx, rms(_qw1), rms(_kw2), rms(_kdd))
        #print('window', self.window_size)
        encoded = torch.zeros((b, n, t, h), dtype=value.dtype, device=query.device)
        #print('attention_mask raw !!!', attention_mask)
        if True or attention_mask is None:
            attention_mask = torch.ones((t, t), dtype=torch.float32, device=query.device).tril() # TODO
            large_negative_number = (-0.7) * torch.finfo(attention_mask.dtype).max 
            attention_mask = (1-attention_mask) * large_negative_number
        
        if self.window_size is not None:  # adapted from limited_context_mask
            #large_negative_number = py_utils.get_large_negative_number(atten_mask.dtype)
            large_negative_number = (-0.7) * torch.finfo(attention_mask.dtype).max 
            col_idx = torch.tile(torch.arange(t).unsqueeze(0), [t, 1])
            row_idx = torch.tile(torch.arange(t).unsqueeze(1), [1, t])
            #col_idx = torch.tile(torch.arange(t)[torch.newaxis, :], [t, 1])
            #row_idx = torch.tile(torch.arange(t)[:, torch.newaxis], [1, t])
            window_mask = (col_idx + self.window_size <= row_idx).to(dtype=attention_mask.dtype, device=query.device) * large_negative_number 
            attention_mask = torch.minimum(attention_mask, window_mask)
        #print('attention mask', rms(attention_mask))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        if self.layer_idx ==0: self._attention_mask = attention_mask 
        #_ = 1 / 0
        #self._attn_mask = attention_mask 
        w = self.query_chunk_size
        #w = 2048

        probs = torch.zeros((b, n, t, key.shape[-2]), dtype=value.dtype, device=query.device)
        for i in range(t // w +1):
            if i * w ==t: continue
            start, stop = i * w, (i + 1) * w
            kv_start = max(0, stop - w - self.window_size) if self.window_size is not None else 0
            _query = query[:, :, start : stop, :]
            _key, _value = key[:, :, kv_start : stop, :], value[:, :, kv_start : stop, :]
            _attention_mask = attention_mask[:, :, start : stop, kv_start : stop]
            def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
                return (qw1[:, start : stop] if qw1 is not None else None,
                    qw2[:, start : stop] if qw2 is not None else None,
                    kw1[:, kv_start : stop] if kw1 is not None else None,
                    kw2[:, kv_start : stop] if kw2 is not None else None,
                    qdd[:, start : stop] if qdd is not None else None,
                    kdd[:, kv_start : stop] if kdd is not None else None)
            _pre_proj_dw_args = slice_dw(*pre_proj_dw_args) if pre_proj_dw_args is not None else ()
            _post_proj_dw_args = slice_dw(*post_proj_dw_args) if post_proj_dw_args is not None else ()
            _encoded, _probs = self._atten_context(i ,_query, _key, _value, _attention_mask,_pre_proj_dw_args, _post_proj_dw_args, query_vec=hidden_states, key_vec=hidden_states)
            encoded[:, :, start : stop, :] = _encoded
            probs[:, :, start : stop, kv_start:stop] = _probs
        encoded = encoded.permute(0, 2, 1, 3).reshape(b,t,n*h)  # bnth->btnh

        #print('encoded_before_o', self.layer_idx, rms(encoded))
        attn_output = encoded
        attn_weights = probs 
        # Compute attention
        #attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        ## Reshape outputs
        #attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        #print('encoded_after_o', self.layer_idx, rms(attn_output))

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _atten_context(self, i, query,key,value,atten_mask,pre_proj_dw_args,post_proj_dw_args,query_vec=None,key_vec=None):
        # logits = self._atten_logits(query, key)
        N = 'N' #if self.num_kv_heads is None else ''
        logits_exp = 'BNTS' if not self.transpose_logits else 'BTSN'
        #if i==15:print('before qk_einsum', rms(query), rms(key))
        #print('qk', query.dtype, key.dtype)
        logits = torch.einsum(f"BNTH,B{N}SH->{logits_exp}", query, key)  # XD BNTH, BNSH -> BNTS
        # logits_exp = 'BTNS' if not self.transpose_logits else 'BTSN'
        # logits = self.qk_einsum(f"BTNH,BS{N}H->{logits_exp}", query, key)  # XD
        #if i==15 and self.layer_idx==0: self._logits = logits

        #if i==15:print('attn logits raw', rms(logits))
        #if True: #self.scale_logits_by_head_dims:
        #logits = torch.multiply(logits, 1.0 / (query.shape[-1] ** 0.5))
        #if self.shared_qk_dim > 0 and self.float32_logits:  # XD
        #    assert not self.scale_logits_by_head_dims
        #    logits = torch.multiply(logits, 1.0 / np.sqrt(self.dim_per_head))

        #if i==15:print('attn logits before preproj', rms(logits))
        logits = self.pre_proj(logits, *pre_proj_dw_args, query_vec=query_vec, key_vec=key_vec)  # XD
        #if i==15:print('attn logits', rms(logits))
        #if i==15 and self.layer_idx ==0:self._logits = logits

        #if self.transpose_logits:
        #    atten_mask = torch.transpose(atten_mask, (0, 2, 3, 1))  # XD: BNTS->BTSN
        #logits = self._cap_logits(logits)
        logits = logits.to(torch.float32)
        padded_logits = apply_mask_to_logits(logits, atten_mask)
        #if i==15 and self.layer_idx==0: self._padded_logits = padded_logits
        #if self.attention_extra_logit is None:
            # XD: -1 -> -2; key -> value: key may have already been turned to fp32 by float32_logits
        probs = F.softmax(padded_logits, dim=-1)#.astype(value.dtype)
        #if i==15 and self.layer_idx==0: self._probs = probs 
        #self._probs = probs
        #else:
        #    probs = torch.exp(self._log_softmax_with_extra_logit(padded_logits))#.astype(value.dtype)
        # XD
        probs = probs.to(dtype=value.dtype)
        #if i==15:print('attn probs before postproj', rms(probs))
        probs_composed = self.post_proj(probs, *post_proj_dw_args, query_vec=query_vec, key_vec=key_vec)
        #if i==15:print('attn probs', rms(probs))
        #if self.float32_probs: probs = probs.astype(value.dtype)
        #if False and getattr(self, 'post_proj', None) is not None:
        #    # mask probs similar to py_utils.apply_mask_to_logits
        #    min_value = py_utils.get_large_negative_number(probs.dtype)
        #    probs = torch.where((atten_mask >= min_value * 0.5), probs, 0.)
        #probs = self.atten_dropout(probs)
        #if self.transpose_logits: probs = torch.transpose(probs, (0, 3, 1, 2)) # XD: BTSN -> BNTS
        #N = 'N' #if self.num_kv_heads is None else ''
        encoded = torch.einsum(f'BNTS,BNSH->BNTH', probs_composed, value)
        # encoded = self.pv_einsum(f'BTNS,BS{N}H->BTNH', probs, value)
        return encoded, probs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs 

class RMSnorm(nn.Module):
    
    def __init__(self, hid_dim=128, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.scale = nn.parameter.Parameter(data=torch.ones(self.hid_dim))

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs 

class CrossHeadProjectionV2(nn.Module):

    def __init__(self, layer_idx, mode, num_heads=16, num_groups=1, squeeze_ratio=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.mode = mode
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.squeeze_ratio = squeeze_ratio
        self.use_static_w = True
        if self.squeeze_ratio is None:
            self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group))
         
    def forward(self, inputs, qw1, qw2, kw1, kw2, qdd, kdd, query_vec=None, key_vec=None): # v2 raw
        #if self.layer_idx==16 and self.mode == 'pre':
        #    return inputs
        shape = inputs.shape
        inputs = rearrange(inputs, 'B (G M) T S -> B G M T S', G=self.num_groups)
        inputs_label = 'BGMTS'
        ret = 0#inputs 
        if self.use_static_w:
            if self.squeeze_ratio is None:
                w = self.w + torch.eye(self.num_heads_per_group, device=self.w.device, dtype=self.w.dtype)
                #print('dtype', inputs.dtype, w.dtype)
                ret = torch.einsum('BGMTS,GMN->BGNTS', inputs, w)
        #if idx ==15: print('after sw', self.mode, rms(ret))
        if qw1 is not None:
            #if idx ==15: print('apply qw inputs',  self.mode, rms(inputs))
            #if idx ==15: print('apply qw qw1',  self.mode, rms(qw1))
            #if idx ==15: print('apply qw qw2',  self.mode, rms(qw2))
            #if idx ==15: print('apply qw kw1',  self.mode, rms(kw1))
            #if idx ==15: print('apply qw kw2',  self.mode, rms(kw2))
            hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I') # BGITS
            for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]): # tag
                dw_label = f'B{sym}G{hidden_sym}M'  # w1: BTGIM, dw_label:BTGIM
                dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
                eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'
                eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'
                #print('eqn1', eqn1)
                #print('eqn2', eqn2)
                #hidden = torch.einsum(eqn1, inputs, w1) # BGMTS,BTG(I)M->BGTS
                #out = torch.einsum(eqn2, hidden, w2) #  'BG(I)TS,BTG(I)M->BGMTS'
                #ret = ret + out
                #if idx ==15: print(f'apply qw {sym}',  self.mode, rms(ret))
                for i in range(dynamic_hidden_dim):
                    hidden = torch.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :]) # BGMTS,BTG(I)M->BGTS
                    out = torch.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :]) #  'BG(I)TS,BTG(I)M->BGMTS'
                    ret = ret + out
                #    if idx ==15: print(f'apply qw {sym}{i}',  self.mode, rms(ret))
        #if idx ==15: print('after qw',  self.mode,rms(ret))
        if qdd is not None:
            for sym, dd in zip(['T', 'S'], [qdd, kdd]):
                dd_label = f'B{sym}GM'
                dout = torch.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd) # BGMTS,B(T/S)GM->BGMTS
                ret = ret + dout
        #if idx ==15: print('after qdd',  self.mode,rms(ret))
        return torch.reshape(ret, shape)  # BGMTS->BMTS

# pythia-pax-talking
#state.mdl_vars.params.lm.embedding_lookup.emb_var (50304, 4096)
#state.mdl_vars.params.lm.final_ln.bias (4096,)
#state.mdl_vars.params.lm.final_ln.scale (4096,)
#state.mdl_vars.params.lm.softmax.logits_ffn.linear.w (4096, 50257)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b (16, 16384)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w (16, 4096, 16384)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w (16, 16384, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.dyn_w_proj.dd (16, 4096, 1, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.dyn_w_proj.dw1 (16, 4096, 1, 4, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.dyn_w_proj.qkw (16, 1, 4, 128, 4, 32)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.k_norm.scale (16, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.b (16, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w (16, 4096, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w (16, 4096, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post_proj.w (16, 1, 32, 32)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.pre_proj.w (16, 1, 32, 32)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.q_norm.scale (16, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.query.b (16, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.query.w (16, 4096, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.value.b (16, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.value.w (16, 4096, 32, 128)
######################################################################################################
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.ff_layer.ffn_layer1.bias.b (16, 16384)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.ff_layer.ffn_layer1.linear.w (16, 4096, 16384)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.ff_layer.ffn_layer2.bias.b (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.ff_layer.ffn_layer2.linear.w (16, 16384, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.ff_layer.layer_norm.bias (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.ff_layer.layer_norm.scale (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.layer_norm.bias (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.layer_norm.scale (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.dyn_w_proj.dd (16, 4096, 1, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.dyn_w_proj.dw1 (16, 4096, 1, 4, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.dyn_w_proj.qkw (16, 1, 4, 128, 4, 32)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.k_norm.scale (16, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.key.b (16, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.key.w (16, 4096, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.post.b (16, 4096)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.post.w (16, 4096, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.post_proj.w (16, 1, 32, 32)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.pre_proj.w (16, 1, 32, 32)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.q_norm.scale (16, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.query.b (16, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.query.w (16, 4096, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.value.b (16, 32, 128)
#state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_1.self_attention.value.w (16, 4096, 32, 128)

#pytorch pythia talking
#gpt_neox.embed_in.weight torch.Size([50432, 4096]) torch.float32
#gpt_neox.final_layer_norm.weight torch.Size([4096]) torch.float32
#gpt_neox.final_layer_norm.bias torch.Size([4096]) torch.float32
#embed_out.weight torch.Size([50432, 4096]) torch.float32
#gpt_neox.layers.0.input_layernorm.weight torch.Size([4096]) torch.float32
#gpt_neox.layers.0.input_layernorm.bias torch.Size([4096]) torch.float32
#gpt_neox.layers.0.post_attention_layernorm.weight torch.Size([4096]) torch.float32
#gpt_neox.layers.0.post_attention_layernorm.bias torch.Size([4096]) torch.float32
#gpt_neox.layers.0.attention.dyn_w_proj.dw1 torch.Size([4096, 1, 4, 128]) torch.float32
#gpt_neox.layers.0.attention.dyn_w_proj.qkw torch.Size([1, 4, 128, 4, 32]) torch.float32
#gpt_neox.layers.0.attention.dyn_w_proj.dd torch.Size([4096, 1, 128]) torch.float32
#gpt_neox.layers.0.attention.pre_proj.w torch.Size([1, 32, 32]) torch.float32
#gpt_neox.layers.0.attention.post_proj.w torch.Size([1, 32, 32]) torch.float32
#gpt_neox.layers.0.attention.query_key_value.weight torch.Size([12288, 4096]) torch.float32
#gpt_neox.layers.0.attention.query_key_value.bias torch.Size([12288]) torch.float32
#gpt_neox.layers.0.attention.dense.weight torch.Size([4096, 4096]) torch.float32
#gpt_neox.layers.0.attention.dense.bias torch.Size([4096]) torch.float32
#gpt_neox.layers.0.mlp.dense_h_to_4h.weight torch.Size([16384, 4096]) torch.float32
#gpt_neox.layers.0.mlp.dense_h_to_4h.bias torch.Size([16384]) torch.float32
#gpt_neox.layers.0.mlp.dense_4h_to_h.weight torch.Size([4096, 16384]) torch.float32
#gpt_neox.layers.0.mlp.dense_4h_to_h.bias torch.Size([4096]) torch.float32

def match_weight_pythia(model, w):
    map_dict={'dense_h_to_4h': 'ffn_layer1', 'dense_4h_to_h': 'ffn_layer2', 'weight': 'linear.w', 'bias':'bias.b'} 
    wb_dict = {'weight':'w', 'bias': 'b'}
    _, E, H, D = w['state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w'].shape # (16, 4096, 32, 128)
    N = w['state.mdl_vars.params.lm.embedding_lookup.emb_var'].shape[0] #50304
    state_dict = {}
    for k, v in model.named_parameters():
        if k == 'gpt_neox.embed_in.weight':
            v = w['state.mdl_vars.params.lm.embedding_lookup.emb_var']#[:50257,:] # TODO
        elif k == 'gpt_neox.final_layer_norm.weight':
            v = w[f'state.mdl_vars.params.lm.final_ln.scale']
        elif k == 'gpt_neox.final_layer_norm.bias':
            v = w[f'state.mdl_vars.params.lm.final_ln.bias']
        elif k == 'embed_out.weight':
            v = torch.tensor(w['state.mdl_vars.params.lm.softmax.logits_ffn.linear.w'].T)#[:50257])  # E,N -> N,E
        else:
            layer = int(k.split('.')[2])
            sub_layer, _layer = layer % 2, layer //2 # sub_layer 0/1, _layer 0-15
            if '.attention.' in k:
                _, _, _, _, ptype, wtype = k.split('.')
                if ptype in ['pre_proj', 'post_proj', 'dyn_w_proj']: # pre post proj
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][_layer]
                elif ptype in ['q_norm', 'k_norm']:
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][_layer]
                elif ptype == 'query_key_value':
                    w_or_bias = k.split('.')[-1]
                    w_or_bias = wb_dict[w_or_bias]
                    _q = torch.tensor(w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.query.{w_or_bias}'][_layer])
                    _k = torch.tensor(w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.key.{w_or_bias}'][_layer])
                    _v = torch.tensor(w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.value.{w_or_bias}'][_layer])

                    if w_or_bias == 'w':
                        #_q, _k, _v = _q.reshape(E,H*D), _k.reshape(E,H*D), _v.reshape(E,H*D)
                        v = torch.stack([_q, _k, _v]).permute(1,2,0,3).contiguous().reshape(E,3*E).T # 3EHD-> EH3D -> E(H3D) -> (H3D)E
                        #v = torch.stack([_q, _k, _v]).permute(1,0,2,3).contiguous().reshape(E,3*E).T # 3EHD-> EH3D
                        #v = torch.stack([_q, _k, _v]).permute(1,0,2,3).contiguous().reshape(E*3,E) # 3EHD-> EH3D
                        #v = torch.concatenate([_q, _k, _v],dim=0)#.contiguous().reshape(-1,_q.shape[-1])
                    elif w_or_bias == 'b':
                        #_q, _k, _v = _q.reshape(H*D), _k.reshape(H*D), _v.reshape(H*D)
                        v = torch.stack([_q, _k, _v]).permute(1,0,2).contiguous().reshape(-1) # 3HD ->H3D -> (H3D)
                        #v = torch.concatenate([_q, _k, _v], dim=0)#.contiguous().reshape(-1)
                else: # dense
                    w_or_bias = k.split('.')[-1]
                    w_or_bias = wb_dict[w_or_bias]
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.post.{w_or_bias}'][_layer]
                    if w_or_bias == 'w':
                        v = v.reshape(E,H*D)#.T
                    elif w_or_bias == 'b':
                        v = v.reshape(H*D)
                #print(ptype, wtype, v.max(), v.min(), v.var())
            elif 'mlp' in k:
                ptype = k.split('.')[4] # dense_h_to_4h, dense_4h_to_h
                w_or_bias = k.split('.')[-1]
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.ff_layer.{map_dict[ptype]}.{map_dict[w_or_bias]}'][_layer]
                if w_or_bias == 'weight': v = v.T
            elif 'post_attention_layernorm' in k: # mlp layernorm
                w_or_bias = k.split('.')[-1]
                if w_or_bias == 'weight': w_or_bias = 'scale'
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.ff_layer.layer_norm.{w_or_bias}'][_layer]
            elif 'input_layernorm' in k: # attention layernorm
                w_or_bias = k.split('.')[-1]
                if w_or_bias == 'weight': w_or_bias = 'scale'
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.layer_norm.{w_or_bias}'][_layer]
            #print(k, v.shape)
        if k.endswith('norm.weight'): v = v+1 # fix layernorm
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=False)
    return model


def match_weight(model, w):
    map_dict={'q_proj':'query', 'k_proj':'key', 'v_proj':'value','o_proj':'post', 'gate_proj': 'ffn_layer1_gate', 'up_proj': 'ffn_layer1', 'down_proj': 'ffn_layer2', 
              'weight': 'w'} # 'pre_proj': 'pre_proj', 'post_proj': 'post_proj'
    L, E, H, D = w['state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w'].shape
    N = w['state.mdl_vars.params.lm.embedding_lookup.emb_var'].shape[0]
    state_dict = {}
    for k, v in model.named_parameters():
        if k == 'model.embed_tokens.weight':
            v = w['state.mdl_vars.params.lm.embedding_lookup.emb_var'][:50257,:]
        elif k == 'model.norm.weight':
            v = w['state.mdl_vars.params.lm.final_ln.scale']
        elif k == 'lm_head.weight':
            v = w['state.mdl_vars.params.lm.softmax.logits_ffn.linear.w'].T  # E,N -> N,E
            #v = torch.zeros_like(v)
            #_v = w['state.mdl_vars.params.lm.softmax.logits_ffn.linear.w'].T  # E,N -> N,E
            #v[:_v.shape[0],:] = torch.tensor(_v) # pad unembedding matrix as 0
        else:
            layer = int(k.split('.')[2])
            if 'self_attn' in k:
                if k.endswith('_m'):continue # merged proj weights
                _, _, _, _, ptype, wtype = k.split('.')
                if k.endswith('_p'): continue # ablation parameters
                if ptype in ['pre_proj', 'post_proj']: # pre post proj
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][layer]
                else: # qkov
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][layer].reshape(E,H*D).T
                    if ptype == 'o_proj': v = v.T
                #print(ptype, wtype, v.max(), v.min(), v.var())
            elif 'mlp' in k: 
                ptype = k.split('.')[4] # gate, up, down proj
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.{map_dict[ptype]}.linear.w'][layer].T
            elif 'post_attention_layernorm' in k: # mlp layernorm
                v = w['state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale'][layer]
            elif 'input_layernorm' in k: # attention layernorm
                v = w['state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale'][layer]
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=False)
    return model

model_paths={
    'Llama_small': 'Llama_raw_checkpoint_00057300.torch.bin', # 0.3B
    'Llama_middle': 'C4SpmdLlamaXLHead16x128_checkpoint_00070600.torch.bin', # 1B
    'Llama_large': 'C4SpmdLlamaXXLv4_checkpoint_00073700.torch.bin', # 2B
    'LlamaTalking_small': 'LlamaTalking_checkpoint_00060000.torch.bin', # 0.3B
    #'LlamaTalking_middle': 'C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormv4_nocap_checkpoint_00050000.torch.bin', # 1B
    'LlamaTalking_middle': 'C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormv4_nocap_checkpoint_00066300.torch.bin', # 1B
    }

def load_model(model='Llama', size='small', match_model=True):
    DATA_DIR = '/home/mengqy/Data'
    model_key = f'{model}_{size}'
    if model == 'Llama':
        with open(f'config_{size}.json', 'r') as f: config = json.loads(f.read())
        config = LlamaConfig(**config)
        model = LlamaForCausalLM(config)
    elif model == 'LlamaTalking':
        with open(f'config_{size}.json', 'r') as f: config = json.loads(f.read())
        config = LlamaConfigTalking(**config)
        model = LlamaForCausalLMTalking(config)
    elif model == 'pythia':
        with open(f'config_pythia_6p9b.json', 'r') as f: config = json.loads(f.read())
        config = GPTNeoXTalkingConfig(**config)
        model = GPTNeoXForCausalLMTalking(config)
    if match_model:
        weight_pax = torch.load(f'{DATA_DIR}/models/{model_paths[model_key]}')
        model = match_weight(model, weight_pax)
    return model

if __name__ == '__main__':
    with open('config_pythia_6p9b.json', 'r') as f:
        config = json.loads(f.read())
    config = GPTNeoXTalkingConfig(**config)
    print(config)
    model = GPTNeoXForCausalLMTalking(config)
    print(model)
