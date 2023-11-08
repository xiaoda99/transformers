import transformers
import json
from transformers.models.llama.modeling_llama import *
from transformers.configuration_utils import PretrainedConfig

from einops import rearrange, repeat

Global_cnt = 0
Global_cnt_thres = -1

class LlamaConfigTalking(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
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
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        talking_version='v1',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.talking_version=talking_version

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

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


class LlamaForCausalLMTalking(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelTalking(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class LlamaModelTalking(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayerTalking(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

class LlamaDecoderLayerTalking(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx=None):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttentionTalking(config=config, layer_idx=layer_idx)
        #self.mlp = LlamaMLP(config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        global Global_cnt, Global_cnt_thres 
        #print('Global_cnt', Global_cnt)
        if Global_cnt <=Global_cnt_thres : self.hidden_states = hidden_states; Global_cnt += 1 

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        if Global_cnt <=Global_cnt_thres : self.attn_hidden_states = hidden_states; Global_cnt += 1 
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)


        if Global_cnt <=Global_cnt_thres : self.out_hidden_states = hidden_states; Global_cnt += 1 

        return outputs




class LlamaAttentionTalking(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        if config.talking_version == 'v1':
            self.pre_proj = CrossHeadProjection(layer_idx, 'pre') # mqy
            self.post_proj = CrossHeadProjection(layer_idx, 'post') # mqy
        elif config.talking_version == 'v2':
            self.pre_proj = CrossHeadProjectionV2(layer_idx, 'pre', squeeze_ratio=2, query_input_dim=config.hidden_size, dynamic_squeeze_ratio=8, dynamic_w_hidden_dim=64) # mqy
            self.post_proj = CrossHeadProjectionV2(layer_idx, 'post', squeeze_ratio=None, query_input_dim=config.hidden_size, dynamic_squeeze_ratio=8, dynamic_w_hidden_dim=64) # mqy

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        global Global_cnt, Global_cnt_thres
        if Global_cnt <=Global_cnt_thres : self.attn_weights_before_pre_prej = attn_weights ; Global_cnt += 1 
        attn_weights = self.pre_proj(attn_weights, query_vec=hidden_states, key_vec=hidden_states) # mqy
        if Global_cnt <=Global_cnt_thres : self.attn_weights_after_pre_prej = attn_weights ; Global_cnt += 1 

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        #if Global_cnt <=Global_cnt_thres : self.attn_weights_after_pre_prej = attn_weights ; Global_cnt += 1 
        attn_weights = self.post_proj(attn_weights, query_vec=hidden_states, key_vec=hidden_states) # mqy
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class CrossHeadProjection(nn.Module):

    def __init__(self, layer_idx, mode, num_heads=16, num_groups=1, residual=True, squeeze_ratio=1, hidden_dim=1, query_input_dim=1024):
        super().__init__()
        self.layer_idx = layer_idx
        self.mode = mode
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.hidden_dim = hidden_dim 
        self.residual = residual 
        self.absorb_residual = False
        #self.init: WeightInit = None
        self.transpose = True 
        self.left_mul = False  # no effect
        #self.use_conv = False
        self.squeeze_ratio = squeeze_ratio
        self.use_squeeze_bias = True
        #self.#squeeze_activation_cls: activations_lib.BaseActivation = activations_lib.Identity
        #self.#output_activation_cls: activations_lib.BaseActivation = None # activations_lib.Identity
        self.learnable_diag = True 
        #self.relative_scale = 0.1
        #self.gate_relative_scale = 0.01
        #self.skip_ffn_weight_decay = False
        #self.#dynamic_squeeze_gate_act_cls: activations_lib.BaseActivation = None
        #self.addictive_gate = False
        self.query_input_dim = query_input_dim 
        #self.key_input_dim = None
        self.use_static_w = True
        self.dynamic_w_init = True 
        self.dynamic_squeeze_ratio = None
        self.use_dw_bias = False
        #self.dw_activation = False 
        self.dynamic_w_hidden_dim = None
        #self.#dw_hidden_activation_cls: activations_lib.BaseActivation = None
        self.tgt_dependent = True
        self.src_dependent = True
        #self.ablations = {} # for model analysis

        if self.squeeze_ratio is None:
            self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group))
        else:
            self.w1 = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.hidden_dim))
            self.w2 = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.hidden_dim, self.num_heads_per_group))
            #self.b = nn.parameter.Parameter(data=torch.zeros(self.hidden_dim))
         
        #self.wg = nn.parameter.Parameter(data=torch.zeros(self.query_input_dim, self.num_groups, self.hidden_dim))
        #self.wg2 = nn.parameter.Parameter(data=torch.zeros(self.key_input_dim, self.num_groups, self.hidden_dim))
        #self.bg = nn.parameter.Parameter(data=torch.zeros(self.hidden_dim))

        dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio \
          if self.dynamic_squeeze_ratio is not None else 1
        if self.dynamic_w_hidden_dim is not None:
            assert self.dynamic_w_init is not None
            #self.dw1 = nn.parameter.Parameter(data=torch.zeros(self.query_input_dim, self.num_groups, self.dynamic_w_hidden_dim * 2))
            #self.dwhb = nn.parameter.Parameter(data=torch.zeros(self.dynamic_w_hidden_dim))
            #self.dw_hidden_activation = nn.Tanh()
            #self.activation = nn.Tanh()
            #self.dw2_w1 = nn.parameter.Parameter(data=torch.zeros())
            #self.dw2_w2 = nn.parameter.Parameter(data=torch.zeros())
            #self.dw2_d = nn.parameter.Parameter(data=torch.zeros())
            #for w_name in ['w1', 'w2', 'd']:
            #    G, K, M, I = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group, dynamic_hidden_dim
            #    shape=[G, K, M, I] if w_name != 'd' else [G, K, M]
            #    bias_shape = [G, M, I]
            #    self.set_attr(f'dw2_{w_name}',  nn.parameter.Parameter(data=torch.zeros(shape)))
            #    if self.use_dw_bias and w_name != 'd':
            #        self.set_attr(f'dwb_{w_name}',  nn.parameter.Parameter(data=torch.zeros(bias_shape)))
        elif self.dynamic_w_init is not None:
            out_shape = [self.num_groups, self.num_heads_per_group, dynamic_hidden_dim * 4] # GM(4I)
            self.dw = nn.parameter.Parameter(data=torch.zeros([self.query_input_dim] + out_shape))
            if self.use_dw_bias:
                self.dwb = nn.parameter.Parameter(data=torch.zeros(out_shape))
            if self.learnable_diag:
                self.dd = nn.parameter.Parameter(data=torch.zeros([self.query_input_dim, self.num_groups, self.num_heads_per_group * 2]))
        p_shape = 1
        #p_shape = num_heads
        ablations={
                'static_w_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_src_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_tgt_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_dd_src_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_dd_tgt_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
}
        for k, v in ablations.items():
            setattr(self, k, v)
        #self.d = nn.parameter.Parameter(data=torch.zeros([self.num_groups, self.num_heads_per_group]))
        #self.dw_activation = nn.Tanh()
        #self.output_activation = nn.Tanh()

    def forward(self, inputs, query_vec=None, key_vec=None):
        shape = inputs.shape
        # inputs = self._cast_to_fprop_dtype(inputs)  # preserve float32 logits
        # assert shape[1] == self.num_heads
        #if self.use_conv:  # an order of magnitude slower than einsum!
        #  ret = lax.conv(inputs, self.w, (1, 1), 'SAME') # BNTS,MN11->BMTS
        #  if self.residual: ret = ret + inputs
        #  return torch.reshape(ret, shape)
        if self.transpose:
            # inputs = torch.reshape(inputs, shape[:3] + (self.num_groups, self.num_heads_per_group))
            inputs = rearrange(inputs, 'B (G M) T S -> B T S G M', G=self.num_groups)
            exp = 'BTSGM,GMN->BTSGN'; exp_gate = 'BTSGI,BTGI->BTSGI'; exp2 = 'BTSGM,GM->BTSGM'
            if hasattr(self, 'b'): b = self.b
        else:
            # inputs = torch.reshape(inputs, (shape[0], self.num_groups, self.num_heads_per_group) + shape[2:])
            inputs = rearrange(inputs, 'B (G M) T S -> B G M T S', G=self.num_groups)
            exp = 'BGMTS,GMN->BGNTS' if not self.left_mul else 'GNM,BGMTS->BGNTS'
            exp_gate = 'BGITS,BTGI->BGITS'; exp2 = 'BGMTS,GM->BGMTS'  # for ffn, N=I=hidden_dim
            if hasattr(self, 'b'): b = torch.expand_dims(self.b, dim=(1, 2))
        ret = 0.
        if self.use_static_w:
            if self.squeeze_ratio is None:
                pass
                #w = self.w + torch.eye(self.num_heads_per_group) \
                #  if self.residual and self.absorb_residual else self.w
                #ret = torch.einsum(exp, inputs, w) if not self.left_mul else torch.einsum(exp, w, inputs)
            else:
                ret = torch.einsum(exp, inputs, self.w1) if not self.left_mul else torch.einsum(exp, self.w1, inputs)
                if hasattr(self, 'b'): ret = ret + b
                #ret = self.activation(ret)
                #if self.dynamic_squeeze_gate_act_cls is not None:
                #    if not self.addictive_gate:
                #      gate_value = torch.einsum('BTD,DGI->BTGI', query_vec, self.wg) + self.bg
                #      ret = torch.einsum(exp_gate, ret, self.gate_activation(gate_value))
                #    else:
                #      gate_value = torch.einsum('BTD,DGI->BTGI', query_vec, self.wg)
                #      gate_value = rearrange(gate_value, 'B T G I -> B T 1 G I')
                #      ret = ret + gate_value
                #      gate_value2 = torch.einsum('BSD,DGI->BSGI', key_vec, self.wg2)
                #      gate_value2 = rearrange(gate_value2, 'B S G I -> B 1 S G I')
                #      ret = ret + gate_value2
                static_w_vec = torch.tensor(self.ablations.get('static_w', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.static_w_p
                ret = torch.einsum(exp, ret, self.w2 * static_w_vec) if not self.left_mul else torch.einsum(exp, self.w2 * static_w_vec, ret)
                #print('static w', ret.shape, (self.w1[0] * self.w2[0]).shape)
                #ret *= self.ablations.get('static_w', 1)
          
        if self.dynamic_w_hidden_dim is not None:
            pass
            #dw_hidden = torch.einsum('BTD,DGK->BTGK', query_vec, self.dw1)
            #q_hidden, k_hidden = torch.split(dw_hidden, 2, dim=-1)
            #dw_hidden = rearrange(q_hidden, 'B T G K -> B T 1 G K') + \
            #  rearrange(k_hidden, 'B S G K -> B 1 S G K')  # BTSGK
            #dw_hidden = self.dw_hidden_activation(dw_hidden + self.dwhb)

            #w1 = torch.einsum('BTSGK,GKMI->BTSGMI', dw_hidden, self.dw2_w1)
            #if self.use_dw_bias: w1 = w1 + self.dwb_w1
            #if self.dw_activation_cls is not None: w1 = self.dw_activation(w1)
            #hidden = torch.einsum('BTSGM,BTSGMI->BTSGI', inputs, w1)
            #w2 = torch.einsum('BTSGK,GKMI->BTSGIM', dw_hidden, self.dw2_w2)
            #if self.use_dw_bias: w2 = w2 + self.dwb_w2
            #if self.dw_activation_cls is not None: w2 = self.dw_activation(w2)
            #ret = ret + torch.einsum('BTSGI,BTSGIM->BTSGM', hidden, w2)

            #if self.learnable_diag:
            #    d = torch.einsum('BTSGK,GKM->BTSGM', dw_hidden, self.dw2_d)
            #    if self.dw_activation_cls is not None: d = self.dw_activation(d)
            #    ret = ret + inputs * d # torch.einsum('BTSGM,BTSGM->BTSGM', inputs, d)

        elif self.dynamic_w_init is not None:
            #dynamic_w_vec = torch.tensor(self.ablations.get('dynamic_w', 1)).to(inputs.device)
            #dynamic_w_dd_vec = torch.tensor(self.ablations.get('dynamic_w_dd', 1)).to(inputs.device)
            dynamic_w_vec_src = torch.tensor(self.ablations.get('dynamic_w_src', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_src_p
            dynamic_w_vec_tgt = torch.tensor(self.ablations.get('dynamic_w_tgt', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_tgt_p
            dynamic_w_dd_vec_src = torch.tensor(self.ablations.get('dynamic_w_dd_src', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_dd_src_p
            dynamic_w_dd_vec_tgt = torch.tensor(self.ablations.get('dynamic_w_dd_tgt', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_dd_tgt_p
            dw = torch.einsum('BTD,DGMI->BTGMI', query_vec, self.dw)
            if self.use_dw_bias: dw = dw + self.dwb  # BTGM(4I)+GM(4I)=BTGM(4I)
            if hasattr(self, 'dw_activation'): dw = self.dw_activation(dw)
            #print('dw', dw.shape, type(arrs), arrs)
            qw1, qw2, kw1, kw2 = torch.split(dw, dw.shape[-1] // 4, dim=-1) # split difference between torch and jax
            qw2, kw2 = rearrange(qw2, 'B T G M I -> B T G I M'), rearrange(kw2, 'B T G M I -> B T G I M')
            if self.tgt_dependent:
                #print('tgt_dependent', inputs.shape, qw1.shape)
                hidden = torch.einsum('BTSGM,BTGMI->BTSGI', inputs, qw1)
                ret = ret + torch.einsum('BTSGI,BTGIM->BTSGM', hidden, qw2 * dynamic_w_vec_tgt)
            if self.src_dependent:
                hidden = torch.einsum('BTSGM,BSGMI->BTSGI', inputs, kw1)
                ret = ret + torch.einsum('BTSGI,BSGIM->BTSGM', hidden, kw2* dynamic_w_vec_src) 
            if self.learnable_diag:
                dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd)
                if hasattr(self, 'dw_activation'): dd = self.dw_activation(dd)
                qdd, kdd = torch.split(dd, dd.shape[-1] // 2, dim=-1)
                if self.tgt_dependent or not self.tgt_dependent and not self.src_dependent:
                    ret = ret + torch.einsum('BTSGM,BTGM->BTSGM', inputs, qdd* dynamic_w_dd_vec_tgt) 
                if self.src_dependent or not self.tgt_dependent and not self.src_dependent:
                    ret = ret + torch.einsum('BTSGM,BSGM->BTSGM', inputs, kdd* dynamic_w_dd_vec_src)
            #print('dynamic w', ret.shape, )
            self._dynamic_w = [qw1, qw2, kw1, kw2, qdd, kdd]

        # ret = self.output_activation(ret)  # for post_proj, relu here degrade performance to baseline
        if self.use_static_w and self.residual and not self.absorb_residual:
            ret = ret + (torch.einsum(exp2, inputs, self.d) \
            if self.learnable_diag and self.dynamic_w_init is None else inputs)
        if hasattr(self, 'output_activation'):
            ret = self.output_activation(ret)  # for post_proj, relu here has no effect on performance
        if self.transpose: ret = ret.permute((0, 3, 4, 1, 2))  # BTSGM->BGMTS
        # inputs = inputs + torch.repeat(inputs[:, :, 0:1, :, :], self.num_heads_per_group, dim=2)  # torch.roll(inputs, 1, dim=2) #
        return torch.reshape(ret, shape)  # BGMTS->BNTS


class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-2):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = torch.mean(torch.square(inputs), dim=self.dim, keepdims=True, dtype=torch.float32)
        normed_inputs = inputs / torch.sqrt(var + self.epsilon *2)
        return normed_inputs 


class CrossHeadProjectionV2(nn.Module):

    def __init__(self, layer_idx, mode, num_heads=16, num_groups=1, residual=True, squeeze_ratio=16, query_input_dim=1024, dynamic_squeeze_ratio=None, dynamic_w_hidden_dim=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.mode = mode
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.residual = residual 
        self.absorb_residual = False
        #self.init: WeightInit = None
        self.transpose = True 
        self.left_mul = False  # no effect
        #self.use_conv = False
        self.squeeze_ratio = squeeze_ratio
        self.use_squeeze_bias = True
        #self.#squeeze_activation_cls: activations_lib.BaseActivation = activations_lib.Identity
        #self.#output_activation_cls: activations_lib.BaseActivation = None # activations_lib.Identity
        self.learnable_diag = True 
        #self.relative_scale = 0.1
        #self.gate_relative_scale = 0.01
        #self.skip_ffn_weight_decay = False
        #self.#dynamic_squeeze_gate_act_cls: activations_lib.BaseActivation = None
        #self.addictive_gate = False
        self.query_input_dim = query_input_dim 
        #self.key_input_dim = None
        self.use_static_w = True
        self.dynamic_w_init = True 
        self.dynamic_squeeze_ratio = dynamic_squeeze_ratio 
        self.use_dw_bias = False
        #self.dw_activation = False 
        #if dynamic_w_hidden_dim is None: dynamic_w_hidden_dim = self.num_heads // self.num_groups
        self.dynamic_w_hidden_dim = dynamic_w_hidden_dim #None
        #self.#dw_hidden_activation_cls: activations_lib.BaseActivation = None
        self.tgt_dependent = True
        self.src_dependent = True
        self.use_dw_hidden_bias = False
        self.dw_activation = nn.Tanh() # v2
        self.activation = nn.GELU()
        #self.ablations = {} # for model analysis

        if self.squeeze_ratio is None:
            self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group))
        else:
            self.hidden_dim = self.num_heads_per_group // squeeze_ratio
            self.w1 = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.hidden_dim))
            self.w2 = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.hidden_dim, self.num_heads_per_group))
            #self.b = nn.parameter.Parameter(data=torch.zeros(self.hidden_dim))
         
        #self.wg = nn.parameter.Parameter(data=torch.zeros(self.query_input_dim, self.num_groups, self.hidden_dim))
        #self.wg2 = nn.parameter.Parameter(data=torch.zeros(self.key_input_dim, self.num_groups, self.hidden_dim))
        #self.bg = nn.parameter.Parameter(data=torch.zeros(self.hidden_dim))

        dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio \
          if self.dynamic_squeeze_ratio is not None else 1
        if self.dynamic_w_hidden_dim is not None:
            assert self.dynamic_w_init is not None
            self.dw1 = nn.parameter.Parameter(data=torch.zeros(self.query_input_dim, self.num_groups, self.dynamic_w_hidden_dim * 2))
            G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
            I = dynamic_hidden_dim * 2
            shape = [G, K, M, I]
            self.qw = nn.parameter.Parameter(data=torch.zeros(shape))
            self.kw = nn.parameter.Parameter(data=torch.zeros(shape))
            self.dw_hidden_activation = nn.GELU()
            self.dw1_norm = RMSnormNoscale()
            #self.dwhb = nn.parameter.Parameter(data=torch.zeros(self.dynamic_w_hidden_dim))
            #self.activation = nn.Tanh()
            #self.dw2_w1 = nn.parameter.Parameter(data=torch.zeros())
            #self.dw2_w2 = nn.parameter.Parameter(data=torch.zeros())
            #self.dw2_d = nn.parameter.Parameter(data=torch.zeros())
            #for w_name in ['w1', 'w2', 'd']:
            #    G, K, M, I = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group, dynamic_hidden_dim
            #    shape=[G, K, M, I] if w_name != 'd' else [G, K, M]
            #    bias_shape = [G, M, I]
            #    self.set_attr(f'dw2_{w_name}',  nn.parameter.Parameter(data=torch.zeros(shape)))
            #    if self.use_dw_bias and w_name != 'd':
            #        self.set_attr(f'dwb_{w_name}',  nn.parameter.Parameter(data=torch.zeros(bias_shape)))
        if self.dynamic_w_init is not None:
            if not hasattr(self, 'dw1'):
                out_shape = [self.num_groups, self.num_heads_per_group, dynamic_hidden_dim * 4] # GM(4I)
                self.dw = nn.parameter.Parameter(data=torch.zeros([self.query_input_dim] + out_shape))
                if self.use_dw_bias:
                    self.dwb = nn.parameter.Parameter(data=torch.zeros(out_shape))
            if self.learnable_diag:
                self.dd = nn.parameter.Parameter(data=torch.zeros([self.query_input_dim, self.num_groups, self.num_heads_per_group * 2]))
        p_shape = 1
        #p_shape = num_heads
        ablations={
                'static_w_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_src_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_tgt_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_dd_src_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
                'dynamic_w_dd_tgt_p': nn.parameter.Parameter(data=torch.ones(p_shape)),
}
        for k, v in ablations.items():
            setattr(self, k, v)
        #self.d = nn.parameter.Parameter(data=torch.zeros([self.num_groups, self.num_heads_per_group]))
        #self.output_activation = nn.Tanh()

    def forward(self, inputs, query_vec=None, key_vec=None):
        shape = inputs.shape
        # inputs = self._cast_to_fprop_dtype(inputs)  # preserve float32 logits
        # assert shape[1] == self.num_heads
        #if self.use_conv:  # an order of magnitude slower than einsum!
        #  ret = lax.conv(inputs, self.w, (1, 1), 'SAME') # BNTS,MN11->BMTS
        #  if self.residual: ret = ret + inputs
        #  return torch.reshape(ret, shape)
        if self.transpose:
            # inputs = torch.reshape(inputs, shape[:3] + (self.num_groups, self.num_heads_per_group))
            inputs = rearrange(inputs, 'B (G M) T S -> B T S G M', G=self.num_groups)
            exp = 'BTSGM,GMN->BTSGN'; exp_gate = 'BTSGI,BTGI->BTSGI'; exp2 = 'BTSGM,GM->BTSGM'
            if hasattr(self, 'b'): b = self.b
        else:
            # inputs = torch.reshape(inputs, (shape[0], self.num_groups, self.num_heads_per_group) + shape[2:])
            inputs = rearrange(inputs, 'B (G M) T S -> B G M T S', G=self.num_groups)
            exp = 'BGMTS,GMN->BGNTS' if not self.left_mul else 'GNM,BGMTS->BGNTS'
            exp_gate = 'BGITS,BTGI->BGITS'; exp2 = 'BGMTS,GM->BGMTS'  # for ffn, N=I=hidden_dim
            if hasattr(self, 'b'): b = torch.expand_dims(self.b, dim=(1, 2))
        ret = 0.
        if self.use_static_w:
            if self.squeeze_ratio is None:
                w = self.w + torch.eye(self.num_heads_per_group) \
                  if self.residual and self.absorb_residual else self.w
                ret = torch.einsum(exp, inputs, w) if not self.left_mul else torch.einsum(exp, w, inputs)
            else:
                ret = torch.einsum(exp, inputs, self.w1) if not self.left_mul else torch.einsum(exp, self.w1, inputs)
                if hasattr(self, 'b'): ret = ret + b
                #ret = self.activation(ret)
                #if self.dynamic_squeeze_gate_act_cls is not None:
                #    if not self.addictive_gate:
                #      gate_value = torch.einsum('BTD,DGI->BTGI', query_vec, self.wg) + self.bg
                #      ret = torch.einsum(exp_gate, ret, self.gate_activation(gate_value))
                #    else:
                #      gate_value = torch.einsum('BTD,DGI->BTGI', query_vec, self.wg)
                #      gate_value = rearrange(gate_value, 'B T G I -> B T 1 G I')
                #      ret = ret + gate_value
                #      gate_value2 = torch.einsum('BSD,DGI->BSGI', key_vec, self.wg2)
                #      gate_value2 = rearrange(gate_value2, 'B S G I -> B 1 S G I')
                #      ret = ret + gate_value2
                if hasattr(self, 'activation'): ret = self.activation(ret)
                static_w_vec = torch.tensor(self.ablations.get('static_w', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.static_w_p
                ret = torch.einsum(exp, ret, self.w2 * static_w_vec) if not self.left_mul else torch.einsum(exp, self.w2 * static_w_vec, ret)
                #print('static w', ret.shape, (self.w1[0] * self.w2[0]).shape)
                #ret *= self.ablations.get('static_w', 1)
          
        if self.dynamic_w_hidden_dim is not None:
            pass
            #dw_hidden = torch.einsum('BTD,DGK->BTGK', query_vec, self.dw1)
            #q_hidden, k_hidden = torch.split(dw_hidden, 2, dim=-1)
            #dw_hidden = rearrange(q_hidden, 'B T G K -> B T 1 G K') + \
            #  rearrange(k_hidden, 'B S G K -> B 1 S G K')  # BTSGK
            #dw_hidden = self.dw_hidden_activation(dw_hidden + self.dwhb)

            #w1 = torch.einsum('BTSGK,GKMI->BTSGMI', dw_hidden, self.dw2_w1)
            #if self.use_dw_bias: w1 = w1 + self.dwb_w1
            #if self.dw_activation_cls is not None: w1 = self.dw_activation(w1)
            #hidden = torch.einsum('BTSGM,BTSGMI->BTSGI', inputs, w1)
            #w2 = torch.einsum('BTSGK,GKMI->BTSGIM', dw_hidden, self.dw2_w2)
            #if self.use_dw_bias: w2 = w2 + self.dwb_w2
            #if self.dw_activation_cls is not None: w2 = self.dw_activation(w2)
            #ret = ret + torch.einsum('BTSGI,BTSGIM->BTSGM', hidden, w2)

            #if self.learnable_diag:
            #    d = torch.einsum('BTSGK,GKM->BTSGM', dw_hidden, self.dw2_d)
            #    if self.dw_activation_cls is not None: d = self.dw_activation(d)
            #    ret = ret + inputs * d # torch.einsum('BTSGM,BTSGM->BTSGM', inputs, d)

        if self.dynamic_w_init is not None:
            #dynamic_w_vec = torch.tensor(self.ablations.get('dynamic_w', 1)).to(inputs.device)
            #dynamic_w_dd_vec = torch.tensor(self.ablations.get('dynamic_w_dd', 1)).to(inputs.device)
            dynamic_w_vec_src = torch.tensor(self.ablations.get('dynamic_w_src', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_src_p
            dynamic_w_vec_tgt = torch.tensor(self.ablations.get('dynamic_w_tgt', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_tgt_p
            dynamic_w_dd_vec_src = torch.tensor(self.ablations.get('dynamic_w_dd_src', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_dd_src_p
            dynamic_w_dd_vec_tgt = torch.tensor(self.ablations.get('dynamic_w_dd_tgt', 1)).to(inputs.device) if hasattr(self, 'ablations') else self.dynamic_w_dd_tgt_p
            if self.use_dw_bias: dw = dw + self.dwb  # BTGM(4I)+GM(4I)=BTGM(4I)
            #if hasattr(self, 'dw_activation'): dw = self.dw_activation(dw) # v2
            #print('dw', dw.shape, type(arrs), arrs)
            if self.dynamic_w_hidden_dim: # v2
                dw_hidden = torch.einsum('BTD,DGK->BTGK', query_vec, self.dw1)
                #if self.use_dw_hidden_bias: dw_hidden += self.dwhb
                dw_hidden = self.dw_hidden_activation(dw_hidden)
                q_hidden, k_hidden = torch.split(dw_hidden, dw_hidden.shape[-1] // 2, dim=-1)
                qw1, qw2 = torch.split(torch.einsum('BTGK,GKMI->BTGMI', q_hidden, self.qw), self.qw.shape[-1] // 2, dim=-1)
                kw1, kw2 = torch.split(torch.einsum('BTGK,GKMI->BTGMI', k_hidden, self.kw), self.kw.shape[-1] // 2, dim=-1)
            else:
                dw = torch.einsum('BTD,DGMI->BTGMI', query_vec, self.dw)
                qw1, qw2, kw1, kw2 = torch.split(dw, dw.shape[-1] // 4, dim=-1) # split difference between torch and jax
            qw2, kw2 = rearrange(qw2, 'B T G M I -> B T G I M'), rearrange(kw2, 'B T G M I -> B T G I M')
            if hasattr(self, 'dw1_norm'): # v2
                #print('inputs', inputs.shape)
                #print('qw1', qw1.shape)
                qw1 = self.dw1_norm(qw1)
                kw1 = self.dw1_norm(kw1)
                #print('qw1', qw1.shape)
            if self.tgt_dependent:
                #print('tgt_dependent', inputs.shape, qw1.shape)
                hidden = torch.einsum('BTSGM,BTGMI->BTSGI', inputs, qw1)
                #print('debug', hidden.shape, qw2.shape)
                ret = ret + torch.einsum('BTSGI,BTGIM->BTSGM', hidden, qw2 * dynamic_w_vec_tgt)
            if self.src_dependent:
                hidden = torch.einsum('BTSGM,BSGMI->BTSGI', inputs, kw1)
                ret = ret + torch.einsum('BTSGI,BSGIM->BTSGM', hidden, kw2* dynamic_w_vec_src) 
            if self.learnable_diag:
                dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd)
                if hasattr(self, 'dw_activation'): dd = self.dw_activation(dd) # v2
                qdd, kdd = torch.split(dd, dd.shape[-1] // 2, dim=-1)
                if self.tgt_dependent or not self.tgt_dependent and not self.src_dependent:
                    ret = ret + torch.einsum('BTSGM,BTGM->BTSGM', inputs, qdd* dynamic_w_dd_vec_tgt) 
                if self.src_dependent or not self.tgt_dependent and not self.src_dependent:
                    ret = ret + torch.einsum('BTSGM,BSGM->BTSGM', inputs, kdd* dynamic_w_dd_vec_src)
            #print('dynamic w', ret.shape, )
            self._dynamic_w = [qw1, qw2, kw1, kw2, qdd, kdd]

        # ret = self.output_activation(ret)  # for post_proj, relu here degrade performance to baseline
        if self.use_static_w and self.residual and not self.absorb_residual:
            ret = ret + (torch.einsum(exp2, inputs, self.d) \
            if self.learnable_diag and self.dynamic_w_init is None else inputs)
        if hasattr(self, 'output_activation'):
            ret = self.output_activation(ret)  # for post_proj, relu here has no effect on performance
        if self.transpose: ret = ret.permute((0, 3, 4, 1, 2))  # BTSGM->BGMTS
        # inputs = inputs + torch.repeat(inputs[:, :, 0:1, :, :], self.num_heads_per_group, dim=2)  # torch.roll(inputs, 1, dim=2) #
        return torch.reshape(ret, shape)  # BGMTS->BNTS

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

def load_model(model='Llama', size='small'):
    DATA_DIR = '/home/mengqy/Data'
    model_key = f'{model}_{size}'
    with open(f'config_{size}.json', 'r') as f: config = json.loads(f.read())
    if model == 'Llama':
        weight_pax = torch.load(f'{DATA_DIR}/models/{model_paths[model_key]}')
        config = LlamaConfig(**config)
        model = LlamaForCausalLM(config)
    elif model == 'LlamaTalking':
        weight_pax = torch.load(f'{DATA_DIR}/models/{model_paths[model_key]}')
        config = LlamaConfigTalking(**config)
        model = LlamaForCausalLMTalking(config)
    model = match_weight(model, weight_pax)
    return model




if __name__ == '__main__':
    with open('config_small.json', 'r') as f:
        config = json.loads(f.read())
    config = LlamaConfigTalking(**config)
    print(config)
    model = LlamaForCausalLMTalking(config)
    print(model)
