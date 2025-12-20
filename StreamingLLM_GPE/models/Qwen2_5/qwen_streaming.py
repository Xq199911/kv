# coding=utf-8
# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).

import os
import sys
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
import math
from transformers.generation.utils import *
from transformers.utils import is_flash_attn_2_available, add_start_docstrings_to_model_forward, logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
    repeat_kv,
    rotate_half,
)
from StreamingLLM_GPE.generation.generate import unified_PreTrainedModel

logger = logging.get_logger(__name__)
if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


# -----------------------------------------------------------------------------
# Helper Function
# -----------------------------------------------------------------------------
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------------------------------------------------------
# Output Data Classes
# -----------------------------------------------------------------------------
@dataclass
class CausalLMOutputWithPast_stream(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    source_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BaseModelOutputWithPast_stream(BaseModelOutputWithPast):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    source_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    causal_mask: Optional[torch.Tensor] = None
    _wait: Optional[list] = None


# -----------------------------------------------------------------------------
# Rotary Embedding
# -----------------------------------------------------------------------------
class Qwen2RotaryEmbedding_streaming(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# -----------------------------------------------------------------------------
# Attention Module (FIXED with Padding Mask)
# -----------------------------------------------------------------------------
class Qwen2Attention_stream(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.rotary_emb = Qwen2RotaryEmbedding_streaming(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        is_training = kwargs.get("is_training", None)
        source_key_values = kwargs.get("source_key_values", None)
        ReadAction = kwargs.get("ReadAction", None)
        generate_mode = kwargs.get("generate_mode", "batch")

        # [Auto-detect streaming mode]
        if generate_mode == 'batch' and (source_key_values is not None or ReadAction):
            generate_mode = 'streaming'

        # =========================================================================
        # Training Mode
        # =========================================================================
        if is_training:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if position_embeddings is None:
                cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
            else:
                cos, sin = position_embeddings

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

        # =========================================================================
        # Inference Mode
        # =========================================================================
        else:
            if generate_mode == 'batch':
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                kv_seq_len = key_states.shape[-2]
                if past_key_value is not None:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

                if position_embeddings is None:
                    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
                else:
                    cos, sin = position_embeddings

                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx,
                                                                     cache_kwargs)

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None

            elif generate_mode == 'streaming':
                assert ReadAction is not None
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if ReadAction:
                    # Reading Source
                    attention_scores_for_source_cache = None
                    if source_key_values is not None and hasattr(source_key_values, 'key_cache'):
                        if len(source_key_values.key_cache) > self.layer_idx:
                            past_source_k = source_key_values.key_cache[self.layer_idx]
                            cand_source_key = torch.cat([past_source_k, key_states], dim=2)
                        else:
                            cand_source_key = key_states

                        if self.num_key_value_groups > 1:
                            cand_source_key = repeat_kv(cand_source_key, self.num_key_value_groups)

                        attn_weights_source_temp = torch.matmul(query_states,
                                                                cand_source_key.transpose(2, 3)) / math.sqrt(
                            self.head_dim)

                        attention_scores_for_source_cache = nn.functional.softmax(attn_weights_source_temp, dim=-1,
                                                                                  dtype=torch.float32)

                    cache_kwargs_source = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": None,
                        "attention_scores": attention_scores_for_source_cache
                    }

                    key_states, value_states = source_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs_source
                    )

                elif not ReadAction:
                    # Generating Target
                    attention_scores_for_cache = None
                    if past_key_value is not None and hasattr(past_key_value, 'key_cache'):
                        if len(past_key_value.key_cache) > self.layer_idx:
                            past_k = past_key_value.key_cache[self.layer_idx]
                            cand_key = torch.cat([past_k, key_states], dim=2)
                            if self.num_key_value_groups > 1:
                                cand_key = repeat_kv(cand_key, self.num_key_value_groups)
                            attn_weights_temp = torch.matmul(query_states, cand_key.transpose(2, 3)) / math.sqrt(
                                self.head_dim)
                            attention_scores_for_cache = nn.functional.softmax(attn_weights_temp, dim=-1,
                                                                               dtype=torch.float32)

                    cache_kwargs_target = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": None,
                        "attention_scores": attention_scores_for_cache
                    }

                    target_key, target_value = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs_target
                    )

                    if hasattr(source_key_values, 'key_cache'):
                        if len(source_key_values.key_cache) <= self.layer_idx:
                            source_key = torch.empty(0, device=target_key.device, dtype=target_key.dtype)
                            source_value = torch.empty(0, device=target_value.device, dtype=target_value.dtype)
                        else:
                            source_key = source_key_values.key_cache[self.layer_idx]
                            source_value = source_key_values.value_cache[self.layer_idx]
                    else:
                        source_key, source_value = source_key_values[self.layer_idx]

                    # [CONCATENATE]
                    if source_key.numel() > 0:
                        key_states = torch.cat([source_key, target_key], dim=2)
                        value_states = torch.cat([source_value, target_value], dim=2)
                    else:
                        key_states, value_states = target_key, target_value

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

                # ==================== [CRITICAL FIX: Apply Head-Aware Padding Mask] ====================
                s_mask = None
                t_mask = None

                # 1. Get Source Mask
                if hasattr(source_key_values, 'padding_masks') and self.layer_idx in source_key_values.padding_masks:
                    s_mask = source_key_values.padding_masks[self.layer_idx]

                # 2. Get Target Mask (Only if not Reading Action, usually)
                if not ReadAction:
                    if hasattr(past_key_value, 'padding_masks') and self.layer_idx in past_key_value.padding_masks:
                        t_mask = past_key_value.padding_masks[self.layer_idx]

                # 3. Construct Combined Mask based on context
                combined_mask = None
                if ReadAction:
                    combined_mask = s_mask
                else:
                    # We need S + T. If masks are missing, generate zeros to match lengths
                    if s_mask is None and hasattr(source_key_values, 'key_cache') and len(
                            source_key_values.key_cache) > self.layer_idx:
                        s_len_actual = source_key_values.key_cache[self.layer_idx].shape[2]
                        s_mask = torch.zeros((bsz, self.num_key_value_heads, 1, s_len_actual),
                                             device=query_states.device, dtype=query_states.dtype)

                    if t_mask is None and hasattr(past_key_value, 'key_cache') and len(
                            past_key_value.key_cache) > self.layer_idx:
                        t_len_actual = past_key_value.key_cache[self.layer_idx].shape[2]
                        t_mask = torch.zeros((bsz, self.num_key_value_heads, 1, t_len_actual),
                                             device=query_states.device, dtype=query_states.dtype)

                    if s_mask is not None and t_mask is not None:
                        combined_mask = torch.cat([s_mask, t_mask], dim=-1)
                    elif s_mask is not None:
                        combined_mask = s_mask
                    elif t_mask is not None:
                        combined_mask = t_mask

                # 4. Apply Mask
                if combined_mask is not None:
                    # GQA Support: Repeat mask if needed
                    if self.num_key_value_groups > 1:
                        combined_mask = repeat_kv(combined_mask, self.num_key_value_groups)

                    # Ensure alignment
                    if combined_mask.shape[-1] == attn_weights.shape[-1]:
                        attn_weights = attn_weights + combined_mask
                # ====================================================================================

                if attention_mask is not None:
                    if attention_mask.size(-1) >= key_states.shape[-2]:
                        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                        attn_weights = attn_weights + causal_mask

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, source_key_values, past_key_value


# -----------------------------------------------------------------------------
# Placeholder Classes
# -----------------------------------------------------------------------------
class Qwen2FlashAttention2_stream(Qwen2Attention_stream):
    pass


class Qwen2SdpaAttention_stream(Qwen2Attention_stream):
    pass


QWEN2_ATTENTION_CLASSES_STREAM = {
    "eager": Qwen2Attention_stream,
    "flash_attention_2": Qwen2FlashAttention2_stream,
    "sdpa": Qwen2SdpaAttention_stream,
}


# -----------------------------------------------------------------------------
# Decoder Layer
# -----------------------------------------------------------------------------
class Qwen2DecoderLayer_stream(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES_STREAM[config._attn_implementation](config, layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        source_key_values = kwargs.pop('source_key_values', None)

        attn_outputs, self_attn_weights, source_key_values, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            source_key_values=source_key_values,
            **kwargs,
        )
        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (source_key_values, past_key_value)

        return outputs


QWEN2_STREAMING_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`].
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens.
        training_mode (`str`, *optional*):
            Training mode for the model. Can be one of `batch`, or `streaming`.
        is_training (`bool`, *optional*):
            Whether the model is in training mode.
        source_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            For streaming decoding, the key and value states of the source tokens.
        ReadAction (`bool`, *optional*):
            For streaming decoding, whether the model is reading the source tokens.
        remove_last_token (`bool`, *optional*):
            For streaming decoding, whether the last token should be removed.
"""


# -----------------------------------------------------------------------------
# Qwen2 Model (FIXED)
# -----------------------------------------------------------------------------
class Qwen2Model_stream(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer_stream(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rotary_emb = Qwen2RotaryEmbedding_streaming(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def generate_attention_mask(self, input_batch_len, source_seg_len, target_seg_len, wait_tokens_list,
                                training_mode='streaming', device='cpu'):
        assert training_mode in ['streaming']
        if not isinstance(wait_tokens_list, list):
            wait_tokens_list = wait_tokens_list.tolist()
        assert len(wait_tokens_list) == len(target_seg_len)

        if training_mode == 'streaming':
            # Step 1: Calculate attn_mask
            prompt_len = source_seg_len[0]
            source_token_len = sum(source_seg_len)
            target_token_len = sum(target_seg_len)
            total_len = source_token_len + target_token_len
            streaming_start = source_token_len
            inf = -3e+38
            attn_mask = torch.triu(torch.ones(input_batch_len, input_batch_len), diagonal=1).to(device) * inf
            if total_len < input_batch_len:
                attn_mask[total_len:, :] = inf
            for index, num in enumerate(target_seg_len):
                wait_words = wait_tokens_list[index]
                wait_tokens = sum(source_seg_len[:wait_words])  # prompt as 1 word
                attn_mask[streaming_start:streaming_start + num, wait_tokens:source_token_len] = inf
                streaming_start += num

            # Need to move the source attention mask in streaming part up by one
            attn_mask[source_token_len: source_token_len + target_token_len - 1, :source_token_len] = attn_mask[
                                                                                                      source_token_len + 1:source_token_len + target_token_len,
                                                                                                      :source_token_len]
        return attn_mask

    def generate_wait_words(self, source_seg_len, target_seg_len, wait_k):
        wait_tokens_list = []
        source_word_total = len(source_seg_len)

        for tgt_idx in range(len(target_seg_len)):
            source_words_waited = min(wait_k + tgt_idx, source_word_total)
            wait_tokens_list.append(source_words_waited)

        return wait_tokens_list

    @add_start_docstrings_to_model_forward(QWEN2_STREAMING_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            training_mode: Optional[str] = 'streaming',
            generate_mode: Optional[str] = 'batch',
            split_mode: Optional[str] = 'word',
            is_training: Optional[bool] = False,
            source_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            ReadAction: Optional[bool] = False,
            remove_last_token: Optional[bool] = False,
            _lengths: Optional[List[dict]] = None,
            _lengths_index: Optional[torch.tensor] = None,
            attn_mask_index: Optional[torch.tensor] = None,
            wait_k: Optional[float] = None,
            **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if source_key_values is None:
            source_key_values = getattr(self, 'source_key_values', None)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False

        if use_cache and past_key_values is not None and not self.training:
            if hasattr(past_key_values, 'update'):
                use_legacy_cache = False
            elif not isinstance(past_key_values, Cache):
                use_legacy_cache = True
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once("Converted legacy tuple to DynamicCache.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

            if source_key_values is not None and not ReadAction:
                if hasattr(source_key_values, '_seen_tokens'):
                    past_seen_tokens += source_key_values._seen_tokens
                elif hasattr(source_key_values, 'get_seq_length'):
                    past_seen_tokens += source_key_values.get_seq_length()

            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if is_training:
            if training_mode == 'batch':
                causal_mask = self._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )
                _wait = None
            elif training_mode in ['streaming']:
                if split_mode == 'word':
                    assert wait_k is not None
                    inf = -3e+38
                    batch_size = inputs_embeds.shape[0]
                    input_batch_len = _lengths[0]['input_batch_len']
                    causal_mask = torch.triu(torch.ones(input_batch_len, input_batch_len), diagonal=1).to(
                        inputs_embeds.device) * inf
                    causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)
                    _wait = []
                    for index, len_dict in enumerate(_lengths):
                        source_len, source_seg_len = len_dict['source_token_len'], len_dict['source_seg_len']
                        target_len, target_seg_len = len_dict['target_token_len'], len_dict['target_seg_len']
                        wait_tokens_list = self.generate_wait_words(source_seg_len, target_seg_len, wait_k)
                        attn_mask = self.generate_attention_mask(input_batch_len, source_seg_len, target_seg_len,
                                                                 wait_tokens_list, training_mode=training_mode,
                                                                 device=inputs_embeds.device)
                        causal_mask[index] = attn_mask
                        _wait.append(wait_tokens_list)

        else:
            if generate_mode == 'batch':
                if past_key_values is not None and len(past_key_values) > 0:
                    history_source_length = past_key_values[0][0].shape[-2]
                else:
                    history_source_length = 0
                current_length = history_source_length + inputs_embeds.shape[1]
                cache_position = torch.arange(history_source_length, current_length, dtype=torch.long,
                                              device=cache_position.device)
                causal_mask = self._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )
                _wait = None
            elif generate_mode == 'streaming':
                _wait = None
                assert ReadAction is not None
                if ReadAction:
                    if hasattr(source_key_values, 'get_seq_length'):
                        history_source_length = source_key_values.get_seq_length()
                    else:
                        history_source_length = 0

                    current_length = history_source_length + inputs_embeds.shape[1]
                    cache_position = torch.arange(history_source_length, current_length, dtype=torch.long,
                                                  device=cache_position.device)
                    causal_mask = self._update_causal_mask(
                        attention_mask[:, :current_length] if attention_mask is not None else None,
                        inputs_embeds, cache_position, past_key_values, output_attentions)
                else:
                    total_past_length = 0
                    if source_key_values is not None:
                        if hasattr(source_key_values, 'get_seq_length'):
                            total_past_length += source_key_values.get_seq_length()
                        elif isinstance(source_key_values, (list, tuple)) and len(source_key_values) > 0:
                            total_past_length += source_key_values[0][0].shape[-2]

                    if past_key_values is not None:
                        total_past_length += past_key_values.get_seq_length()
                    current_len = total_past_length + inputs_embeds.shape[1]

                    if attention_mask is not None:
                        if attention_mask.shape[-1] >= current_len:
                            attention_mask_sliced = attention_mask[:, :current_len]
                        else:
                            pad_len = current_len - attention_mask.shape[-1]
                            ones = torch.ones((attention_mask.shape[0], pad_len), device=attention_mask.device,
                                              dtype=attention_mask.dtype)
                            attention_mask_sliced = torch.cat([attention_mask, ones], dim=-1)
                    else:
                        attention_mask_sliced = None

                    import inspect
                    sig = inspect.signature(_prepare_4d_causal_attention_mask)
                    kwargs = {
                        "attention_mask": attention_mask_sliced,
                        "past_key_values_length": total_past_length,
                        "sliding_window": self.config.sliding_window,
                    }
                    if "cache_position" in sig.parameters:
                        kwargs["cache_position"] = cache_position
                    if "input_shape" in sig.parameters:
                        kwargs["input_shape"] = inputs_embeds.shape[:-1]
                    if "sequence_length" in sig.parameters:
                        kwargs["sequence_length"] = inputs_embeds.shape[1]
                    if "inputs_embeds" in sig.parameters:
                        kwargs["inputs_embeds"] = inputs_embeds
                    if "embedding_layer" in sig.parameters:
                        kwargs["embedding_layer"] = inputs_embeds

                    causal_mask = _prepare_4d_causal_attention_mask(**kwargs)
            else:
                raise ValueError(f"generate_mode should be 'batch' or 'streaming', but got {generate_mode}")

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    is_training=is_training,
                    source_key_values=source_key_values,
                    ReadAction=ReadAction,
                    remove_last_token=remove_last_token,
                    generate_mode=generate_mode,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[-1]
                source_key_values = layer_outputs[-2]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast_stream(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            source_key_values=source_key_values,
            causal_mask=causal_mask,
            _wait=_wait,
        )


class Qwen2ForCausalLM_stream(unified_PreTrainedModel, Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model_stream(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: List[torch.FloatTensor],
            output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        import inspect
        sig = inspect.signature(_prepare_4d_causal_attention_mask)
        kwargs = {
            "attention_mask": attention_mask,
            "past_key_values_length": past_key_values.get_seq_length() if past_key_values is not None else 0,
            "sliding_window": self.config.sliding_window,
            "cache_position": cache_position,
        }

        if "input_shape" in sig.parameters:
            kwargs["input_shape"] = input_tensor.shape[:-1]
        if "sequence_length" in sig.parameters:
            kwargs["sequence_length"] = input_tensor.shape[1]
        if "inputs_embeds" in sig.parameters:
            kwargs["inputs_embeds"] = input_tensor
        if "embedding_layer" in sig.parameters:
            kwargs["embedding_layer"] = input_tensor

        return _prepare_4d_causal_attention_mask(**kwargs)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QWEN2_STREAMING_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            training_mode: Optional[str] = 'streaming',
            generate_mode: Optional[str] = 'batch',
            split_mode: Optional[str] = 'word',
            is_training: Optional[bool] = False,
            source_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            ReadAction: Optional[bool] = False,
            remove_last_token: Optional[bool] = False,
            _lengths: Optional[List[dict]] = None,
            _lengths_index: Optional[torch.tensor] = None,
            attn_mask_index: Optional[torch.tensor] = None,
            wait_k: Optional[float] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast_stream]:

        if source_key_values is None:
            source_key_values = getattr(self, 'source_key_values', None)

        if _lengths_index is not None:
            _lengths = [_lengths[i] for i in _lengths_index]
            if is_training:
                attn_mask_index = [attn_mask_index[i] for i in _lengths_index]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            training_mode=training_mode,
            generate_mode=generate_mode,
            split_mode=split_mode,
            is_training=is_training,
            source_key_values=source_key_values,
            ReadAction=ReadAction,
            remove_last_token=remove_last_token,
            _lengths=_lengths,
            wait_k=wait_k,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and attn_mask_index is not None:
            loss = self.loss_function(training_mode=training_mode, logits=logits, labels=labels,
                                      attn_mask_index=attn_mask_index)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast_stream(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def loss_function(self, training_mode, logits, labels, attn_mask_index):
        if training_mode == "batch":
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = labels.contiguous().view(-1)
            loss_mask = torch.cat(attn_mask_index).contiguous().view(-1)
            loss_mask_batch_labels = torch.where(loss_mask == 1)[0]
            loss_mask_batch_logits = loss_mask_batch_labels.clone() - 1
            loss_NLL = F.cross_entropy(shift_logits[loss_mask_batch_logits], shift_labels[loss_mask_batch_labels])

        elif training_mode == "streaming":
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = labels.contiguous().view(-1)
            loss_mask = torch.cat(attn_mask_index).contiguous().view(-1)
            loss_mask_streaming_labels = torch.where(loss_mask == 2)[0]
            loss_mask_streaming_logits = (torch.cat(attn_mask_index) == 2)
            count_per_sample = loss_mask_streaming_logits.sum(dim=1)
            first_idx = torch.cat([torch.tensor([0], device=count_per_sample.device), count_per_sample[:-1]]).cumsum(
                dim=0)
            loss_mask_streaming_logits = loss_mask_streaming_labels.clone() - 1
            loss_mask_streaming_logits[first_idx] = torch.where(torch.cat(attn_mask_index).contiguous().view(-1) == -1)[
                0]
            loss_NLL = F.cross_entropy(shift_logits[loss_mask_streaming_logits],
                                       shift_labels[loss_mask_streaming_labels])
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        return loss_NLL