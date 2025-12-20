# coding=utf-8
# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology, Ningbo).
# Fixed by User (2025): Full Code - Fix Stale Cache Position Bug - Fix EOS - Fix Position IDs

import inspect
import logging
from typing import List, Dict, Optional, Union, Callable, Any
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache as TransformersDynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    GenerationMode
)
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

try:
    from transformers.utils import is_torchdynamo_compiling
except ImportError:
    def is_torchdynamo_compiling():
        return False
try:
    from transformers.utils import is_deepspeed_zero3_enabled
except ImportError:
    def is_deepspeed_zero3_enabled():
        return False
from .Stopping_criteria import StopTokenCriteria


def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask


class DynamicCache(TransformersDynamicCache):
    def __init__(self) -> None:
        super().__init__()

    def pop(self):
        self._seen_tokens -= 1
        target_key_cache = []
        target_value_cache = []
        for key_cache, value_cache in zip(self.key_cache, self.value_cache):
            target_key_cache.append(key_cache[..., :-1, :])
            target_value_cache.append(value_cache[..., :-1, :])
        self.key_cache = target_key_cache
        self.value_cache = target_value_cache


class unified_PreTrainedModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generate_mode = kwargs.get("generate_mode", "batch")
        split_mode = kwargs.get("split_mode", None)
        if generate_mode == "batch":
            wait_lagging = None
            return super().generate(
                inputs,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                assistant_model,
                streamer,
                negative_prompt_ids,
                negative_prompt_attention_mask,
                **kwargs,
            ), wait_lagging

        elif generate_mode == "streaming":
            assert split_mode in ["token",
                                  "word"], f"streaming_split must be one of ['token', 'word'], but got {split_mode}."
            result, wait_lagging = self.streaming_generate(
                split_mode,
                inputs,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                assistant_model,
                streamer,
                negative_prompt_ids,
                negative_prompt_attention_mask,
                **kwargs,
            )
            return result, wait_lagging
        else:
            raise ValueError(f"generate_mode must be one of ['batch', 'streaming'], but got {generate_mode}.")

    def streaming_generate(
            self,
            streaming_split: str = "word",
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        tokenizer = kwargs.pop("tokenizer", None)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs_streaming(model_kwargs.copy())
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # [FIX] Ensure pad_token_id is set
        if generation_config.pad_token_id is None and tokenizer is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id

        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            if (
                    generation_config._pad_token_tensor is not None
                    and batch_size > 1
                    and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning("Decoder-only architecture with right-padding detected.")
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache
        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)
        if streamer is not None:
            streamer.put(input_ids.cpu())
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
        use_dynamic_cache_by_default = False
        cache_name = "past_key_values"
        source_cache_name = "source_key_values"
        if generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            if past is None:
                model_kwargs[cache_name] = DynamicCache()
                use_dynamic_cache_by_default = True

            source_past = model_kwargs.get(source_cache_name, None)
            if source_past is None:
                model_kwargs[source_cache_name] = DynamicCache()
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        generation_mode = generation_config.get_generation_mode(assistant_model)
        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError("`streamer` cannot be used with beam search.")
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            prepared_logits_warper = None
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            result, wait_lagging = self._sample_streaming(
                streaming_split=streaming_split,
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                tokenizer=tokenizer,
                **model_kwargs,
            )
        return result, wait_lagging

    def _validate_model_kwargs_streaming(self, model_kwargs: Dict[str, Any]):
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(f"{self.__class__.__name__} does not support an instance of `Cache`.")
        pass

    def merge_source_target(self):
        pass

    def separate_source_target(self):
        pass

    def _get_initial_cache_position_for_streaming(self, input_length, model_kwargs):
        assert self.source_key_values is not None
        if hasattr(self.source_key_values, 'get_seq_length'):
            current_len = self.source_key_values.get_seq_length()
        else:
            current_len = 0
        cache_position = torch.arange(
            current_len, input_length[0], dtype=torch.int64, device=model_kwargs.get('assistant_token').device
        )
        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def prepare_inputs_for_generation_streaming(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            is_streaming=False,
            input_length=None,
            pe_cache_length=0,
            assistant_token=None,
            source_words=None,
            end_Instruct=None,
            ReadAction=True,
            **kwargs,
    ):
        if not is_streaming:
            pass
        elif is_streaming:
            assert input_length is not None, "input_length must be provided for streaming generation"
            model_inputs = kwargs.copy()
            past_source_length = 0
            past_target_length = 0

            if self.source_key_values is not None:
                if hasattr(self.source_key_values, 'get_seq_length'):
                    past_source_length = self.source_key_values.get_seq_length()
                elif isinstance(self.source_key_values, (list, tuple)) and len(self.source_key_values) > 0:
                    past_source_length = self.source_key_values[0][0].shape[-2]

            if self.past_key_values is not None:
                if hasattr(self.past_key_values, 'get_seq_length'):
                    past_target_length = self.past_key_values.get_seq_length()
                elif isinstance(self.past_key_values, (list, tuple)) and len(self.past_key_values) > 0:
                    past_target_length = self.past_key_values[0][0].shape[-2]

            if ReadAction:
                position_ids_source = torch.arange(past_source_length, input_length[0]).to(
                    assistant_token.device).unsqueeze(0)
                position_ids = position_ids_source.clone().detach()
                input_ids = input_ids[:, past_source_length:input_length[0]]
            elif not ReadAction:
                # Generating Target

                # [CRITICAL FIX] Use Stable Position IDs for Target
                # Using continuous IDs from the STABLE pe_cache_length passed in.
                # This ensures Target[0] is always pe_cache_length, Target[1] is pe_cache_length+1...
                # regardless of future source tokens.
                current_pos = pe_cache_length + past_target_length

                position_ids = torch.arange(
                    current_pos,
                    current_pos + input_ids.shape[-1],
                    dtype=torch.long,
                    device=assistant_token.device
                ).unsqueeze(0)
                input_ids = input_ids

                # ================= [CRITICAL FIX] =================
                # IMPORTANT: Set cache_position to None during generation!
                # This fixes the Mismatch between Physical Cache Index (0,1,2..) and RoPE Position ID (15,16,17..)
                cache_position = None
                # ==================================================

            if ReadAction:
                model_inputs.update(
                    {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "use_cache": use_cache,
                        "attention_mask": attention_mask,
                        "source_key_values": self.source_key_values,
                        "pe_cache_length": pe_cache_length
                    }
                )
            else:
                model_inputs.update(
                    {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "use_cache": use_cache,
                        "attention_mask": attention_mask,
                        "past_key_values": self.past_key_values,
                        "pe_cache_length": pe_cache_length,
                        "source_key_values": self.source_key_values,
                        "cache_position": cache_position  # Explicitly update this!
                    }
                )
        return model_inputs

    def _sample_streaming(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional["BaseStreamer"],
            tokenizer: Optional["PreTrainedTokenizerBase"],
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        if self.source_key_values is None:
            self.source_key_values = DynamicCache()
        if self.target_key_values is None:
            self.target_key_values = DynamicCache()
        if self.past_key_values is None:
            self.past_key_values = DynamicCache()
        end_Instruct = model_kwargs.get("end_Instruct", None)
        max_new_tokens = generation_config.max_new_tokens if hasattr(generation_config,
                                                                     'max_new_tokens') and generation_config.max_new_tokens is not None else 1024
        ReadAction_criteria = StopTokenCriteria(tokenizer, max_new_tokens=max_new_tokens, end_Instruct=end_Instruct)

        _lengths = model_kwargs.get("_lengths", None)
        source_seg_len = _lengths[0]['source_seg_len']

        full_source_len = sum(source_seg_len)

        # [CRITICAL FIX] Define Stable Start Position for Target
        # This LOCKS the target generation start point to the end of the initial Wait-k context.
        # This prevents the "Gibberish" caused by shifting/fragmented position IDs.
        # Target IDs will be: streaming_start_len, streaming_start_len+1, ...
        wait_k = model_kwargs.get("wait_k", 5)
        streaming_start_len = sum(source_seg_len[:wait_k + 1])

        assistant_token = model_kwargs.get("assistant_token", None)
        ReadAction = True
        source_words = wait_k if wait_k is not None else 0
        target_words = 0
        max_distance = 100
        next_tokens = model_kwargs['assistant_token'].unsqueeze(0)
        target_tokens = [next_tokens[:, :-1], next_tokens[:, -1:]]
        target_tokens_this_write = []
        wait_lagging = []
        input_length = (sum(source_seg_len[:model_kwargs['wait_k'] + 1]), 1)
        source_input_length = sum(source_seg_len[:model_kwargs['wait_k'] + 1])
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        batch_size, input_len = input_ids.shape
        cur_len = input_len
        generated_tokens_count = 0
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # Initial cache position setup
        model_kwargs = self._get_initial_cache_position_for_streaming(input_length, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if ReadAction:
                model_inputs = self.prepare_inputs_for_generation_streaming(input_ids, input_length=input_length,
                                                                            ReadAction=ReadAction, is_streaming=True,
                                                                            **model_kwargs)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
                _outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    ReadAction=ReadAction,
                )
                ReadAction = False
                token_count = 0
            elif not ReadAction:
                token_count += 1

                kwargs_without_pe = {k: v for k, v in model_kwargs.items() if k != 'pe_cache_length'}

                # [CRITICAL] Fix: Pass streaming_start_len as fixed offset.
                # This is the "Anchor" that stabilizes generation and prevents gibberish.
                # Do NOT set this to 0.
                model_inputs = self.prepare_inputs_for_generation_streaming(
                    next_tokens,
                    input_length=input_length,
                    ReadAction=ReadAction,
                    is_streaming=True,
                    pe_cache_length=streaming_start_len,  # 使用锁定的基准位置
                    **kwargs_without_pe
                )

                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    ReadAction=ReadAction,
                )
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                if synced_gpus and this_peer_finished:
                    continue
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_scores = logits_processor(input_ids, next_token_logits)
                if return_dict_in_generate:
                    if output_scores: scores += (next_token_scores,)
                    if output_logits: raw_logits += (next_token_logits,)
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # ================= [CRITICAL FIX] Force Stop on EOS =================
                current_token_id = next_tokens.item()
                is_eos = False
                if current_token_id == 151645:
                    is_eos = True
                elif current_token_id == 151643:
                    is_eos = True
                elif tokenizer is not None and tokenizer.eos_token_id is not None:
                    if isinstance(tokenizer.eos_token_id, list):
                        if current_token_id in tokenizer.eos_token_id:
                            is_eos = True
                    elif current_token_id == tokenizer.eos_token_id:
                        is_eos = True

                if is_eos:
                    this_peer_finished = True
                    unfinished_sequences.fill_(0)
                    break
                    # ====================================================================

                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                del outputs
                next_tokens = next_tokens.unsqueeze(0)
                target_tokens.append(next_tokens)
                target_ids = torch.cat(target_tokens, dim=-1)
                target_tokens_this_write.append(next_tokens)
                target_ids_this_write = torch.cat(target_tokens_this_write, dim=-1)
                ReadAction_new, remove_last_token = ReadAction_criteria(target_ids_this_write, scores, token_count)
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(target_ids[0:, 2:], scores)
                generated_tokens_count += 1
                if generated_tokens_count >= max_new_tokens:
                    this_peer_finished = True
                cur_len += 1
                wait_lagging.append(source_words)
                target_words += 1
                source_finished = source_words >= len(source_seg_len) - 1
                if source_finished:
                    ReadAction = False
                elif not ReadAction_new:
                    ReadAction = True
                    source_words += 1
                    if source_words < len(source_seg_len):
                        num_tokens = source_seg_len[source_words]
                        source_input_length += num_tokens
                        target_input_length = 1
                        input_length = (source_input_length, target_input_length)
                else:
                    ReadAction = False
                    if remove_last_token:
                        target_tokens.pop()
                        next_tokens = target_tokens[-1]
                        target_tokens_this_write = []
                    distance = target_words - source_words
                    if distance > max_distance and source_words >= len(source_seg_len) - 1:
                        this_peer_finished = True
        if streamer is not None:
            streamer.end()
        assistant_token = model_kwargs.get('assistant_token', None)
        if assistant_token is not None and len(target_tokens) > 0:
            if len(target_tokens) >= 2:
                actual_generated_tokens = target_tokens[2:]
                if len(actual_generated_tokens) > 0:
                    target_ids = torch.cat(actual_generated_tokens, dim=-1)
                else:
                    device = target_tokens[0].device if len(target_tokens) > 0 else input_ids.device
                    target_ids = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=target_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return target_ids, wait_lagging