from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch
from torch.nn.functional import scaled_dot_product_attention

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def image_ranges_to_block_ids(
    ranges: torch.Tensor, lengths: Sequence[int], prefixes: Sequence[int]
) -> torch.Tensor:
    if ranges.shape != (sum(lengths), 2) or len(lengths) != len(prefixes):
        raise ValueError("Gemma4 image range metadata has inconsistent lengths")
    parts = []
    offset = 0
    for length, prefix in zip(lengths, prefixes):
        parts.append(
            torch.cat(
                (
                    ranges.new_full((prefix,), -1),
                    ranges[offset : offset + length, 0],
                )
            )
        )
        offset += length
    return torch.cat(parts)


def gemma4_image_attention(
    *,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids: torch.Tensor,
    translate_locs: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    if layer.logit_cap:
        raise NotImplementedError(
            "Gemma4 XPU image SDPA does not support attention softcap"
        )
    if any(
        cache.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        for cache in (key_cache, value_cache)
    ):
        raise NotImplementedError("Gemma4 XPU image SDPA requires unquantized KV")
    if (k is None) != (v is None):
        raise ValueError(
            "Gemma4 image attention requires both current K and V or neither"
        )
    query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    keys = None if k is None else k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
    values = None if v is None else v.reshape(-1, layer.tp_k_head_num, layer.v_head_dim)
    key_cache = key_cache.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
    value_cache = value_cache.reshape(-1, layer.tp_k_head_num, layer.v_head_dim)
    output = query.new_empty((len(query), layer.tp_q_head_num, layer.v_head_dim))
    prefixes = forward_batch.extend_prefix_lens_cpu
    lengths = forward_batch.extend_seq_lens_cpu
    slots = forward_batch.req_pool_indices.tolist()
    if block_ids.numel() != sum(p + n for p, n in zip(prefixes, lengths)):
        raise ValueError(
            "Gemma4 image ids do not cover the whole cached+extend sequence"
        )
    query_offset = block_offset = 0
    window = layer.sliding_window_size
    for slot, prefix, length in zip(slots, prefixes, lengths):
        end = prefix + length
        start = max(0, prefix - window) if window >= 0 else 0
        cache_end = prefix if keys is not None and values is not None else end
        locations = req_to_token[slot, start:cache_end]
        if translate_locs is not None:
            locations = translate_locs(locations)
        per_key = key_cache[locations].to(query.dtype)
        per_value = value_cache[locations].to(query.dtype)
        if keys is not None and values is not None:
            per_key = torch.cat((per_key, keys[query_offset : query_offset + length]))
            per_value = torch.cat(
                (per_value, values[query_offset : query_offset + length])
            )
        key_positions = torch.arange(start, end, device=q.device).unsqueeze(0)
        key_ids = block_ids[block_offset + start : block_offset + end].unsqueeze(0)
        # Bound each FP32 score tile to 64 MiB at 8K keys and 16 heads.
        for begin in range(0, length, 128):
            stop = min(begin + 128, length)
            query_positions = torch.arange(
                prefix + begin, prefix + stop, device=q.device
            ).unsqueeze(1)
            allow = key_positions <= query_positions
            if window >= 0:
                query_ids = block_ids[
                    block_offset + prefix + begin : block_offset + prefix + stop
                ].unsqueeze(1)
                allow = (allow | ((query_ids >= 0) & (query_ids == key_ids))) & (
                    key_positions >= query_positions - window
                )
            result = scaled_dot_product_attention(
                query[query_offset + begin : query_offset + stop]
                .transpose(0, 1)
                .unsqueeze(0),
                per_key.transpose(0, 1).unsqueeze(0),
                per_value.transpose(0, 1).unsqueeze(0),
                attn_mask=allow,
                scale=layer.scaling,
                enable_gqa=layer.tp_q_head_num != layer.tp_k_head_num,
            )
            output[query_offset + begin : query_offset + stop] = result.squeeze(
                0
            ).transpose(0, 1)
        query_offset += length
        block_offset += end
    if query_offset != len(query):
        raise ValueError("Gemma4 image request geometry does not cover all queries")
    return output.flatten(1)
