# SPDX-License-Identifier: Apache-2.0
"""ComposableKV single-row one-step layout (F9, V-COMP-9): attention for the
sparse rows of a batch.

A sparse row's query tokens sit at disjoint prompt positions (its recompute
segments); the KV between them is spliced PI-KV already in the paged cache.
Paged FlashAttention/Triton kernels take one (query span, kv length) per row,
so they compute such a row wrongly (they do not write KV, so nothing else is
harmed). This module recomputes just those rows with FlexAttention, whose
mask is causal by absolute position (``FlexAttentionMetadata.q_positions``),
and overwrites their output: the dense backend handles the batch as usual,
FlexAttention only the sparse rows. Both backends lay the paged KV cache out
the same way (NHD), so one cache serves both kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata

if TYPE_CHECKING:
    from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

Q_PAD = 128  # q_positions padding: the mask may be evaluated past the last token


def sub_batch(
    common: CommonAttentionMetadata, rows: list[int]
) -> tuple[CommonAttentionMetadata, torch.Tensor]:
    """The sparse rows as a batch of their own, plus the indices of their
    tokens in the full batch (for the query gather / output scatter)."""
    assert common.positions is not None, "sparse rows need per-token positions"
    device = common.seq_lens.device
    qsl = common.query_start_loc_cpu
    starts = [int(qsl[i]) for i in rows]
    ends = [int(qsl[i + 1]) for i in rows]
    lens = [e - s for s, e in zip(starts, ends)]
    token_idx_cpu = torch.cat(
        [torch.arange(s, e, dtype=torch.int64) for s, e in zip(starts, ends)]
    )
    n = int(token_idx_cpu.shape[0])
    token_idx = token_idx_cpu.to(device)
    sub_qsl_cpu = torch.zeros(len(rows) + 1, dtype=torch.int32)
    sub_qsl_cpu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    rows_cpu = torch.tensor(rows, dtype=torch.int64)
    rows_t = rows_cpu.to(device)
    seq_lens = common.seq_lens[rows_t]
    if common.seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu = common.seq_lens_cpu_upper_bound[rows_cpu].to(torch.int32)
    else:
        seq_lens_cpu = seq_lens.to("cpu")
    positions = common.positions[token_idx]
    pad = round_up(n, Q_PAD) - n
    if pad:
        positions = torch.cat([positions, positions.new_zeros(pad)])
    sub = CommonAttentionMetadata(
        query_start_loc=sub_qsl_cpu.to(device),
        query_start_loc_cpu=sub_qsl_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        num_reqs=len(rows),
        num_actual_tokens=n,
        max_query_len=max(lens),
        max_seq_len=int(seq_lens_cpu.max()),
        block_table_tensor=common.block_table_tensor[rows_t],
        slot_mapping=common.slot_mapping[token_idx],
        causal=True,
        positions=positions,
        ckv_sparse_req_indices=list(range(len(rows))),
    )
    return sub, token_idx


def build(
    builder: Any, common: CommonAttentionMetadata
) -> tuple["FlexAttentionMetadata", torch.Tensor]:
    """Flex metadata for the sparse rows of `common` (built by the dense
    backend's metadata builder, which lazily owns a Flex builder)."""
    flex_builder = getattr(builder, "_ckv_flex_builder", None)
    if flex_builder is None:
        from vllm.v1.attention.backends.flex_attention import (
            FlexAttentionMetadataBuilder,
        )

        flex_builder = FlexAttentionMetadataBuilder(
            builder.kv_cache_spec,
            builder.layer_names,
            builder.vllm_config,
            builder.device,
        )
        builder._ckv_flex_builder = flex_builder
    rows = common.ckv_sparse_req_indices
    assert rows
    sub, token_idx = sub_batch(common, rows)
    return flex_builder.build(0, sub), token_idx


def _flex_impl(impl: Any):
    """A FlexAttention impl matching the dense impl's layer parameters."""
    flex = getattr(impl, "_ckv_flex_impl", None)
    if flex is not None:
        return flex
    from vllm.utils.torch_utils import is_quantized_kv_cache
    from vllm.v1.attention.backends.flex_attention import FlexAttentionImpl
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    unsupported = []
    if impl.attn_type != AttentionType.DECODER:
        unsupported.append(f"attn_type {impl.attn_type}")
    if getattr(impl, "alibi_slopes", None) is not None:
        unsupported.append("alibi")
    if tuple(getattr(impl, "sliding_window", (-1, -1))) != (-1, -1):
        unsupported.append("sliding window")
    if getattr(impl, "logits_soft_cap", 0) not in (None, 0, 0.0):
        unsupported.append("logits soft cap")
    if getattr(impl, "sinks", None) is not None:
        unsupported.append("sinks")
    if is_quantized_kv_cache(impl.kv_cache_dtype):
        unsupported.append(f"kv cache dtype {impl.kv_cache_dtype}")
    try:
        layout = get_kv_cache_layout()  # needs the engine's config (not in unit tests)
    except AssertionError:
        layout = "NHD"
    if layout != "NHD":
        unsupported.append(f"kv cache layout {layout}")
    if unsupported:
        raise NotImplementedError(
            "ComposableKV single-row layout (CKV_ONE_STEP=flex) cannot run "
            "FlexAttention for this layer: " + ", ".join(unsupported)
        )
    flex = FlexAttentionImpl(
        impl.num_heads,
        impl.head_size,
        impl.scale,
        impl.num_kv_heads,
        None,
        None,
        impl.kv_cache_dtype,
        None,
        AttentionType.DECODER,
    )
    impl._ckv_flex_impl = flex
    return flex


def forward(
    impl: Any,
    layer: torch.nn.Module,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    """Recompute the sparse rows' attention with FlexAttention over the same
    paged KV cache and overwrite their rows of `output`."""
    flex = _flex_impl(impl)
    token_idx = attn_metadata.ckv_sparse_token_idx
    q = query.index_select(0, token_idx)
    out = torch.empty_like(q)
    flex.forward(layer, q, q, q, kv_cache, attn_metadata.ckv_sparse, out)
    output.index_copy_(0, token_idx, out)
