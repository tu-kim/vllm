# SPDX-License-Identifier: Apache-2.0
"""ComposableKV single-row one-step layout (F9, V-COMP-9).

A request laid out this way computes several disjoint prompt segments in one
row; the tokens between them are spliced PI-KV already in its blocks. The
scheduler lists the segments in ``SchedulerOutput.ckv_segments`` and the
model runner derives everything position-dependent from them instead of from
``num_computed_tokens``: the token positions (RoPE, slot mapping, prompt token
ids) and the row's sequence length (attention, sampling). The attention
backend gets the rows' batch indices through ``CommonAttentionMetadata``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

Segments = list[tuple[int, int]]


def sparse_rows(
    scheduler_output, req_ids: Sequence[str]
) -> list[tuple[int, Segments]]:
    """(batch index, segments) of the requests in this batch with a
    single-row layout, in batch order."""
    segs = getattr(scheduler_output, "ckv_segments", None)
    if not segs:
        return []
    return [(i, segs[rid]) for i, rid in enumerate(req_ids) if rid in segs]


def positions_of(segments: Segments) -> np.ndarray:
    """Absolute prompt positions of the row's tokens, in row order."""
    return np.concatenate(
        [np.arange(start, start + n, dtype=np.int64) for start, n in segments]
    )


def end_of(segments: Segments) -> int:
    """The computed boundary after the row: the last segment's end."""
    start, n = segments[-1]
    return start + n


def apply_mrv1(
    rows: list[tuple[int, Segments]],
    cu_num_tokens: np.ndarray,
    num_computed_tokens_cpu: np.ndarray,
    positions_np: np.ndarray,
    query_pos_np: np.ndarray,
) -> None:
    """GPUModelRunner (MRV1): overwrite the rows' positions in ``positions_np``
    (the CPU positions that select the prompt token ids) and in
    ``query_pos_np`` as offsets from the request's computed count, so the
    device-side ``num_computed_tokens[req] + query_pos`` gives the same
    positions."""
    for i, segments in rows:
        start = int(cu_num_tokens[i - 1]) if i > 0 else 0
        end = int(cu_num_tokens[i])
        pos = positions_of(segments)
        assert pos.shape[0] == end - start, (segments, end - start)
        positions_np[start:end] = pos
        query_pos_np[start:end] = pos - int(num_computed_tokens_cpu[i])


def apply_mrv2(
    rows: list[tuple[int, Segments]],
    query_start_loc_np: np.ndarray,
    idx_mapping_np: np.ndarray,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    input_ids: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> list[int]:
    """GPUModelRunner (MRV2): after ``prepare_prefill_inputs`` and
    ``prepare_pos_seq_lens`` filled the device buffers from
    ``num_computed_tokens``, overwrite the rows' positions, sequence length
    and prompt token ids from their segments. Returns the rows' batch
    indices."""
    out = []
    for i, segments in rows:
        start, end = int(query_start_loc_np[i]), int(query_start_loc_np[i + 1])
        pos = torch.from_numpy(positions_of(segments)).to(
            positions.device, non_blocking=True
        )
        assert pos.shape[0] == end - start, (segments, end - start)
        positions[start:end] = pos
        seq_lens[i] = end_of(segments)
        input_ids[start:end] = all_token_ids[int(idx_mapping_np[i]), pos].to(
            input_ids.dtype
        )
        out.append(i)
    return out
