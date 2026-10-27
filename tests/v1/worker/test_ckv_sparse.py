# SPDX-License-Identifier: Apache-2.0
"""ComposableKV F9 (V-COMP-9), runner side on CPU (T9-3): a single-row request's
positions, sequence length and prompt token ids come from its segments.

    pytest tests/v1/worker/test_ckv_sparse.py -q
"""

import numpy as np
import torch

from vllm.v1.worker import ckv_sparse

SEGMENTS = [(0, 48), (160, 16), (240, 20)]


class Out:
    def __init__(self, segs):
        self.ckv_segments = segs


def test_positions_and_end():
    pos = ckv_sparse.positions_of(SEGMENTS)
    assert pos.dtype == np.int64 and pos.shape == (84,)
    assert pos[:48].tolist() == list(range(48))
    assert pos[48:64].tolist() == list(range(160, 176))
    assert pos[64:].tolist() == list(range(240, 260))
    assert ckv_sparse.end_of(SEGMENTS) == 260


def test_sparse_rows_follow_batch_order():
    out = Out({"b": SEGMENTS, "zzz": [(5, 3)]})
    assert ckv_sparse.sparse_rows(out, ["a", "b", "c"]) == [(1, SEGMENTS)]
    assert ckv_sparse.sparse_rows(Out(None), ["a"]) == []
    assert ckv_sparse.sparse_rows(object(), ["a"]) == []


def test_mrv1_positions_and_query_pos():
    """Batch: a decode row (1 token at 99), the sparse row (84 tokens, computed
    count 0), a plain prefill chunk (3 tokens at 10)."""
    num_scheduled = np.array([1, 84, 3], dtype=np.int32)
    cu = np.cumsum(num_scheduled)
    num_computed = np.array([99, 0, 10], dtype=np.int32)
    req_indices = np.repeat(np.arange(3), num_scheduled)
    query_pos = np.concatenate([np.arange(n, dtype=np.int64) for n in num_scheduled])
    positions = num_computed[req_indices] + query_pos
    rows = [(1, SEGMENTS)]
    ckv_sparse.apply_mrv1(rows, cu, num_computed, positions, query_pos)
    assert positions[0] == 99 and positions[85:].tolist() == [10, 11, 12]
    assert positions[1:85].tolist() == ckv_sparse.positions_of(SEGMENTS).tolist()
    # the device formula num_computed[req] + query_pos reproduces the positions
    assert (num_computed[req_indices] + query_pos).tolist() == positions.tolist()


def test_mrv1_sparse_row_after_a_budget_cut():
    """Second step of a budget split: computed count 172, segments
    [(172, 4), (240, 20)]: query_pos is the offset from 172."""
    segs = [(172, 4), (240, 20)]
    num_scheduled = np.array([24], dtype=np.int32)
    cu = np.cumsum(num_scheduled)
    num_computed = np.array([172], dtype=np.int32)
    query_pos = np.arange(24, dtype=np.int64)
    positions = num_computed[np.zeros(24, dtype=np.int64)] + query_pos
    ckv_sparse.apply_mrv1([(0, segs)], cu, num_computed, positions, query_pos)
    assert positions.tolist() == list(range(172, 176)) + list(range(240, 260))
    assert query_pos.tolist() == list(range(4)) + list(range(68, 88))


def test_mrv2_buffers_from_segments():
    """Device buffers (here CPU) filled as the kernels would from num_computed,
    then overwritten for the sparse row: positions, seq_len, prompt token ids."""
    query_start_loc = np.array([0, 1, 85, 88], dtype=np.int32)
    idx_mapping = np.array([7, 3, 5], dtype=np.intp)  # request state slots
    positions = torch.zeros(88, dtype=torch.int64)
    positions[1:85] = torch.arange(84)  # num_computed 0 + i
    seq_lens = torch.tensor([100, 84, 13], dtype=torch.int32)
    input_ids = torch.full((88,), -1, dtype=torch.int32)
    all_token_ids = torch.zeros((8, 300), dtype=torch.int32)
    all_token_ids[3] = torch.arange(300) + 1000  # token id == 1000 + position
    input_ids[1:85] = all_token_ids[3, :84]
    rows = ckv_sparse.apply_mrv2(
        [(1, SEGMENTS)], query_start_loc, idx_mapping, positions, seq_lens, input_ids, all_token_ids
    )
    assert rows == [1]
    assert positions[1:85].tolist() == ckv_sparse.positions_of(SEGMENTS).tolist()
    assert seq_lens.tolist() == [100, 260, 13]
    assert input_ids[1:85].tolist() == [1000 + p for p in ckv_sparse.positions_of(SEGMENTS)]
    assert input_ids[0] == -1 and input_ids[85:].tolist() == [-1, -1, -1]
