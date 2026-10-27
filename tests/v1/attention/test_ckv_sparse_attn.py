# SPDX-License-Identifier: Apache-2.0
"""ComposableKV F9 (V-COMP-9), T9-4 on CPU: FlexAttention with per-token query
positions gives, for a row whose tokens sit at disjoint prompt positions, the
same result as dense causal attention over the whole logical sequence; the
sparse sub-batch is cut out of a mixed batch correctly.

    pytest tests/v1/attention/test_ckv_sparse_attn.py -q
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention

import vllm.v1.attention.backends.flex_attention as flex_mod
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends import ckv_sparse_attn
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionImpl,
    FlexAttentionMetadata,
    physical_to_logical_mapping,
)

BLOCK = 16
NUM_BLOCKS = 16
H, HKV, D = 2, 1, 16
SEQ = 48
PHYS = [5, 2, 7]  # logical blocks 0..2 of the sequence
SEGMENTS = [(0, 16), (32, 16)]  # the row computes these; [16, 32) is spliced KV


def positions_of(segments):
    return torch.cat([torch.arange(s, s + n) for s, n in segments])


def paged_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(num_blocks, HKV, BLOCK, 2*D) cache holding K/V of the 48-token sequence."""
    cache = torch.randn(NUM_BLOCKS, HKV, BLOCK, 2 * D)
    for p in range(SEQ):
        blk, off = PHYS[p // BLOCK], p % BLOCK
        cache[blk, :, off, :D] = k[p]
        cache[blk, :, off, D:] = v[p]
    return cache


def dense_reference(q, k, v, pos):
    """Causal attention of the query rows at `pos` over the logical sequence."""
    kk = k.repeat_interleave(H // HKV, dim=1)  # (SEQ, H, D)
    vv = v.repeat_interleave(H // HKV, dim=1)
    out = torch.empty(len(pos), H, D)
    for i, p in enumerate(pos.tolist()):
        s = torch.einsum("hd,khd->hk", q[i], kk[: p + 1]) / math.sqrt(D)
        out[i] = torch.einsum("hk,khd->hd", torch.softmax(s, dim=-1), vv[: p + 1])
    return out


def sparse_metadata(pos: torch.Tensor) -> FlexAttentionMetadata:
    n = pos.shape[0]
    seq_lens = torch.tensor([SEQ], dtype=torch.int32)
    block_table = torch.tensor([PHYS], dtype=torch.int32)
    qsl = torch.tensor([0, n], dtype=torch.int32)
    q_positions = torch.cat([pos, torch.zeros(128 - n, dtype=torch.int64)])
    return FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=n,
        max_query_len=n,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        max_seq_len=SEQ,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.zeros(n, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=NUM_BLOCKS * BLOCK,
        block_size=BLOCK,
        max_possible_sequence_length=SEQ,
        num_reqs=1,
        physical_to_logical=physical_to_logical_mapping(block_table, seq_lens, BLOCK, NUM_BLOCKS),
        decode_offset=torch.tensor([0], dtype=torch.int32),
        num_blocks_per_seq=cdiv(seq_lens, BLOCK),
        persistent_kv_indices=torch.empty(cdiv(n, 16), 16 * cdiv(SEQ, BLOCK), dtype=torch.int32),
        persistent_kv_num_blocks=torch.empty(cdiv(n, 16), dtype=torch.int32),
        persistent_doc_ids=torch.empty(n, dtype=torch.int32),
        q_block_size=16,
        kv_block_size=16,
        direct_build=True,
        q_positions=q_positions,
    )


def test_mask_is_causal_by_absolute_position():
    pos = positions_of(SEGMENTS)
    meta = sparse_metadata(pos)
    q_idx = torch.arange(pos.shape[0])
    # kv physical index of logical position p
    kv_phys = torch.tensor([PHYS[p // BLOCK] * BLOCK + p % BLOCK for p in range(SEQ)])
    b = torch.zeros((), dtype=torch.int64)
    allowed = meta.mask_mod(b, b, q_idx[:, None].expand(-1, SEQ), kv_phys[None, :].expand(pos.shape[0], -1))
    expected = kv_phys.new_zeros((pos.shape[0], SEQ), dtype=torch.bool)
    for i, p in enumerate(pos.tolist()):
        expected[i, : p + 1] = True
    assert torch.equal(allowed, expected)
    # a never-used physical page is masked everywhere
    unused = torch.full((pos.shape[0], 1), 9 * BLOCK + 3)
    assert not meta.mask_mod(b, b, q_idx[:, None], unused).any()
    # block pruning: the second segment's queries see blocks 0..2, the first only block 0
    bm = meta._build_block_mask_direct()
    kv_num = bm.kv_num_blocks[0, 0].tolist()
    assert kv_num == [1, 3]


def test_flex_matches_dense_causal_reference(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(flex_mod, "flex_attention_compiled", flex_attention)
    pos = positions_of(SEGMENTS)
    k, v = torch.randn(SEQ, HKV, D), torch.randn(SEQ, HKV, D)
    q = torch.randn(pos.shape[0], H, D)
    cache = paged_kv(k, v)
    meta = sparse_metadata(pos)
    impl = FlexAttentionImpl(H, D, 1 / math.sqrt(D), HKV, None, None, "auto", None, AttentionType.DECODER)
    out = torch.empty_like(q)
    impl.forward(SimpleNamespace(), q, q, q, cache, meta, out)
    ref = dense_reference(q, k, v, pos)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    # the same query tokens treated as a contiguous row at 0.. (no q_positions)
    # attend to the wrong keys: the position map is what makes the row right
    meta2 = sparse_metadata(pos)
    meta2.q_positions = None
    meta2.mask_mod = meta2.get_mask_mod()
    out2 = torch.empty_like(q)
    impl.forward(SimpleNamespace(), q, q, q, cache, meta2, out2)
    assert not torch.allclose(out2, ref, atol=1e-3)


def test_sub_batch_cuts_the_sparse_rows():
    """Batch: decode row (1 token), sparse row (32 tokens, seq 48), prefill
    row (3 tokens). The sub-batch is the middle row alone, re-based."""
    qsl = torch.tensor([0, 1, 33, 36], dtype=torch.int32)
    positions = torch.cat([torch.tensor([99]), positions_of(SEGMENTS), torch.tensor([10, 11, 12])])
    common = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor([100, 48, 13], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([100, 48, 13], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=36,
        max_query_len=32,
        max_seq_len=100,
        block_table_tensor=torch.tensor([[1, 2, 3, 4, 5, 6, 7], [5, 2, 7, 0, 0, 0, 0], [8, 0, 0, 0, 0, 0, 0]], dtype=torch.int32),
        slot_mapping=torch.arange(36, dtype=torch.int64) * 10,
        positions=positions,
        ckv_sparse_req_indices=[1],
    )
    sub, token_idx = ckv_sparse_attn.sub_batch(common, [1])
    assert token_idx.tolist() == list(range(1, 33))
    assert sub.num_reqs == 1 and sub.num_actual_tokens == 32 and sub.max_query_len == 32
    assert sub.query_start_loc_cpu.tolist() == [0, 32]
    assert sub.seq_lens.tolist() == [48] and sub.max_seq_len == 48
    assert sub.block_table_tensor.tolist() == [[5, 2, 7, 0, 0, 0, 0]]
    assert sub.slot_mapping.tolist() == [i * 10 for i in range(1, 33)]
    assert sub.positions.shape[0] == 128 and sub.positions[:32].tolist() == positions_of(SEGMENTS).tolist()
    assert sub.ckv_sparse_req_indices == [0]


def test_flex_impl_refuses_unsupported_layers():
    impl = SimpleNamespace(attn_type=AttentionType.DECODER, alibi_slopes=None, sliding_window=(3, 0),
                           logits_soft_cap=0, sinks=None, kv_cache_dtype="auto", num_heads=H, head_size=D,
                           scale=1.0, num_kv_heads=HKV)
    with pytest.raises(NotImplementedError, match="sliding window"):
        ckv_sparse_attn._flex_impl(impl)
