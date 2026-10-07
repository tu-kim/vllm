# SPDX-License-Identifier: Apache-2.0
"""ComposableKV F2: GPU prefix/PI pool split (docs/TEST_CASES.md T2-1..T2-8).

    pytest tests/v1/core/test_ckv_pi_pool.py -q
"""

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.ckv_pi_pool import PiKey, PiPool
from vllm.v1.core.kv_cache_utils import init_none_hash

from .test_prefix_caching import make_kv_cache_config, make_kv_cache_manager, make_request

BLOCK = 16


@pytest.fixture(autouse=True)
def _none_hash():
    # Request block hashing chains from NONE_HASH, which vLLM's tests seed per hash fn.
    init_none_hash(sha256)


def manager(num_blocks: int, num_pi_blocks: int):
    cfg = make_kv_cache_config(BLOCK, num_blocks)
    cfg.num_pi_blocks = num_pi_blocks
    return make_kv_cache_manager(cfg, max_model_len=8192, hash_block_size=BLOCK, enable_caching=True)


def test_t2_1_split_by_ratio():
    """pi_pool_ratio=0.25 over 1000 blocks: 750 prefix, 250 PI, disjoint id ranges."""
    m = manager(1000, int(1000 * 0.25))
    assert m.block_pool.num_gpu_blocks == 750
    assert m.pi_pool is not None and m.pi_pool.num_blocks == 250
    prefix_ids = {b.block_id for b in m.block_pool.blocks}
    pi_ids = {b.block_id for b in m.pi_pool.blocks}
    assert prefix_ids == set(range(750)) and pi_ids == set(range(750, 1000))
    assert m.pi_usage == 0.0 and m.usage == 0.0


def use(pool: PiPool, key: PiKey, m=None):
    """A request takes its reference on an entry's blocks (what BlockPool.touch does)."""
    for b in pool.entries[key].blocks:
        b.ref_cnt += 1


def unuse(pool: PiPool, key: PiKey):
    """A request drops its reference (what BlockPool.free_blocks does)."""
    for b in pool.entries[key].blocks:
        b.ref_cnt -= 1


def test_t2_2_prefix_pool_exhaustion_leaves_pi_pool_alone():
    m = manager(100, 20)
    pi = m.pi_pool
    pi.allocate(PiKey("c1", 0), 5 * BLOCK)  # unreferenced: evictable *within* the PI pool
    before = {k: [b.block_id for b in v.blocks] for k, v in pi.entries.items()}
    # Fill the prefix pool completely (79 usable blocks; block 0 is the null block).
    blocks = m.block_pool.get_new_blocks(m.block_pool.get_num_free_blocks())
    assert m.block_pool.get_num_free_blocks() == 0
    with pytest.raises(ValueError):
        m.block_pool.get_new_blocks(1)
    assert {k: [b.block_id for b in v.blocks] for k, v in pi.entries.items()} == before
    assert pi.get_num_free_blocks() == 15
    m.block_pool.free_blocks(blocks)


def test_t2_3_pi_pool_evicts_lru_whole_entries_only():
    m = manager(100, 12)  # PI pool: 12 blocks
    pi = m.pi_pool
    for name in ("a", "b", "c"):
        pi.allocate(PiKey(name, 0), 4 * BLOCK)
    assert pi.get_num_free_blocks() == 0
    pi.touch(PiKey("a", 0))  # "a" is now most recently used
    prefix_free = m.block_pool.get_num_free_blocks()
    e = pi.allocate(PiKey("d", 0), 2 * BLOCK)  # needs 2 blocks -> evicts one whole entry: LRU is "b"
    assert e is not None
    assert PiKey("b", 0) not in pi.entries and PiKey("a", 0) in pi.entries and PiKey("c", 0) in pi.entries
    assert pi.get_num_free_blocks() == 2  # 4 freed, 2 reused; no partial-entry eviction
    assert m.block_pool.get_num_free_blocks() == prefix_free  # prefix pool untouched
    kinds = [(ev.kind, ev.key.chunk_hash) for ev in pi.take_events()]
    assert kinds == [("stored", "a"), ("stored", "b"), ("stored", "c"), ("removed", "b"), ("stored", "d")]


def test_t2_4_referenced_entries_are_never_evicted():
    m = manager(100, 8)
    pi = m.pi_pool
    a = pi.allocate(PiKey("a", 0), 4 * BLOCK)
    b = pi.allocate(PiKey("b", 0), 4 * BLOCK)
    use(pi, a.key)
    use(pi, b.key)
    assert pi.allocate(PiKey("c", 0), 1 * BLOCK) is None  # everything referenced
    assert set(pi.entries) == {a.key, b.key} and pi.get_num_free_blocks() == 0
    unuse(pi, b.key)
    assert pi.allocate(PiKey("c", 0), 1 * BLOCK) is not None  # now "b" can go
    assert b.key not in pi.entries and a.key in pi.entries


def test_t2_5_same_chunk_different_offsets_are_separate_entries():
    pi = PiPool(first_block_id=100, num_blocks=10, block_size=BLOCK)
    e1 = pi.allocate(PiKey("c", 1024), 3 * BLOCK)
    e2 = pi.allocate(PiKey("c", 2048), 3 * BLOCK)
    assert e1 is not None and e2 is not None and e1.key != e2.key
    assert not {b.block_id for b in e1.blocks} & {b.block_id for b in e2.blocks}
    assert pi.allocate(PiKey("c", 1024), 3 * BLOCK) is None  # duplicate key


def test_t2_6_shared_entry_refcount_rides_on_blocks():
    """Two requests use one entry through ordinary block refs; the entry's own
    reference keeps the blocks out of the prefix pool's free path."""
    m = manager(100, 10)
    pi, bp = m.pi_pool, m.block_pool
    e = pi.allocate(PiKey("c", 0), 2 * BLOCK)
    assert e.ref_cnt == 0 and all(b.ref_cnt == 1 for b in e.blocks)
    bp.touch(e.blocks)  # request 1
    bp.touch(e.blocks)  # request 2
    assert e.ref_cnt == 2
    bp.free_blocks(reversed(e.blocks))  # request 1 done: the usual free path
    assert e.ref_cnt == 1 and not pi.remove(e.key)  # still referenced
    assert bp.get_num_free_blocks() == 89  # PI blocks never enter the prefix free queue
    bp.free_blocks(reversed(e.blocks))
    assert e.ref_cnt == 0 and pi.remove(e.key) and pi.get_num_free_blocks() == 10
    assert bp.get_num_free_blocks() == 89


def test_t2_8_ratio_zero_keeps_vllm_behavior():
    m = manager(100, 0)
    assert m.pi_pool is None and m.block_pool.num_gpu_blocks == 100 and m.pi_usage == 0.0
    req = make_request("r", list(range(3 * BLOCK)), BLOCK, sha256)
    blocks, n_hit, _ = m.get_computed_blocks(req)
    assert n_hit == 0
    new = m.allocate_slots(req, 3 * BLOCK, 0, blocks)
    assert new is not None and len(new.get_block_ids()[0]) == 3
    m.free(req)


def test_requests_allocate_from_prefix_pool_only():
    """V-POOL-6: request blocks never come from the PI id range."""
    m = manager(100, 40)
    req = make_request("r", list(range(5 * BLOCK)), BLOCK, sha256)
    blocks, _, _ = m.get_computed_blocks(req)
    new = m.allocate_slots(req, 5 * BLOCK, 0, blocks)
    assert all(bid < 60 for bid in new.get_block_ids()[0])
    assert m.pi_usage == 0.0 and m.usage > 0.0
    m.free(req)


def test_allocation_bigger_than_pool_is_rejected():
    pi = PiPool(first_block_id=0, num_blocks=4, block_size=BLOCK)
    assert pi.allocate(PiKey("big", 0), 5 * BLOCK) is None
    assert pi.get_num_free_blocks() == 4 and not pi.entries


def test_reset_drops_unreferenced_only():
    pi = PiPool(first_block_id=0, num_blocks=8, block_size=BLOCK)
    a = pi.allocate(PiKey("a", 0), BLOCK)
    pi.allocate(PiKey("b", 0), BLOCK)
    use(pi, a.key)
    assert not pi.reset() and set(pi.entries) == {a.key}
    unuse(pi, a.key)
    assert pi.reset() and not pi.entries and pi.get_num_free_blocks() == 8


def test_pi_blocks_in_prefix_cache_follow_the_entry():
    """V-COMP-6: a PI block cached under a request's chain hash is hit by a later
    request with the same prefix, keeps one hash per distinct prefix, and loses
    them all when the entry is evicted."""
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

    m = manager(100, 10)
    pi, bp = m.pi_pool, m.block_pool
    e = pi.allocate(PiKey("c", 0), BLOCK)
    blk = e.blocks[0]
    h1 = make_block_hash_with_group_id(BlockHash(b"prefix-1"), 0)
    h2 = make_block_hash_with_group_id(BlockHash(b"prefix-2"), 0)
    bp._insert_block_hash(h1, blk, num_tokens=BLOCK)
    bp._insert_block_hash(h2, blk, num_tokens=BLOCK)
    assert bp.cached_block_hash_to_block.get_one_block(h1) is blk
    assert bp.cached_block_hash_to_block.get_one_block(h2) is blk
    pi.allocate(PiKey("big", 0), 9 * BLOCK)  # needs the whole pool: evicts "c"
    assert PiKey("c", 0) not in pi.entries
    assert bp.cached_block_hash_to_block.get_one_block(h1) is None
    assert bp.cached_block_hash_to_block.get_one_block(h2) is None
    assert blk.block_hash is None
