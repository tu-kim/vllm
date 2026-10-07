# SPDX-License-Identifier: Apache-2.0
"""ComposableKV: GPU block pool for position-independent KV (PI-KV) chunks.

The GPU KV cache is split by block id: [0, num_prefix_blocks) belong to the
ordinary prefix-cache BlockPool, [num_prefix_blocks, num_blocks) to this pool.
Entries are whole chunks keyed by (chunk_hash, offset): the chunk's KV as
rotated for absolute position `offset`, so the same chunk at two offsets is two
entries. Eviction is per entry (LRU among entries no request references) and
never touches the prefix pool, and vice versa.

Reference counting rides on KVCacheBlock.ref_cnt like every other block: the
entry itself holds one reference on each of its blocks, and requests add theirs
through BlockPool.touch / free_blocks (a PI block never reaches ref_cnt 0 while
its entry exists, so the prefix pool never recycles it). An entry is evictable
when every block is back to ref_cnt == 1.

PI blocks may also be registered in the prefix-cache hash map (under the chain
hash of a request that used them) so a later request with the same prefix hits
them like any cached block; evicting the entry removes those hashes through
`evict_callback`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


class PiKey(NamedTuple):
    chunk_hash: str
    offset: int


@dataclass
class PiEntry:
    key: PiKey
    num_tokens: int
    blocks: list[KVCacheBlock]
    last_used: float = field(default_factory=time.monotonic)
    # Set once the worker has copied the KV into the blocks.
    loaded: bool = False

    @property
    def ref_cnt(self) -> int:
        """Number of requests referencing the entry (block refs minus the entry's own)."""
        return max(b.ref_cnt for b in self.blocks) - 1


@dataclass
class PiEvent:
    kind: str  # "stored" | "removed" | "cleared"
    key: PiKey | None
    num_tokens: int = 0


class PiPool:
    def __init__(self, first_block_id: int, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.first_block_id = first_block_id
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [KVCacheBlock(first_block_id + i, is_shared=True) for i in range(num_blocks)]
        self.free_queue = FreeKVCacheBlockQueue(self.blocks)
        self.entries: dict[PiKey, PiEntry] = {}
        self.events: list[PiEvent] = []
        # Called with an entry's blocks right before they are freed, so the
        # owner can drop prefix-cache hashes that point at them.
        self.evict_callback: Callable[[list[KVCacheBlock]], None] | None = None

    # ---- queries
    def lookup(self, key: PiKey) -> PiEntry | None:
        return self.entries.get(key)

    def is_pi_block(self, block: KVCacheBlock) -> bool:
        return block.is_shared and self.first_block_id <= block.block_id < self.first_block_id + self.num_blocks

    def get_num_free_blocks(self) -> int:
        return self.free_queue.num_free_blocks

    def get_usage(self) -> float:
        return 1.0 - self.get_num_free_blocks() / self.num_blocks

    def num_blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def evictable(self) -> list[PiEntry]:
        """Unreferenced entries, least recently used first."""
        return sorted((e for e in self.entries.values() if e.ref_cnt == 0), key=lambda e: e.last_used)

    # ---- lifecycle
    def allocate(self, key: PiKey, num_tokens: int) -> PiEntry | None:
        """Reserve blocks for a new entry. Evicts unreferenced entries, least
        recently used first, until the blocks fit. Returns None (and evicts
        nothing) if the referenced entries alone leave too little room, or if
        the key already exists. The blocks carry only the entry's own reference;
        the caller adds the request's via BlockPool.touch."""
        if key in self.entries:
            return None
        need = self.num_blocks_for(num_tokens)
        if need > self.num_blocks:
            return None
        victims = self.evictable()
        if need > self.get_num_free_blocks() + sum(len(v.blocks) for v in victims):
            return None
        for victim in victims:
            if need <= self.get_num_free_blocks():
                break
            self._remove(victim.key)
        blocks = self.free_queue.popleft_n(need)
        for b in blocks:
            assert b.ref_cnt == 0
            b.ref_cnt = 1
        entry = PiEntry(key, num_tokens, blocks)
        self.entries[key] = entry
        self.events.append(PiEvent("stored", key, num_tokens))
        return entry

    def touch(self, key: PiKey) -> PiEntry | None:
        """Mark an entry as used now (LRU bookkeeping only)."""
        entry = self.entries.get(key)
        if entry is not None:
            entry.last_used = time.monotonic()
        return entry

    def remove(self, key: PiKey) -> bool:
        """Drop an unreferenced entry explicitly (e.g. its chunk went stale)."""
        entry = self.entries.get(key)
        if entry is None or entry.ref_cnt > 0:
            return False
        self._remove(key)
        return True

    def _remove(self, key: PiKey) -> None:
        entry = self.entries.pop(key)
        if self.evict_callback is not None:
            self.evict_callback(entry.blocks)
        for b in entry.blocks:
            b.ref_cnt = 0
        self.free_queue.append_n(entry.blocks)
        self.events.append(PiEvent("removed", key))

    def reset(self) -> bool:
        """Drop every unreferenced entry. False if some entry is still in use."""
        for entry in self.evictable():
            self._remove(entry.key)
        if not self.entries:
            # One event instead of a removal per entry (V-OBS-1 ChunksCleared).
            self.events = [e for e in self.events if e.kind != "removed"]
            self.events.append(PiEvent("cleared", None))
        return not self.entries

    def take_events(self) -> list[PiEvent]:
        events, self.events = self.events, []
        return events
