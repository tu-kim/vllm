# SPDX-License-Identifier: Apache-2.0
"""ComposableKV: GPU block pool for position-independent KV (PI-KV) chunks.

The GPU KV cache is split by block id: [0, num_prefix_blocks) belong to the
ordinary prefix-cache BlockPool, [num_prefix_blocks, num_blocks) to this pool.
Entries are whole chunks keyed by (chunk_hash, offset): the chunk's KV as
rotated for absolute position `offset`, so the same chunk at two offsets is two
entries. Eviction is per entry (LRU among entries no request references) and
never touches the prefix pool, and vice versa.
"""

from __future__ import annotations

import time
from collections import OrderedDict
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
    ref_cnt: int = 0
    last_used: float = field(default_factory=time.monotonic)
    # Set by the worker side once the KV has actually been copied into the blocks.
    loaded: bool = False


@dataclass
class PiEvent:
    kind: str  # "stored" | "removed"
    key: PiKey
    num_tokens: int = 0


class PiPool:
    def __init__(self, first_block_id: int, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.first_block_id = first_block_id
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [KVCacheBlock(first_block_id + i) for i in range(num_blocks)]
        self.free_queue = FreeKVCacheBlockQueue(self.blocks)
        self.entries: dict[PiKey, PiEntry] = {}
        # Unreferenced entries in LRU order (oldest first): eviction candidates.
        self._lru: OrderedDict[PiKey, None] = OrderedDict()
        self.events: list[PiEvent] = []

    # ---- queries
    def lookup(self, key: PiKey) -> PiEntry | None:
        return self.entries.get(key)

    def get_num_free_blocks(self) -> int:
        return self.free_queue.num_free_blocks

    def get_usage(self) -> float:
        return 1.0 - self.get_num_free_blocks() / self.num_blocks

    def num_blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    # ---- lifecycle
    def allocate(self, key: PiKey, num_tokens: int) -> PiEntry | None:
        """Reserve blocks for a new entry and reference it once. Evicts
        unreferenced entries, least recently used first, until the blocks fit.
        Returns None (and evicts nothing) if the referenced entries alone leave
        too little room, or if the key already exists."""
        if key in self.entries:
            return None
        need = self.num_blocks_for(num_tokens)
        if need > self.num_blocks:
            return None
        evictable = sum(len(self.entries[k].blocks) for k in self._lru)
        if need > self.get_num_free_blocks() + evictable:
            return None
        while need > self.get_num_free_blocks():
            victim, _ = self._lru.popitem(last=False)
            self._remove(victim)
        blocks = self.free_queue.popleft_n(need)
        for b in blocks:
            assert b.ref_cnt == 0
            b.ref_cnt = 1
        entry = PiEntry(key, num_tokens, blocks, ref_cnt=1)
        self.entries[key] = entry
        self.events.append(PiEvent("stored", key, num_tokens))
        return entry

    def acquire(self, key: PiKey) -> PiEntry | None:
        """Reference an existing entry (a request is about to use its blocks)."""
        entry = self.entries.get(key)
        if entry is None:
            return None
        if entry.ref_cnt == 0:
            self._lru.pop(key, None)
        entry.ref_cnt += 1
        entry.last_used = time.monotonic()
        return entry

    def release(self, key: PiKey) -> None:
        entry = self.entries[key]
        assert entry.ref_cnt > 0, f"release of unreferenced PI entry {key}"
        entry.ref_cnt -= 1
        if entry.ref_cnt == 0:
            entry.last_used = time.monotonic()
            self._lru[key] = None  # newest at the end

    def remove(self, key: PiKey) -> bool:
        """Drop an unreferenced entry explicitly (e.g. its chunk went stale)."""
        entry = self.entries.get(key)
        if entry is None or entry.ref_cnt > 0:
            return False
        self._lru.pop(key, None)
        self._remove(key)
        return True

    def _remove(self, key: PiKey) -> None:
        entry = self.entries.pop(key)
        for b in entry.blocks:
            b.ref_cnt = 0
        self.free_queue.append_n(entry.blocks)
        self.events.append(PiEvent("removed", key))

    def reset(self) -> bool:
        """Drop every unreferenced entry. False if some entry is still in use."""
        for key in list(self._lru):
            self._lru.pop(key)
            self._remove(key)
        return not self.entries

    def take_events(self) -> list[PiEvent]:
        events, self.events = self.events, []
        return events
