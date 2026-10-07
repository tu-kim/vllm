# SPDX-License-Identifier: Apache-2.0
"""ComposableKV F3: scheduler-level composition (segment-wise prefill with
PI blocks spliced in), driven by a fake connector. No GPU.

    pytest tests/v1/core/test_ckv_compose.py -q
"""

import pytest

from vllm.v1.core.ckv_pi_pool import PiKey
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

BLOCK = 16
PROMPT = 200  # tokens
# plan: recompute [0,48)  pi [48,160) chunk "c" (link 48 tokens => offset 48, key (c, 0))  recompute [160,200)
PI_START, PI_END = 48, 160


class FakeComposer:
    """Wraps the scheduler's real connector; adds the composition hooks."""

    def __init__(self, inner, pi_pool, plan_reqs: set[str]):
        self.inner, self.pi_pool, self.plan_reqs = inner, pi_pool, plan_reqs
        self.spliced: list[str] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def bind_pi_pool(self, pi_pool):
        self.pi_pool = pi_pool

    def take_events(self):
        # What CkvConnector.take_events does for the GPU pool (the wrapped
        # ExampleConnector has no PI pool); the DRAM part needs a store.
        from ckv.vllm_connector import chunk_events_from_pool

        return chunk_events_from_pool(self.pi_pool)

    def get_prefill_limit_at(self, request, num_computed_tokens):
        if request.request_id not in self.plan_reqs:
            return None
        if num_computed_tokens < PI_START:
            return PI_START
        return None

    def get_shared_blocks(self, request):
        if request.request_id not in self.plan_reqs or request.num_computed_tokens != PI_START:
            return None
        key = PiKey("c", 0)
        entry = self.pi_pool.lookup(key) or self.pi_pool.allocate(key, PI_END - PI_START)
        assert entry is not None
        self.spliced.append(request.request_id)
        return list(entry.blocks), PI_END - PI_START


def step_output(reqs, finished: bool) -> ModelRunnerOutput:
    req_ids = [r.request_id for r in reqs]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        # Still prefilling: the model runner returns no tokens for the request.
        sampled_token_ids=[[EOS_TOKEN_ID] if finished else [] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=None,
    )


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setenv("CKV_PI_POOL_RATIO", "0.25")
    s = create_scheduler(
        use_kv_connector=True, enable_prefix_caching=True, num_blocks=200, block_size=BLOCK,
        max_num_batched_tokens=8192,
    )
    assert s.kv_cache_manager.pi_pool is not None and s.kv_cache_manager.pi_pool.num_blocks == 50
    s.connector = FakeComposer(s.connector, s.kv_cache_manager.pi_pool, set())
    return s


def run_prefill(scheduler, req) -> list[SchedulerOutput]:
    """Schedule steps until the prompt is computed; return the step outputs."""
    outs = []
    for _ in range(10):
        out = scheduler.schedule()
        outs.append(out)
        done = req.num_computed_tokens >= req.num_tokens
        scheduler.update_from_output(out, step_output([req], finished=done))
        if done:
            break
    return outs


def test_segmented_prefill_splices_pi_blocks(scheduler):
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    req = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1)[0]
    scheduler.connector.plan_reqs.add(req.request_id)
    scheduler.add_request(req)

    # step 1: only the first recompute segment (48 tokens) is scheduled.
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens[req.request_id] == PI_START
    assert req.num_computed_tokens == PI_START
    scheduler.update_from_output(out1, step_output([req], finished=False))

    # step 2: PI blocks are spliced, the request jumps to 160 and computes the tail.
    out2 = scheduler.schedule()
    assert scheduler.connector.spliced == [req.request_id]
    assert out2.num_scheduled_tokens[req.request_id] == PROMPT - PI_END
    cached = out2.scheduled_cached_reqs
    i = cached.req_ids.index(req.request_id)
    assert cached.num_computed_tokens[i] == PI_END  # what the model runner sees
    # MRV2 keeps a device-side counter that only grows by computed tokens:
    # the splice must be flagged so it reloads the value (once).
    assert cached.resync_num_computed_tokens == {req.request_id}
    entry = pi.lookup(PiKey("c", 0))
    pi_ids = [b.block_id for b in entry.blocks]
    new_ids = cached.new_block_ids[i][0]
    assert new_ids[: len(pi_ids)] == pi_ids and len(new_ids) == len(pi_ids) + 3
    assert all(bid >= pi.first_block_id for bid in pi_ids)
    assert km.get_block_ids(req.request_id)[0][3:10] == pi_ids
    assert entry.ref_cnt == 1
    scheduler.update_from_output(out2, step_output([req], finished=True))

    # finished: the PI entry is unreferenced but kept; its blocks never hit the prefix free queue.
    assert req.status == RequestStatus.FINISHED_STOPPED
    assert entry.ref_cnt == 0 and PiKey("c", 0) in pi.entries
    assert km.block_pool.get_num_free_blocks() == km.block_pool.num_gpu_blocks - 1
    assert all(b.ref_cnt == 1 for b in entry.blocks)


def test_resync_flag_is_sent_only_once(scheduler):
    req = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=4)[0]
    scheduler.connector.plan_reqs.add(req.request_id)
    scheduler.add_request(req)
    outs = run_prefill(scheduler, req)
    flagged = [o.scheduled_cached_reqs.resync_num_computed_tokens for o in outs]
    assert flagged == [set(), {req.request_id}]
    # decode steps: no flag
    out = scheduler.schedule()
    assert out.scheduled_cached_reqs.resync_num_computed_tokens == set()
    scheduler.update_from_output(out, step_output([req], finished=True))


def test_second_request_hits_the_whole_prefix_through_pi_blocks(scheduler):
    """V-COMP-6 / T3-7: PI blocks get the chain hash of the request that used
    them, so the same prompt again is a full prefix hit with no plan needed."""
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    r1, r2 = create_requests(2, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1, same_prompt=True)
    scheduler.connector.plan_reqs.add(r1.request_id)
    scheduler.add_request(r1)
    run_prefill(scheduler, r1)
    assert scheduler.connector.spliced == [r1.request_id]
    pi_ids = [b.block_id for b in pi.lookup(PiKey("c", 0)).blocks]

    scheduler.add_request(r2)  # no plan: plain prefix-cache lookup
    out = scheduler.schedule()
    hit_blocks = 12  # 200 tokens -> hit capped at 199 -> 12 full blocks
    assert out.num_scheduled_tokens[r2.request_id] == PROMPT - hit_blocks * BLOCK
    new = out.scheduled_new_reqs[0]
    assert new.num_computed_tokens == hit_blocks * BLOCK
    assert new.block_ids[0][3:10] == pi_ids  # the hit walked straight through the PI blocks
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 1
    scheduler.update_from_output(out, step_output([r2], finished=True))
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 0
    assert scheduler.connector.spliced == [r1.request_id]  # r2 never needed a splice


def test_evicting_the_entry_drops_prefix_hashes(scheduler):
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    r1 = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1, same_prompt=True)[0]
    scheduler.connector.plan_reqs.add(r1.request_id)
    scheduler.add_request(r1)
    run_prefill(scheduler, r1)
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 0
    # Something else needs the whole PI pool: "c" is evicted along with its hashes.
    assert pi.allocate(PiKey("big", 0), pi.num_blocks * BLOCK) is not None
    assert PiKey("c", 0) not in pi.entries
    r2 = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1, same_prompt=True)[0]
    r2.request_id = "r-after-evict"
    blocks, n_hit, _ = km.get_computed_blocks(r2)
    assert n_hit == PI_START  # hit stops where the PI blocks were


class RecordingPublisher:
    def __init__(self):
        self.batches = []

    def publish(self, batch):
        self.batches.append(batch)

    def shutdown(self):
        pass


def test_pi_pool_changes_are_published_as_chunk_events(scheduler):
    """T5-1: splice -> ChunkStored(GPU); eviction -> ChunkRemoved; reset -> ChunksCleared."""
    from vllm.distributed.kv_events import ChunkRemoved, ChunksCleared, ChunkStored

    pub = RecordingPublisher()
    scheduler.kv_event_publisher = pub
    pi = scheduler.kv_cache_manager.pi_pool
    req = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1)[0]
    scheduler.connector.plan_reqs.add(req.request_id)
    scheduler.add_request(req)
    run_prefill(scheduler, req)
    events = [e for b in pub.batches for e in b.events]
    assert events == [ChunkStored("c", 0, PI_END - PI_START, "GPU")]

    def step():  # events are published when the scheduler processes a step's output
        scheduler.update_from_output(scheduler.schedule(), step_output([], finished=False))

    pub.batches.clear()
    assert pi.allocate(PiKey("big", 0), pi.num_blocks * BLOCK) is not None  # evicts "c"
    step()
    events = [e for b in pub.batches for e in b.events]
    assert events == [ChunkRemoved("c", 0, "GPU"), ChunkStored("big", 0, pi.num_blocks * BLOCK, "GPU")]

    pub.batches.clear()
    assert pi.reset()
    step()
    events = [e for b in pub.batches for e in b.events]
    assert events == [ChunksCleared("GPU")]


def test_requests_without_plan_are_untouched(scheduler):
    req = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1)[0]
    scheduler.add_request(req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[req.request_id] == PROMPT
    assert scheduler.connector.spliced == [] and not scheduler.kv_cache_manager.pi_pool.entries
