# SPDX-License-Identifier: Apache-2.0
"""ComposableKV F8: one-step forward (V-COMP-8). The scheduler walks the
plan ahead of the computed boundary, splices every PI segment and batches the
earlier recompute segments as pseudo-request rows of the same step; the last
segment is the request's own row. Driven by a fake connector, no GPU.

    pytest tests/v1/core/test_ckv_onestep.py -q
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.ckv_pi_pool import PiKey
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

BLOCK = 16
PROMPT = 260
# (type, start, len, key): [R 0-48][PI 48-160 'c'][R 160-176][PI 176-240 'd'][R 240-260]
PLAN = [
    ("r", 0, 48, None),
    ("pi", 48, 112, "c"),
    ("r", 160, 16, None),
    ("pi", 176, 64, "d"),
    ("r", 240, 20, None),
]
PLAN_NO_PI = [("r", 0, PROMPT, None)]
# Block-aligned prompt (256) ending with a PI segment:
# [R 0-48][PI 'c' 48-160][R 160-176][PI 'd' 176-256]
PLAN_TRAILING_PI = [
    ("r", 0, 48, None),
    ("pi", 48, 112, "c"),
    ("r", 160, 16, None),
    ("pi", 176, 80, "d"),
]
C_START, D_START = 48, 176


class FakeComposer:
    """Wraps the scheduler's real connector; adds the composition hooks for a
    per-request segment list."""

    def __init__(self, inner, pi_pool, one_step: bool):
        self.inner, self.pi_pool, self.one_step = inner, pi_pool, one_step
        self.plans: dict[str, list[tuple]] = {}
        # (request_id, pos argument as passed) per get_shared_blocks hit
        self.spliced: list[tuple[str, int | None]] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def bind_pi_pool(self, pi_pool):
        self.pi_pool = pi_pool

    def take_events(self):
        from ckv.vllm_connector import chunk_events_from_pool

        return chunk_events_from_pool(self.pi_pool)

    def ckv_one_step(self, request):
        return self.one_step

    # Requests whose local prefix hit ends exactly at a PI start: CkvConnector
    # backs the hit off one block (ckv_local_prefix) so the PI is spliced next step.
    backoff_reqs: set[str] = set()

    def ckv_local_prefix(self, request):
        if request.request_id in self.backoff_reqs:
            return C_START - BLOCK
        return None

    def get_prefill_limit_at(self, request, num_computed_tokens):
        plan = self.plans.get(request.request_id)
        if plan is None:
            return None
        # Start of the next PI segment after the computed boundary.
        return next((s for t, s, _, _ in plan if t == "pi" and s > num_computed_tokens), None)

    def get_shared_blocks(self, request, pos=None):
        plan = self.plans.get(request.request_id)
        if plan is None:
            return None
        at = request.num_computed_tokens if pos is None else pos
        for t, start, length, key in plan:
            if t == "pi" and start == at:
                pk = PiKey(key, 0)
                entry = self.pi_pool.lookup(pk) or self.pi_pool.allocate(pk, length)
                assert entry is not None
                self.spliced.append((request.request_id, pos))
                return list(entry.blocks), length
        return None


def step_output(req_ids: list[str], sampled: dict[str, list[int]] | None = None) -> ModelRunnerOutput:
    """Runner output for the given rows; rows not in `sampled` return no token
    (still prefilling, or a pseudo row whose sample the runner discarded)."""
    sampled = sampled or {}
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[sampled.get(rid, []) for rid in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=None,
    )


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def make_scheduler(monkeypatch, request):
    """Every test runs under the plain Scheduler and under AsyncScheduler (the
    server default; Dynamo's InstrumentedScheduler derives from it)."""
    monkeypatch.setenv("CKV_PI_POOL_RATIO", "0.25")
    async_sched = request.param

    def _make(one_step: bool = True, **kwargs):
        kwargs.setdefault("max_num_batched_tokens", 8192)
        kwargs.setdefault("num_blocks", 200)
        kwargs.setdefault("async_scheduling", async_sched)
        s = create_scheduler(
            use_kv_connector=True, enable_prefix_caching=True, block_size=BLOCK, **kwargs
        )
        assert s.kv_cache_manager.pi_pool is not None
        s.connector = FakeComposer(s.connector, s.kv_cache_manager.pi_pool, one_step)
        return s

    return _make


def add_planned_request(scheduler, plan=PLAN, max_tokens: int = 1, num_tokens: int = PROMPT):
    req = create_requests(1, num_tokens=num_tokens, block_size=BLOCK, max_tokens=max_tokens)[0]
    if plan is not None:
        scheduler.connector.plans[req.request_id] = plan
    scheduler.add_request(req)
    return req


def run_prefill(scheduler, req):
    """Schedule until the prompt is computed; returns the step outputs. Pseudo
    rows are echoed back in the runner output like a real runner would."""
    outs = []
    for _ in range(20):
        out = scheduler.schedule()
        outs.append(out)
        rows = list(out.num_scheduled_tokens)
        scheduler.update_from_output(out, step_output(rows))
        if req.num_computed_tokens >= req.num_tokens:
            return outs
    raise AssertionError("prompt not computed in 20 steps")


def drive(
    scheduler, req, others: dict[str, list[int]] | None = None, max_steps: int = 20
):
    """Schedule until the request's prompt is computed, echoing what a runner
    returns: a token for every decoding request in `others`, and the request's
    first token (7) in the step that completes its prompt."""
    outs = []
    for _ in range(max_steps):
        out = scheduler.schedule()
        outs.append(out)
        rows = list(out.num_scheduled_tokens)
        sampled = {rid: tok for rid, tok in (others or {}).items() if rid in rows}
        done = (
            req.request_id in rows
            and req.num_computed_tokens >= req.num_prompt_tokens
        )
        if done:
            sampled[req.request_id] = [7]
        scheduler.update_from_output(out, step_output(rows, sampled))
        if done:
            return outs
    raise AssertionError(f"prompt not computed in {max_steps} steps")


def pi_ids(pi, key: str) -> list[int]:
    return [b.block_id for b in pi.lookup(PiKey(key, 0)).blocks]


# T8-1 / T8-4 ----------------------------------------------------------------


@pytest.mark.parametrize("use_v2", [False, True])
def test_one_step_folds_recompute_segments(make_scheduler, use_v2):
    scheduler = make_scheduler(use_v2_model_runner=use_v2)
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    req = add_planned_request(scheduler, max_tokens=2)
    rid = req.request_id

    # step 0 (admission): only the first recompute segment.
    out0 = scheduler.schedule()
    assert out0.num_scheduled_tokens == {rid: C_START}
    assert out0.scheduled_new_reqs[0].req_id == rid and len(out0.scheduled_new_reqs) == 1
    scheduler.update_from_output(out0, step_output([rid]))
    assert req.num_computed_tokens == C_START
    step0_ids = km.get_block_ids(rid)[0]
    assert len(step0_ids) == 3

    # step 1: PI 'c' spliced, R 160-176 as a pseudo row, PI 'd' spliced, R 240-260 as
    # the request's own row -- one forward.
    out1 = scheduler.schedule()
    assert len(out1.scheduled_new_reqs) == 1
    pseudo = out1.scheduled_new_reqs[0]
    pid = pseudo.req_id
    assert pid == f"{rid}-ckv0"
    assert out1.num_scheduled_tokens == {rid: 20, pid: 16}
    assert out1.total_num_scheduled_tokens == 36
    assert out1.num_common_prefix_blocks == [0]  # cascade attention off
    assert scheduler.connector.spliced == [(rid, C_START), (rid, D_START)]  # explicit pos

    # the pseudo row: parent's prompt and (full) block table, starts at its segment
    full_ids = km.get_block_ids(rid)
    assert pseudo.num_computed_tokens == 160
    assert pseudo.block_ids == full_ids
    assert pseudo.prompt_token_ids == req.prompt_token_ids
    assert pseudo.mm_features == [] and pseudo.pooling_params is None
    assert pseudo.sampling_params is not req.sampling_params
    assert pseudo.sampling_params.logprobs is None and pseudo.sampling_params.prompt_logprobs is None
    assert pseudo.sampling_params.max_tokens == req.sampling_params.max_tokens
    if use_v2:
        assert pseudo.prefill_token_ids == req._all_token_ids
    else:
        assert pseudo.prefill_token_ids is None

    # the parent row: computed boundary at its segment start, resync flagged once
    cached = out1.scheduled_cached_reqs
    assert cached.req_ids == [rid]
    assert cached.num_computed_tokens == [240]
    assert cached.resync_num_computed_tokens == {rid}

    # block order: [3 step-0][7 'c'][1 new][4 'd'][2 new], new_block_ids in the same order
    c_ids, d_ids = pi_ids(pi, "c"), pi_ids(pi, "d")
    assert len(c_ids) == 7 and len(d_ids) == 4
    new_ids = cached.new_block_ids[0][0]
    assert len(new_ids) == 14
    assert new_ids[:7] == c_ids and new_ids[8:12] == d_ids
    fresh = [new_ids[7], *new_ids[12:]]
    assert all(b < pi.first_block_id and b not in step0_ids for b in fresh)
    assert len(set(fresh)) == 3
    assert full_ids[0] == list(step0_ids) + new_ids
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 1 and pi.lookup(PiKey("d", 0)).ref_cnt == 1
    assert pid not in out1.finished_req_ids and pid not in scheduler.requests

    # the runner echoes both rows: the parent samples its first token (the
    # prompt is complete), the pseudo row returns nothing.
    scheduler.update_from_output(out1, step_output([rid, pid], {rid: [7]}))
    assert req.num_computed_tokens == PROMPT and list(req.output_token_ids) == [7]
    assert pid not in scheduler.requests

    # step 2: plain decode; the pseudo id is retired through finished_req_ids
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens == {rid: 1}
    assert out2.scheduled_new_reqs == []
    assert pid in out2.finished_req_ids
    assert out2.scheduled_cached_reqs.resync_num_computed_tokens == set()
    scheduler.update_from_output(out2, step_output([rid], {rid: [EOS_TOKEN_ID]}))
    assert req.status == RequestStatus.FINISHED_STOPPED
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 0 and pi.lookup(PiKey("d", 0)).ref_cnt == 0
    assert km.block_pool.get_num_free_blocks() == km.block_pool.num_gpu_blocks - 1

    out3 = scheduler.schedule()
    assert rid in out3.finished_req_ids and pid not in out3.finished_req_ids


# T8-2 -----------------------------------------------------------------------


def test_one_step_off_is_the_sequential_path(make_scheduler):
    scheduler = make_scheduler(one_step=False)
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    req = add_planned_request(scheduler)
    rid = req.request_id
    outs = run_prefill(scheduler, req)

    # one step per recompute segment, one splice per step, no pseudo rows anywhere
    assert [o.num_scheduled_tokens for o in outs] == [{rid: 48}, {rid: 16}, {rid: 20}]
    assert [len(o.scheduled_new_reqs) for o in outs] == [1, 0, 0]
    assert all(not o.finished_req_ids for o in outs)
    assert scheduler.connector.spliced == [(rid, None), (rid, None)]  # no pos: boundary
    flagged = [o.scheduled_cached_reqs.resync_num_computed_tokens for o in outs]
    assert flagged == [set(), {rid}, {rid}]
    assert [o.scheduled_cached_reqs.num_computed_tokens for o in outs] == [[], [160], [240]]
    c_ids, d_ids = pi_ids(pi, "c"), pi_ids(pi, "d")
    nb1 = outs[1].scheduled_cached_reqs.new_block_ids[0][0]
    nb2 = outs[2].scheduled_cached_reqs.new_block_ids[0][0]
    assert nb1[:7] == c_ids and len(nb1) == 8
    assert nb2[:4] == d_ids and len(nb2) == 6
    assert km.get_block_ids(rid)[0][3:] == nb1 + nb2


# T8-3 -----------------------------------------------------------------------


def test_token_budget_cuts_the_walk(make_scheduler):
    """Budget 20 per step: a walk may start mid-segment; a segment that the
    budget would cut (or use up) becomes the request's own row and the rest of
    it continues next step."""
    scheduler = make_scheduler(max_num_batched_tokens=20, max_model_len=8192)
    req = add_planned_request(scheduler)
    rid = req.request_id
    outs = run_prefill(scheduler, req)
    pid = f"{rid}-ckv0"
    assert [o.num_scheduled_tokens for o in outs] == [
        {rid: 20},  # admission: 0-20
        {rid: 20},  # 20-40
        {rid: 12, pid: 8},  # pseudo 40-48, 'c' spliced, parent 160-172 (cut)
        {rid: 16, pid: 4},  # pseudo 172-176, 'd' spliced, parent 240-256 (cut)
        {rid: 4},  # 256-260
    ]
    assert [len(o.scheduled_new_reqs) for o in outs] == [1, 0, 1, 1, 0]
    assert [o.scheduled_new_reqs[0].num_computed_tokens for o in outs[2:4]] == [40, 172]
    cached = [o.scheduled_cached_reqs for o in outs]
    assert [c.num_computed_tokens for c in cached] == [[], [20], [160], [240], [256]]
    assert [c.resync_num_computed_tokens for c in cached] == [set(), set(), {rid}, {rid}, set()]
    # the pseudo id of a step is retired in the next one (and may be reused)
    assert [pid in o.finished_req_ids for o in outs] == [False, False, False, True, True]
    assert scheduler.connector.spliced == [(rid, C_START), (rid, D_START)]
    assert req.num_computed_tokens == PROMPT


def test_no_free_row_slot_falls_back_to_sequential(make_scheduler):
    """max_num_seqs == 1: no runner slot for a pseudo row, so the walk only
    splices and the segments go one per step (one step degenerates to V-COMP-2)."""
    scheduler = make_scheduler(max_num_seqs=1)
    req = add_planned_request(scheduler)
    rid = req.request_id
    outs = run_prefill(scheduler, req)
    assert [o.num_scheduled_tokens for o in outs] == [{rid: 48}, {rid: 16}, {rid: 20}]
    assert [len(o.scheduled_new_reqs) for o in outs] == [1, 0, 0]
    assert all(not o.finished_req_ids for o in outs)
    assert scheduler.connector.spliced == [(rid, C_START), (rid, D_START)]
    flagged = [o.scheduled_cached_reqs.resync_num_computed_tokens for o in outs]
    assert flagged == [set(), {rid}, {rid}]


def test_pseudo_rows_reserve_runner_slots_against_admission(make_scheduler):
    """A pseudo row occupies a runner slot this step: with max_num_seqs == 2 a
    waiting request is not admitted in the same step as the pseudo row."""
    scheduler = make_scheduler(max_num_seqs=2)
    req = add_planned_request(scheduler)
    rid = req.request_id
    out0 = scheduler.schedule()
    scheduler.update_from_output(out0, step_output([rid]))
    # index 1: a prompt that does not prefix-hit the planned request's
    other = create_requests(2, num_tokens=40, block_size=BLOCK, max_tokens=1, req_ids=["x", "other"])[1]
    scheduler.add_request(other)
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens == {rid: 20, f"{rid}-ckv0": 16}
    assert other.status == RequestStatus.WAITING
    scheduler.update_from_output(out1, step_output(list(out1.num_scheduled_tokens), {rid: [7]}))
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    out2 = scheduler.schedule()  # the slots are free again
    assert out2.num_scheduled_tokens == {"other": 40}
    assert {rid, f"{rid}-ckv0"} <= out2.finished_req_ids


def test_walk_stops_before_a_row_it_cannot_allocate(make_scheduler):
    """No free prefix block for the next recompute row: the walk stops at that
    segment (after splicing 'c') and the normal allocation loop takes over, here
    by preempting the other request; no pseudo row, and the splice is kept."""
    scheduler = make_scheduler(num_blocks=100)  # prefix pool 75, PI pool 25
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    free0 = km.block_pool.get_num_free_blocks()
    req = add_planned_request(scheduler)
    rid = req.request_id
    # Admitted right after the request: a plain prompt taking every other block.
    hog = create_requests(
        2, num_tokens=(free0 - 3) * BLOCK, block_size=BLOCK, max_tokens=4, req_ids=["x", "hog"]
    )[1]
    scheduler.add_request(hog)
    out0 = scheduler.schedule()
    assert out0.num_scheduled_tokens == {rid: 48, "hog": hog.num_tokens}
    scheduler.update_from_output(out0, step_output([rid, "hog"]))
    assert km.block_pool.get_num_free_blocks() == 0

    out1 = scheduler.schedule()
    assert scheduler.connector.spliced == [(rid, C_START)]
    assert out1.num_scheduled_tokens == {rid: 16}  # R 160-176 as the request's own row
    assert out1.scheduled_new_reqs == []
    assert hog.status == RequestStatus.PREEMPTED
    cached = out1.scheduled_cached_reqs
    assert cached.req_ids == [rid] and cached.num_computed_tokens == [160]
    assert cached.resync_num_computed_tokens == {rid}
    c_ids = pi_ids(pi, "c")
    assert cached.new_block_ids[0][0][:7] == c_ids and len(cached.new_block_ids[0][0]) == 8
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 1


# T8-5 -----------------------------------------------------------------------


def test_request_without_plan_is_untouched(make_scheduler):
    scheduler = make_scheduler()
    req = add_planned_request(scheduler, plan=None)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {req.request_id: PROMPT}
    assert len(out.scheduled_new_reqs) == 1
    assert scheduler.connector.spliced == [] and not scheduler.kv_cache_manager.pi_pool.entries
    assert out.scheduled_cached_reqs.resync_num_computed_tokens == set()


def _trace(outs):
    return [
        (
            o.num_scheduled_tokens,
            [r.req_id for r in o.scheduled_new_reqs],
            o.scheduled_cached_reqs.new_block_ids,
            o.scheduled_cached_reqs.num_computed_tokens,
            o.scheduled_cached_reqs.resync_num_computed_tokens,
            o.num_common_prefix_blocks,
            o.finished_req_ids,
        )
        for o in outs
    ]


def test_plan_without_pi_segment_matches_plain_prefill(make_scheduler):
    """V-COMP-5: a plan with no PI segment yields the same steps as no plan,
    even though the chunked prefill goes through the walk."""
    traces = []
    for plan in (None, PLAN_NO_PI):
        scheduler = make_scheduler(max_num_batched_tokens=100, max_model_len=8192)
        req = add_planned_request(scheduler, plan=plan)
        outs = run_prefill(scheduler, req)
        assert [o.num_scheduled_tokens[req.request_id] for o in outs] == [100, 100, 60]
        assert scheduler.connector.spliced == []
        traces.append(_trace(outs))
    assert traces[0] == traces[1]


def test_pseudo_row_output_is_ignored(make_scheduler):
    scheduler = make_scheduler()
    req = add_planned_request(scheduler, max_tokens=4)
    rid = req.request_id
    scheduler.update_from_output(scheduler.schedule(), step_output([rid]))
    out1 = scheduler.schedule()
    pid = f"{rid}-ckv0"
    assert pid in out1.num_scheduled_tokens
    # A runner that did not discard the pseudo row's sample: still ignored.
    scheduler.update_from_output(out1, step_output([pid, rid], {pid: [123]}))
    assert pid not in scheduler.requests
    assert list(req.output_token_ids) == []
    assert req.num_computed_tokens == PROMPT and req.status == RequestStatus.RUNNING
    assert scheduler.get_num_unfinished_requests() == 1


# Review fixes ---------------------------------------------------------------


@pytest.mark.parametrize("policy", [SchedulingPolicy.FCFS, SchedulingPolicy.PRIORITY])
def test_self_preemption_after_the_walk_caches_no_uncomputed_block(
    make_scheduler, policy
):
    """The own-row allocation fails after the walk allocated a pseudo-row block
    and the parent itself is the victim (FCFS: last in running; PRIORITY: the
    latest arrival). That block was never computed: it must not be in the
    prefix cache, so the resumed request's hit stops at step 0's 48 tokens."""
    scheduler = make_scheduler(num_blocks=100, max_num_seqs=32, max_model_len=8192)
    scheduler.policy = policy
    km = scheduler.kv_cache_manager
    # 14 decoding hogs (4 full blocks each) ahead of the parent: in step 1 each
    # decode takes a block before the parent's turn, leaving one free block,
    # which the pseudo row takes; the own row (2 blocks) cannot be allocated.
    hogs = create_requests(15, num_tokens=64, block_size=BLOCK, max_tokens=8)[1:]
    for hog in hogs:
        scheduler.add_request(hog)
    req = add_planned_request(scheduler)
    req.arrival_time = max(h.arrival_time for h in hogs) + 1.0
    rid = req.request_id
    hog_tokens = {h.request_id: [5] for h in hogs}

    out0 = scheduler.schedule()
    assert out0.num_scheduled_tokens[rid] == C_START and scheduler.running[-1] is req
    scheduler.update_from_output(
        out0, step_output(list(out0.num_scheduled_tokens), hog_tokens)
    )
    step0_ids = list(km.get_block_ids(rid)[0])
    assert km.block_pool.get_num_free_blocks() == 15

    out1 = scheduler.schedule()
    assert req.status == RequestStatus.PREEMPTED and rid in out1.preempted_req_ids
    assert rid not in out1.num_scheduled_tokens and out1.scheduled_new_reqs == []
    assert scheduler.connector.spliced == [(rid, C_START), (rid, D_START)]
    assert rid not in scheduler._ckv_resync_req_ids
    # step 0's 3 blocks and the pseudo-row block are back in the pool ...
    assert km.block_pool.get_num_free_blocks() == 1 - 1 + len(step0_ids) + 1
    # ... and only the computed ones are cached: the hit stops at 48 tokens.
    hit, num_hit, _ = km.get_computed_blocks(req)
    assert num_hit == C_START and [b.block_id for b in hit.blocks[0]] == step0_ids
    scheduler.update_from_output(
        out1, step_output(list(out1.num_scheduled_tokens), hog_tokens)
    )

    # The resumed request re-enters on that hit and finishes its prompt.
    outs = drive(scheduler, req, others=hog_tokens)
    cached, i = next(
        (o.scheduled_cached_reqs, i)
        for o in outs
        for i, r in enumerate(o.scheduled_cached_reqs.req_ids)
        if r == rid and rid in o.scheduled_cached_reqs.resumed_req_ids
    )
    assert cached.num_computed_tokens[i] == C_START
    assert cached.new_block_ids[i][0][: len(step0_ids)] == step0_ids
    assert req.num_computed_tokens == PROMPT and list(req.output_token_ids) == [7]


@pytest.mark.parametrize("one_step", [True, False])
def test_pi_segment_at_the_prompt_end_is_recomputed(make_scheduler, one_step):
    """A plan ending with a PI segment at the prompt end: the splice is refused
    (the last prompt token must be computed to be sampled), the segment becomes
    the request's own row and the prefill completes with a sampled token."""
    scheduler = make_scheduler(one_step=one_step, max_model_len=8192)
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    req = add_planned_request(
        scheduler, plan=PLAN_TRAILING_PI, num_tokens=256, max_tokens=2
    )
    rid, pid = req.request_id, f"{req.request_id}-ckv0"
    outs = drive(scheduler, req)
    if one_step:
        assert [o.num_scheduled_tokens for o in outs] == [{rid: 48}, {rid: 80, pid: 16}]
        assert outs[1].scheduled_cached_reqs.num_computed_tokens == [176]
    else:
        steps = [{rid: 48}, {rid: 16}, {rid: 80}]
        assert [o.num_scheduled_tokens for o in outs] == steps
    assert req.num_computed_tokens == 256 and list(req.output_token_ids) == [7]
    # 'd' was offered but not taken: no reference, none of its blocks in the table
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 1
    assert pi.lookup(PiKey("d", 0)).ref_cnt == 0
    ids = km.get_block_ids(rid)[0]
    assert len(ids) == 16 and not set(ids) & set(pi_ids(pi, "d"))
    out = scheduler.schedule()  # decode
    assert out.num_scheduled_tokens == {rid: 1}


def test_sliding_window_group_keeps_the_sequential_path(make_scheduler):
    """A manager that frees blocks by position could null blocks an earlier
    pseudo row of the same forward still reads: one-step stays off."""
    spec = SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=1024,
    )
    scheduler = make_scheduler(kv_cache_spec=spec)
    req = add_planned_request(scheduler)
    rid = req.request_id
    outs = run_prefill(scheduler, req)
    assert [o.num_scheduled_tokens for o in outs] == [{rid: 48}, {rid: 16}, {rid: 20}]
    assert [len(o.scheduled_new_reqs) for o in outs] == [1, 0, 0]
    assert scheduler.connector.spliced == [(rid, None), (rid, None)]  # boundary: no pos


@pytest.mark.parametrize("with_ids", [False, True])
def test_prompt_embeds_keep_the_sequential_path(make_scheduler, with_ids):
    """A pseudo NewRequestData carries token ids only: a request with prompt
    embeds (with or without ids) takes the sequential path."""
    scheduler = make_scheduler()
    if with_ids:
        req = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1)[0]
        req.prompt_embeds = torch.zeros(PROMPT, 4)
    else:
        req = Request(
            request_id="0",
            prompt_token_ids=None,
            prompt_embeds=torch.zeros(PROMPT, 4),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK, sha256),
        )
    scheduler.connector.plans[req.request_id] = PLAN
    scheduler.add_request(req)
    rid = req.request_id
    outs = run_prefill(scheduler, req)
    assert [o.num_scheduled_tokens for o in outs] == [{rid: 48}, {rid: 16}, {rid: 20}]
    assert [len(o.scheduled_new_reqs) for o in outs] == [1, 0, 0]
    assert scheduler.connector.spliced == [(rid, None), (rid, None)]


def test_prefix_hit_at_pi_start_backs_off_then_one_step(make_scheduler):
    """V-COMP-2 back-off + V-COMP-8: a local hit ending at the first PI start is
    shortened by one block (step 1 recomputes it, capped at the PI start) and the
    running pass then lays out the whole rest in one step: both PI splices and
    the middle recompute segment as a pseudo row, the tail as the own row."""
    scheduler = make_scheduler(one_step=True)
    r0 = create_requests(1, num_tokens=C_START, block_size=BLOCK, max_tokens=1, same_prompt=True, req_ids=["p"])[0]
    scheduler.add_request(r0)
    run_prefill(scheduler, r0)  # caches blocks [0, 48)
    req = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1, same_prompt=True, req_ids=["q"])[0]
    scheduler.connector.plans[req.request_id] = PLAN
    scheduler.connector.backoff_reqs.add(req.request_id)
    scheduler.add_request(req)

    out1 = scheduler.schedule()
    new = out1.scheduled_new_reqs[0]
    assert new.num_computed_tokens == C_START - BLOCK
    assert out1.num_scheduled_tokens == {req.request_id: BLOCK}
    assert scheduler.connector.spliced == []
    scheduler.update_from_output(out1, step_output([req.request_id]))

    out2 = scheduler.schedule()
    pseudo = [r for r in out2.num_scheduled_tokens if r != req.request_id]
    assert len(pseudo) == 1 and out2.num_scheduled_tokens[pseudo[0]] == 16  # [160,176)
    assert out2.num_scheduled_tokens[req.request_id] == 20  # [240,260)
    assert [pos for _, pos in scheduler.connector.spliced] == [C_START, D_START]
    assert req.num_computed_tokens == PROMPT  # scheduled tokens are counted at schedule time
    scheduler.update_from_output(out2, step_output(list(out2.num_scheduled_tokens), {req.request_id: [7]}))
    assert req.status == RequestStatus.FINISHED_STOPPED or req.num_computed_tokens >= PROMPT
