# SPDX-License-Identifier: Apache-2.0
"""ComposableKV F9: single-row one-step (V-COMP-9, docs/F9_DESIGN.md). With
CKV_ONE_STEP=flex the scheduler walks the plan at admission and lays every
recompute segment of a request into its one row, as `ckv_segments`; no pseudo
rows, prefill_steps 1 unless the token budget cuts. Scheduler-level, no GPU.

    pytest tests/v1/core/test_ckv_flex.py -q
"""

import pytest

from vllm.v1.core.ckv_pi_pool import PiKey
from vllm.v1.request import RequestStatus
from vllm.v1.core.sched.request_queue import SchedulingPolicy

from .test_ckv_onestep import (
    BLOCK,
    C_START,
    D_START,
    PLAN,
    PLAN_NO_PI,
    PLAN_TRAILING_PI,
    PROMPT,
    add_planned_request,
    drive,
    make_scheduler,  # noqa: F401  (fixture)
    pi_ids,
    run_prefill,
    step_output,
)
from .utils import create_requests

SEGMENTS = [(0, 48), (160, 16), (240, 20)]


def flex_scheduler(make_scheduler, **kwargs):
    return make_scheduler(one_step="flex", **kwargs)


# T9-1 -----------------------------------------------------------------------


@pytest.mark.parametrize("use_v2", [False, True])
def test_admission_lays_out_the_whole_plan_in_one_row(make_scheduler, use_v2):
    scheduler = flex_scheduler(make_scheduler, use_v2_model_runner=use_v2)
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    req = add_planned_request(scheduler, max_tokens=2)
    rid = req.request_id

    out0 = scheduler.schedule()
    assert out0.num_scheduled_tokens == {rid: 84}
    assert out0.total_num_scheduled_tokens == 84
    assert out0.ckv_segments == {rid: SEGMENTS}
    assert len(out0.scheduled_new_reqs) == 1  # no pseudo rows
    new = out0.scheduled_new_reqs[0]
    assert new.req_id == rid and new.num_computed_tokens == 0
    assert out0.num_common_prefix_blocks == [0]
    assert scheduler.connector.spliced == [(rid, C_START), (rid, D_START)]
    # scheduled tokens are counted at schedule time: the computed boundary is
    # the end of the last segment, not 0 + 84
    assert req.num_computed_tokens == PROMPT
    assert req.num_in_flight_tokens == 84

    # block order: [3 new][7 'c'][1 new][4 'd'][2 new]
    c_ids, d_ids = pi_ids(pi, "c"), pi_ids(pi, "d")
    ids = new.block_ids[0]
    assert len(ids) == 17
    assert ids[3:10] == c_ids and ids[11:15] == d_ids
    fresh = ids[:3] + [ids[10]] + ids[15:]
    assert all(b < pi.first_block_id for b in fresh) and len(set(fresh)) == 6
    assert km.get_block_ids(rid)[0] == ids
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 1 and pi.lookup(PiKey("d", 0)).ref_cnt == 1
    # every block of the row (recompute and spliced) is in the prefix cache
    assert km.coordinator.single_type_managers[0].num_cached_block[rid] == 16

    scheduler.update_from_output(out0, step_output([rid], {rid: [7]}))
    assert list(req.output_token_ids) == [7]

    # decode: the runner is told to reload the computed count (its own count
    # advanced by 84, not to 260)
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens == {rid: 1}
    assert out1.ckv_segments is None
    assert out1.scheduled_cached_reqs.num_computed_tokens == [PROMPT]
    assert out1.scheduled_cached_reqs.resync_num_computed_tokens == {rid}
    assert not out1.finished_req_ids
    scheduler.update_from_output(out1, step_output([rid], {rid: [8]}))
    out2 = scheduler.schedule()
    assert out2.scheduled_cached_reqs.resync_num_computed_tokens == set()


def test_token_budget_splits_the_row(make_scheduler):
    """Budget 60: admission computes [0,48) and 12 of the segment the budget
    cuts; the running pass finishes that segment, splices 'd' and computes the
    tail -- two steps, each one row."""
    scheduler = flex_scheduler(make_scheduler, max_num_batched_tokens=60, max_model_len=8192)
    req = add_planned_request(scheduler)
    rid = req.request_id
    outs = run_prefill(scheduler, req)
    assert [o.num_scheduled_tokens for o in outs] == [{rid: 60}, {rid: 24}]
    assert [o.ckv_segments for o in outs] == [
        {rid: [(0, 48), (160, 12)]},
        {rid: [(172, 4), (240, 20)]},
    ]
    assert [len(o.scheduled_new_reqs) for o in outs] == [1, 0]
    cached = outs[1].scheduled_cached_reqs
    assert cached.num_computed_tokens == [172]  # the row's first segment start
    assert cached.resync_num_computed_tokens == {rid}
    assert scheduler.connector.spliced == [(rid, C_START), (rid, D_START)]
    assert req.num_computed_tokens == PROMPT


def test_rows_mode_and_off_are_unchanged(make_scheduler):
    """The pseudo-row layout (rows) and the sequential path (off) do not see
    ckv_segments at all."""
    for one_step, steps in ((True, 2), (False, 3)):
        scheduler = make_scheduler(one_step=one_step)
        req = add_planned_request(scheduler)
        outs = run_prefill(scheduler, req)
        assert len(outs) == steps
        assert all(o.ckv_segments is None for o in outs)


def test_plan_without_pi_is_untouched(make_scheduler):
    scheduler = flex_scheduler(make_scheduler)
    req = add_planned_request(scheduler, plan=PLAN_NO_PI)
    outs = run_prefill(scheduler, req)
    assert [o.num_scheduled_tokens for o in outs] == [{req.request_id: PROMPT}]
    assert outs[0].ckv_segments is None


# T9-2 -----------------------------------------------------------------------


def test_prefix_hit_at_pi_start_backs_off_and_still_one_step(make_scheduler):
    """A local hit ending at the first PI start: the connector backs the hit off
    one block (V-COMP-2 rule, kept), and the flex admission computes that block,
    splices both PI segments and the two recompute segments in the same step."""
    scheduler = flex_scheduler(make_scheduler)
    r0 = create_requests(1, num_tokens=C_START, block_size=BLOCK, max_tokens=1, same_prompt=True, req_ids=["p"])[0]
    scheduler.add_request(r0)
    run_prefill(scheduler, r0)  # caches blocks [0, 48)
    req = create_requests(1, num_tokens=PROMPT, block_size=BLOCK, max_tokens=1, same_prompt=True, req_ids=["q"])[0]
    scheduler.connector.plans[req.request_id] = PLAN
    scheduler.connector.backoff_reqs.add(req.request_id)
    scheduler.add_request(req)

    out = scheduler.schedule()
    new = out.scheduled_new_reqs[0]
    assert new.num_computed_tokens == C_START - BLOCK
    assert out.num_scheduled_tokens == {req.request_id: BLOCK + 16 + 20}
    assert out.ckv_segments == {req.request_id: [(C_START - BLOCK, BLOCK), (160, 16), (240, 20)]}
    assert [pos for _, pos in scheduler.connector.spliced] == [C_START, D_START]
    assert req.num_computed_tokens == PROMPT


def test_pi_segment_at_the_prompt_end_is_recomputed(make_scheduler):
    """A trailing PI segment is never spliced: it is the row's last segment,
    recomputed in the same step."""
    scheduler = flex_scheduler(make_scheduler, max_model_len=8192)
    pi = scheduler.kv_cache_manager.pi_pool
    req = add_planned_request(scheduler, plan=PLAN_TRAILING_PI, num_tokens=256, max_tokens=2)
    rid = req.request_id
    outs = drive(scheduler, req)
    assert [o.num_scheduled_tokens for o in outs] == [{rid: 144}]
    assert outs[0].ckv_segments == {rid: [(0, 48), (160, 16), (176, 80)]}
    assert req.num_computed_tokens == 256 and list(req.output_token_ids) == [7]
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 1 and pi.lookup(PiKey("d", 0)).ref_cnt == 0


def test_last_segment_allocation_failure_admits_nothing(make_scheduler):
    """The row's last segment cannot be allocated at admission (forced here: the
    full-ISL reservation of the test scheduler makes it unreachable otherwise):
    the request stays waiting with no blocks and nothing cached, the PI entries
    unreferenced; it is admitted in one row once allocation works again."""
    scheduler = flex_scheduler(make_scheduler)
    km, pi = scheduler.kv_cache_manager, scheduler.kv_cache_manager.pi_pool
    free0 = km.block_pool.get_num_free_blocks()
    req = add_planned_request(scheduler, max_tokens=2)
    rid = req.request_id
    real = km.allocate_slots

    def failing(request, num_new_tokens, *a, **k):
        if request.num_computed_tokens == 240:  # the walk's last segment
            return None
        return real(request, num_new_tokens, *a, **k)

    km.allocate_slots = failing
    out1 = scheduler.schedule()
    assert rid not in out1.num_scheduled_tokens and out1.ckv_segments is None
    assert req.status == RequestStatus.WAITING and req.num_computed_tokens == 0
    assert km.block_pool.get_num_free_blocks() == free0
    assert rid not in km.coordinator.single_type_managers[0].req_to_blocks
    assert km.coordinator.single_type_managers[0].num_cached_block.get(rid, 0) == 0
    assert pi.lookup(PiKey("c", 0)).ref_cnt == 0 and pi.lookup(PiKey("d", 0)).ref_cnt == 0
    scheduler.update_from_output(out1, step_output([]))

    km.allocate_slots = real
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens == {rid: 84}
    assert out2.ckv_segments == {rid: SEGMENTS}
    assert km.block_pool.get_num_free_blocks() == free0 - 6


def test_flex_refused_only_under_pure_full_cudagraphs(make_scheduler):
    """FULL_AND_PIECEWISE (vLLM's default) runs prefill batches piecewise, so
    the sparse row is allowed; a pure FULL mode could capture it and is refused
    (sequential path)."""
    from vllm.config import CUDAGraphMode

    for mode, steps in ((CUDAGraphMode.FULL_AND_PIECEWISE, 1), (CUDAGraphMode.FULL, 3)):
        scheduler = flex_scheduler(make_scheduler)
        scheduler.vllm_config.compilation_config.cudagraph_mode = mode
        req = add_planned_request(scheduler)
        outs = run_prefill(scheduler, req)
        assert len(outs) == steps, mode
        assert (outs[0].ckv_segments is not None) == (steps == 1)
