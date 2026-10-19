# SPDX-License-Identifier: Apache-2.0
"""ComposableKV F8 (V-COMP-8), runner side on CPU: a step's pseudo-request
rows enter the persistent batch like new requests and leave it through the
next step's finished set, before a reused pseudo id is added again. Drives
the real scheduler (token budget 20, T8-3: step 2 emits `<rid>-ckv0` at 40,
step 3 retires it and emits it again at 172) and mirrors the order of each
model runner's bookkeeping (MRV1: InputBatch; MRV2: RequestState).

    pytest tests/v1/worker/test_ckv_onestep_runner.py -q
"""

import pytest
import torch

import vllm.v1.worker.gpu.buffer_utils as buffer_utils
from tests.v1.core.test_ckv_onestep import (
    BLOCK,
    PROMPT,
    FakeComposer,
    add_planned_request,
    step_output,
)
from tests.v1.core.utils import create_scheduler
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

MAX_LEN = PROMPT + BLOCK


@pytest.fixture
def steps(monkeypatch):
    """T8-3's five scheduler outputs: (outs, parent id, pseudo id)."""
    monkeypatch.setenv("CKV_PI_POOL_RATIO", "0.25")
    scheduler = create_scheduler(
        use_kv_connector=True,
        enable_prefix_caching=True,
        block_size=BLOCK,
        num_blocks=200,
        max_num_batched_tokens=20,
        max_model_len=8192,
        use_v2_model_runner=True,
    )
    scheduler.connector = FakeComposer(
        scheduler.connector, scheduler.kv_cache_manager.pi_pool, one_step=True
    )
    req = add_planned_request(scheduler)
    rid, pid = req.request_id, f"{req.request_id}-ckv0"
    outs: list[SchedulerOutput] = []
    for _ in range(5):
        out = scheduler.schedule()
        outs.append(out)
        scheduler.update_from_output(out, step_output(list(out.num_scheduled_tokens)))
    assert [o.num_scheduled_tokens for o in outs] == [
        {rid: 20},
        {rid: 20},
        {rid: 12, pid: 8},
        {rid: 16, pid: 4},
        {rid: 4},
    ]
    assert [o.finished_req_ids for o in outs] == [set(), set(), set(), {pid}, {pid}]
    new_ids = [[r.req_id for r in o.scheduled_new_reqs] for o in outs]
    assert new_ids == [[rid], [], [pid], [pid], []]
    return outs, rid, pid


class Mrv1Batch:
    """The row placement of GPUModelRunner._update_states, in its order:
    finished and unscheduled rows out, cached rows updated, new rows in."""

    def __init__(self, max_num_reqs: int):
        self.batch = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=MAX_LEN,
            max_num_batched_tokens=64,
            device=torch.device("cpu"),
            vocab_size=1024,
            block_sizes=[BLOCK],
            kernel_block_sizes=[BLOCK],
            max_num_blocks_per_req=[cdiv(MAX_LEN, BLOCK)],
        )
        self.requests: dict[str, CachedRequestState] = {}

    def update(self, out: SchedulerOutput) -> None:
        for rid in out.finished_req_ids:
            self.requests.pop(rid, None)
            self.batch.remove_request(rid)
        for rid in set(self.batch.req_id_to_index) - set(out.num_scheduled_tokens):
            self.batch.remove_request(rid)
        to_add = []
        cached = out.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            state = self.requests[rid]
            state.num_computed_tokens = cached.num_computed_tokens[i]
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                for ids, new in zip(state.block_ids, new_block_ids):
                    ids.extend(new)
            idx = self.batch.req_id_to_index.get(rid)
            if idx is None:
                to_add.append(state)
                continue
            self.batch.num_computed_tokens_cpu[idx] = state.num_computed_tokens
            if new_block_ids is not None:
                self.batch.block_table.append_row(new_block_ids, idx)
        for new in out.scheduled_new_reqs:
            state = CachedRequestState(
                req_id=new.req_id,
                prompt_token_ids=new.prompt_token_ids,
                mm_features=new.mm_features,
                sampling_params=new.sampling_params,
                pooling_params=new.pooling_params,
                generator=None,
                block_ids=tuple(list(ids) for ids in new.block_ids),
                num_computed_tokens=new.num_computed_tokens,
                output_token_ids=[],
                lora_request=new.lora_request,
            )
            self.requests[new.req_id] = state
            to_add.append(state)
        for state in to_add:
            self.batch.add_request(state)
        self.batch.condense()
        self.batch.refresh_metadata()

    def row(self, rid: str) -> tuple[int, list[int]]:
        """(num_computed_tokens, block table row) of a request."""
        idx = self.batch.req_id_to_index[rid]
        table = self.batch.block_table[0]
        n = table.num_blocks_per_row[idx]
        start = int(self.batch.num_computed_tokens_cpu[idx])
        return start, table.block_table.np[idx, :n].tolist()


def test_mrv1_input_batch_places_pseudo_rows(steps):
    outs, rid, pid = steps
    runner = Mrv1Batch(max_num_reqs=4)
    for out in outs[:2]:
        runner.update(out)
    assert runner.batch.num_reqs == 1

    for k, (start, parent_start) in {2: (40, 160), 3: (172, 240)}.items():
        runner.update(outs[k])
        pseudo = outs[k].scheduled_new_reqs[0]
        batch = runner.batch
        # one row each, the reused id in step 3 placed after the old row left
        assert batch.num_reqs == 2
        assert sorted(r for r in batch.req_ids if r is not None) == sorted([rid, pid])
        assert len(set(batch.req_id_to_index.values())) == 2
        # the pseudo row starts at its segment and shares the parent's table
        assert runner.row(pid) == (start, list(pseudo.block_ids[0]))
        assert runner.row(rid) == (parent_start, list(pseudo.block_ids[0]))
        assert pseudo.num_computed_tokens == start

    runner.update(outs[4])
    assert runner.batch.num_reqs == 1 and pid not in runner.batch.req_id_to_index


class Mrv2States:
    """The row placement of gpu/model_runner.py (finish_requests, then
    add_requests, which removes an id before adding it)."""

    def __init__(self, max_num_reqs: int):
        self.states = RequestState(
            max_num_reqs=max_num_reqs,
            max_model_len=MAX_LEN,
            max_num_batched_tokens=64,
            num_speculative_steps=0,
            vocab_size=1024,
            device=torch.device("cpu"),
        )

    def update(self, out: SchedulerOutput) -> None:
        for rid in out.finished_req_ids | (out.preempted_req_ids or set()):
            self.states.remove_request(rid)
        for new in out.scheduled_new_reqs:
            assert new.prompt_token_ids is not None
            assert new.prefill_token_ids is not None
            assert new.sampling_params is not None
            self.states.remove_request(new.req_id)
            self.states.add_request(
                req_id=new.req_id,
                prompt_len=len(new.prompt_token_ids),
                all_token_ids=new.prefill_token_ids,
                num_computed_tokens=new.num_computed_tokens,
                max_tokens=new.sampling_params.max_tokens,
            )


def test_mrv2_request_state_places_pseudo_rows(steps, monkeypatch):
    # UVA views need an accelerator; the staged device writes are not applied here.
    monkeypatch.setattr(
        buffer_utils, "get_accelerator_view_from_cpu_tensor", lambda t: t
    )
    outs, rid, pid = steps
    runner = Mrv2States(max_num_reqs=4)
    states = runner.states
    for out in outs[:2]:
        runner.update(out)
    assert states.num_reqs == 1 and len(states.free_indices) == 3

    for k, start in {2: 40, 3: 172}.items():
        runner.update(outs[k])
        pseudo = outs[k].scheduled_new_reqs[0]
        # the step-2 slot was freed by step 3's finished set and is reused
        assert states.num_reqs == 2 and len(states.free_indices) == 2
        assert len(set(states.req_id_to_index.values())) == 2
        idx = states.req_id_to_index[pid]
        assert states.index_to_req_id[idx] == pid
        assert states.prompt_len.np[idx] == PROMPT
        assert states.prefill_len.np[idx] == len(pseudo.prefill_token_ids) == PROMPT
        assert pseudo.num_computed_tokens == start
        assert states.num_computed_prefill_tokens[idx] == start
        assert states.num_computed_tokens_np[idx] == start

    runner.update(outs[4])
    assert states.num_reqs == 1 and len(states.free_indices) == 3
    assert pid not in states.req_id_to_index and rid in states.req_id_to_index
