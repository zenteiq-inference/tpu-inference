# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""X-data-parallel-rank on the SPMD DPScheduler path.

Covers, from the HTTP header value down to the rank a request lands on:

* ``get_dp_rank_hint``: reading the hint back (pure).
* ``install_input_processor_hook``: the REAL vLLM ``InputProcessor`` behind
  the REAL tpu-inference DP override, at DP 2, 3, 4 and 8. Stock vLLM accepts
  rank 0 (silently dropped) and answers 400 for rank >= 1; with the hook every
  rank in ``[0, dp)`` is accepted and carried, anything else is a 400 with the
  real range.
* the msgpack hop between API server and EngineCore, which is what carries the
  hint to the scheduler.
* ``DPScheduler`` routing with mocked IPC, and end to end over real forked
  rank workers.

CPU only: ``VLLM_TARGET_DEVICE=empty JAX_PLATFORMS=cpu``.
"""
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from tests.core import dp_testlib as lib
from tpu_inference.core.sched import dp_rank_hint as hint
from tpu_inference.core.sched.dp_scheduler import DPScheduler, SchedulerCommand

try:  # newer vLLM raises this (a 400) instead of a plain ValueError
    from vllm.exceptions import VLLMValidationError
    REJECTED = (ValueError, VLLMValidationError)
except ImportError:
    REJECTED = (ValueError, )

DP_SIZES = [2, 3, 4, 8]
PROMPT = {"prompt_token_ids": list(range(1, 40))}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return lib.write_tiny_llama(tmp_path_factory.mktemp("tiny-llama"))


@pytest.fixture
def hook_installed():
    """Install the hook for the test and always restore stock vLLM after."""
    assert hint.install_input_processor_hook()
    try:
        yield
    finally:
        hint.uninstall_input_processor_hook()


def _processor(model_dir, dp):
    """InputProcessor built from a config that went through the real
    tpu-inference DP override, exactly as in an API-server process."""
    vc = lib.make_vllm_config(model_dir, parallel_dp=dp)
    assert vc.parallel_config.data_parallel_size == dp
    lib.apply_tpu_sharding(
        vc)  # pins data_parallel_size to 1, selects DPScheduler
    assert vc.parallel_config.data_parallel_size == 1
    assert vc.sharding_config.total_dp_size == dp
    return vc, InputProcessor(vc)


def _process(ip, rank, params=None, positional=False, rid="r"):
    params = params or SamplingParams(max_tokens=4)
    if positional:  # data_parallel_rank is the 10th positional parameter
        return ip.process_inputs(rid, PROMPT, params, ("generate", ), None,
                                 None, None, None, 0, rank)
    return ip.process_inputs(rid,
                             PROMPT,
                             params,
                             supported_tasks=("generate", ),
                             data_parallel_rank=rank)


# ---------------------------------------------------------------------------
# 1. get_dp_rank_hint (pure)
# ---------------------------------------------------------------------------
def _req(extra_args=None, pooling=None):
    return SimpleNamespace(
        sampling_params=None if extra_args is None else SimpleNamespace(
            extra_args=extra_args),
        pooling_params=pooling)


class TestGetDpRankHint:

    @pytest.mark.parametrize("dp", DP_SIZES)
    def test_every_rank_in_range(self, dp):
        for rank in range(dp):
            assert hint.get_dp_rank_hint(_req({"dp_rank": rank}), dp) == rank

    @pytest.mark.parametrize("extra",
                             [None, {}, {
                                 "other": 1
                             }, {
                                 "dp_rank": None
                             }])
    def test_no_hint(self, extra):
        assert hint.get_dp_rank_hint(_req(extra), 4) is None

    @pytest.mark.parametrize("bad", [-1, 4, 99, True, False, 1.0, "1", [1]])
    def test_unusable_values_are_ignored_not_raised(self, bad):
        assert hint.get_dp_rank_hint(_req({"dp_rank": bad}), 4) is None

    def test_no_params_objects(self):
        assert hint.get_dp_rank_hint(SimpleNamespace(), 2) is None
        assert hint.get_dp_rank_hint(MagicMock(spec=Request), 2) is None

    def test_pooling_params_channel(self):
        pooling = SimpleNamespace(extra_kwargs={"dp_rank": 1})
        assert hint.get_dp_rank_hint(_req(None, pooling), 2) == 1

    def test_non_dict_extras_are_ignored(self):
        assert hint.get_dp_rank_hint(MagicMock(), 2) is None


class TestAttachHint:

    def test_sampling_params_keep_other_extras(self):
        params = SimpleNamespace(extra_args={"keep": "me"})
        hint._attach_hint(SimpleNamespace(sampling_params=params), 3)
        assert params.extra_args == {"keep": "me", "dp_rank": 3}

    def test_sampling_params_without_extras(self):
        params = SimpleNamespace(extra_args=None)
        hint._attach_hint(SimpleNamespace(sampling_params=params), 0)
        assert params.extra_args == {"dp_rank": 0}

    def test_pooling_params(self):
        pooling = SimpleNamespace(extra_kwargs=None)
        hint._attach_hint(
            SimpleNamespace(sampling_params=None, pooling_params=pooling), 1)
        assert pooling.extra_kwargs == {"dp_rank": 1}


# ---------------------------------------------------------------------------
# 2. the InputProcessor hook against real vLLM, DP 2/3/4/8
# ---------------------------------------------------------------------------
class TestInputProcessorHook:

    @pytest.mark.parametrize("dp", DP_SIZES)
    def test_stock_vllm_rejects_every_rank_above_0(self, model_dir, dp):
        """The bug, reproduced on the real code: without the hook rank 0 is
        accepted and dropped, rank >= 1 is a 400 ("out of range [0, 1)")."""
        _, ip = _processor(model_dir, dp)
        hint.uninstall_input_processor_hook()
        ecr = _process(ip, 0)
        assert ecr.data_parallel_rank == 0  # accepted ...
        assert not ecr.sampling_params.extra_args  # ... and nothing carries it
        for rank in range(1, dp):
            with pytest.raises(REJECTED, match=r"out of range \[0, 1\)"):
                _process(ip, rank)

    @pytest.mark.parametrize("dp", DP_SIZES)
    @pytest.mark.parametrize("positional", [False, True])
    def test_every_rank_is_accepted_and_carried(self, model_dir,
                                                hook_installed, dp,
                                                positional):
        _, ip = _processor(model_dir, dp)
        for rank in range(dp):
            ecr = _process(ip, rank, positional=positional)
            assert ecr.sampling_params.extra_args == {"dp_rank": rank}
            # vLLM's own field stays None: DPLB clients never see a hint they
            # cannot honour
            assert ecr.data_parallel_rank is None

    @pytest.mark.parametrize("dp", DP_SIZES)
    def test_out_of_range_is_400_with_the_real_range(self, model_dir,
                                                     hook_installed, dp):
        _, ip = _processor(model_dir, dp)
        for rank in (dp, dp + 5, -1):
            with pytest.raises(
                    REJECTED,
                    match=rf"data_parallel_rank {rank} is out of range "
                    rf"\[0, {dp}\)"):
                _process(ip, rank)

    def test_no_header_leaves_the_request_untouched(self, model_dir,
                                                    hook_installed):
        _, ip = _processor(model_dir, 2)
        ecr = _process(ip, None)
        assert ecr.data_parallel_rank is None
        assert not ecr.sampling_params.extra_args

    def test_header_overrides_xargs_and_keeps_other_extras(
            self, model_dir, hook_installed):
        _, ip = _processor(model_dir, 2)
        params = SamplingParams(max_tokens=4,
                                extra_args={
                                    "dp_rank": 0,
                                    "other": 1
                                })
        ecr = _process(ip, 1, params=params)
        assert ecr.sampling_params.extra_args == {"dp_rank": 1, "other": 1}
        # the caller's object is not mutated (vLLM clones params)
        assert params.extra_args == {"dp_rank": 0, "other": 1}

    def test_xargs_alone_still_passes_through(self, model_dir, hook_installed):
        _, ip = _processor(model_dir, 2)
        params = SamplingParams(max_tokens=4, extra_args={"dp_rank": 1})
        ecr = _process(ip, None, params=params)
        assert ecr.sampling_params.extra_args == {"dp_rank": 1}

    def test_steps_aside_when_vllm_handles_dp_itself(self, model_dir,
                                                     hook_installed):
        """vLLM-native DP (MPMD): parallel_config keeps the real DP size and
        no DPScheduler is configured, so vLLM validates and honours the rank."""
        vc = lib.make_vllm_config(model_dir, parallel_dp=2)
        vc.sharding_config = SimpleNamespace(total_dp_size=1)
        ip = InputProcessor(vc)
        ecr = _process(ip, 1)
        assert ecr.data_parallel_rank == 1
        assert not ecr.sampling_params.extra_args
        with pytest.raises(REJECTED, match=r"out of range \[0, 2\)"):
            _process(ip, 2)

    def test_steps_aside_for_a_plain_scheduler(self, model_dir,
                                               hook_installed):
        vc = lib.make_vllm_config(model_dir)
        vc.sharding_config = SimpleNamespace(total_dp_size=4)  # no DPScheduler
        assert hint.dp_scheduler_rank_count(vc) == 1
        ip = InputProcessor(vc)
        with pytest.raises(REJECTED, match=r"out of range \[0, 1\)"):
            _process(ip, 1)

    def test_install_is_idempotent(self, hook_installed):
        first = InputProcessor.process_inputs
        assert hint.install_input_processor_hook()
        assert InputProcessor.process_inputs is first

    def test_update_vllm_config_installs_the_hook(self, model_dir):
        hint.uninstall_input_processor_hook()
        try:
            _processor(model_dir, 2)
            assert getattr(InputProcessor.process_inputs, hint._HOOK_MARKER,
                           False)
        finally:
            hint.uninstall_input_processor_hook()

    def test_unrelated_call_errors_are_vllms_own(self, model_dir,
                                                 hook_installed):
        _, ip = _processor(model_dir, 2)
        with pytest.raises(TypeError):
            ip.process_inputs("r",
                              PROMPT,
                              SamplingParams(), ("generate", ),
                              not_a_parameter=1)


# ---------------------------------------------------------------------------
# 3. the hop to the scheduler, then DPScheduler routing
# ---------------------------------------------------------------------------
def _header_parser():
    """vLLM's own header parser, wherever this vLLM keeps it."""
    for module, name in (
        ("vllm.entrypoints.openai.engine.serving", "OpenAIServing"),
        ("vllm.entrypoints.generate.base.serving", "GenerateBaseServing"),
    ):
        try:
            cls = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
        if hasattr(cls, "_get_data_parallel_rank"):
            return cls._get_data_parallel_rank
    pytest.skip("this vLLM has no X-data-parallel-rank header parser")


@pytest.mark.parametrize("dp", [2, 8])
def test_header_text_to_accepted_rank(model_dir, hook_installed, dp):
    """The header's text through vLLM's own parser, then the hook: every
    rank's string is accepted, out-of-range text is a 400, junk text is
    silently no hint (vLLM's parser returns None for it)."""
    parse = _header_parser()
    _, ip = _processor(model_dir, dp)

    def request_with(value):
        headers = {} if value is None else {"X-data-parallel-rank": value}
        return SimpleNamespace(headers=headers)

    for rank in range(dp):
        parsed = parse(request_with(str(rank)))
        assert parsed == rank
        ecr = _process(ip, parsed)
        assert ecr.sampling_params.extra_args == {"dp_rank": rank}
    with pytest.raises(REJECTED, match=rf"out of range \[0, {dp}\)"):
        _process(ip, parse(request_with(str(dp))))
    for junk in (None, "", "one", "1.5"):
        assert parse(request_with(junk)) is None
        assert not _process(ip, parse(
            request_with(junk))).sampling_params.extra_args


def test_hint_survives_the_engine_core_msgpack_hop(model_dir, hook_installed):
    """API server -> (msgpack over ZMQ) -> EngineCore -> Request."""
    _, ip = _processor(model_dir, 4)
    ecr = _process(ip, 3)
    decoded = MsgpackDecoder(EngineCoreRequest).decode(
        MsgpackEncoder().encode(ecr))
    request = Request.from_engine_core_request(decoded, lib._block_hasher())
    assert request.sampling_params.extra_args == {"dp_rank": 3}
    assert hint.get_dp_rank_hint(request, 4) == 3


def _mock_config(dp, caching=True):
    cfg = MagicMock()
    cfg.sharding_config = MagicMock()
    cfg.sharding_config.total_dp_size = dp
    cfg.scheduler_config = MagicMock()
    cfg.scheduler_config._original_scheduler_cls = Scheduler
    cfg.scheduler_config.max_num_seqs = 8
    cfg.scheduler_config.max_num_batched_tokens = 1024
    cfg.scheduler_config.async_scheduling = False
    cfg.cache_config = MagicMock()
    cfg.cache_config.enable_prefix_caching = caching
    cfg.cache_config.prefix_caching_hash_algo = "sha256"
    cfg.cache_config.mamba_num_blocks = None
    return cfg


def _mocked_scheduler(dp, monkeypatch, caching=True):
    monkeypatch.delenv("DP_SCHED_ROUTING", raising=False)
    ctx = MagicMock()
    ctx.Pipe = MagicMock(return_value=(MagicMock(), MagicMock()))
    ctx.Process = MagicMock(return_value=MagicMock())
    kv_cache_config = MagicMock(spec=KVCacheConfig)
    kv_cache_config.num_blocks = 100 * dp
    with patch("tpu_inference.core.sched.dp_scheduler._scheduler_worker_process"
               ), patch("multiprocessing.get_context", return_value=ctx), \
            patch("vllm.v1.core.kv_cache_utils.init_none_hash"):
        return DPScheduler(vllm_config=_mock_config(dp, caching),
                           kv_cache_config=kv_cache_config,
                           structured_output_manager=MagicMock(),
                           block_size=16,
                           log_stats=False)


def _plain():
    request = MagicMock(spec=Request)
    request.num_tokens = 100  # read by the least-loaded picker
    request.sampling_params = None
    return request


def _hinted(rank):
    request = _plain()
    request.sampling_params = SimpleNamespace(extra_args={"dp_rank": rank})
    return request


def _route(scheduler, request):
    """Entry point that applies hints: ``_pick_rank_for_request`` where the
    tree has a routing-policy dispatcher, else ``_find_best_rank_for_request``."""
    pick = getattr(scheduler, "_pick_rank_for_request",
                   scheduler._find_best_rank_for_request)
    return pick(request)


def _fake_rank_answers(scheduler):
    """Answer every per-rank query the pickers send: no prefix hit anywhere,
    rank 0 has 40 pending prefill tokens and rank 1+ has 10."""
    C = SchedulerCommand
    scheduler._send_command = MagicMock()

    def answer(rank, command=None):
        return {
            C.PROBE_COMPUTED_BLOCKS: 0,
            C.GET_PENDING_PREFILL_TOKENS: 40 if rank == 0 else 10,
            C.GET_REQUEST_COUNTS: (1, 0),
            getattr(C, "GET_MIN_REMAINING_OUTPUT", None): 5,
        }[command]

    scheduler._get_result = MagicMock(side_effect=answer)


class TestDPSchedulerRouting:

    @pytest.mark.parametrize("dp", DP_SIZES)
    def test_hint_wins_without_any_rank_query(self, dp, monkeypatch):
        s = _mocked_scheduler(dp, monkeypatch)
        s._send_command = MagicMock()
        s._get_result = MagicMock()
        for rank in range(dp):
            assert _route(s, _hinted(rank)) == rank
        s._send_command.assert_not_called()
        s._get_result.assert_not_called()

    @pytest.mark.parametrize("bad", [-1, 2, 7, "0", True, None])
    def test_unusable_hint_falls_back_to_the_picker(self, bad, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch)
        _fake_rank_answers(s)
        assert _route(s, _hinted(bad)) == 1  # rank 1 has fewer pending tokens
        sent = [c.args[1] for c in s._send_command.call_args_list]
        assert SchedulerCommand.GET_PENDING_PREFILL_TOKENS in sent

    def test_no_hint_is_unchanged(self, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch)
        _fake_rank_answers(s)
        assert _route(s, _plain()) == 1

    def test_route_and_forward_uses_the_hint(self, monkeypatch):
        s = _mocked_scheduler(4, monkeypatch)
        s._send_command = MagicMock()
        s._get_result = MagicMock(return_value=None)
        request = _hinted(3)
        request.request_id = "q"
        s._route_and_forward_request(request)
        assert s.assigned_dp_rank["q"] == 3
        assert s._send_command.call_args.args[:2] == (
            3, SchedulerCommand.ADD_REQUEST)


# ---------------------------------------------------------------------------
# 4. end to end: real DPScheduler, real forked rank workers, real vLLM schedulers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dp", [2, 3, 4, 8])
def test_hinted_requests_land_on_their_rank_end_to_end(model_dir, monkeypatch,
                                                       dp):
    monkeypatch.delenv("DP_SCHED_ROUTING", raising=False)
    _, s = lib.make_real_dp_scheduler(model_dir, dp)
    try:
        s._batch_prefills = False
        # all hinted at ONE rank (not the one the picker would choose), then
        # the same traffic hinted round-robin, then unhinted
        target = dp - 1
        for i in range(6):
            s.add_request(
                lib.make_request(f"pin{i}",
                                 seed=i,
                                 extra_args={"dp_rank": target}))
        for i in range(dp * 2):
            s.add_request(
                lib.make_request(f"rr{i}",
                                 seed=100 + i,
                                 extra_args={"dp_rank": i % dp}))
        for i in range(4):
            s.add_request(lib.make_request(f"free{i}", seed=200 + i))
        for i in range(6):
            assert s.assigned_dp_rank[f"pin{i}"] == target
        for i in range(dp * 2):
            assert s.assigned_dp_rank[f"rr{i}"] == i % dp
        assert all(s.assigned_dp_rank[f"free{i}"] in range(dp)
                   for i in range(4))

        # the request really is held by that rank's vLLM scheduler
        lib.step(s)
        running_plus_waiting = []
        for rank in range(dp):
            s._send_command(rank, SchedulerCommand.GET_REQUEST_COUNTS)
            running, waiting = s._get_result(
                rank, SchedulerCommand.GET_REQUEST_COUNTS)
            running_plus_waiting.append(running + waiting)
        pinned_to_rank = sum(1 for r in s.assigned_dp_rank.values()
                             if r == target)
        assert running_plus_waiting[target] == pinned_to_rank
        assert sum(running_plus_waiting) == len(s.assigned_dp_rank)
    finally:
        s.shutdown()


def test_header_to_rank_chain(model_dir, monkeypatch, hook_installed):
    """X-data-parallel-rank value -> real InputProcessor -> msgpack -> Request
    -> real DPScheduler -> the rank that runs it."""
    monkeypatch.delenv("DP_SCHED_ROUTING", raising=False)
    vc, ip = _processor(model_dir, 2)
    _, s = lib.make_real_dp_scheduler(model_dir, 2, vllm_config=vc)
    try:
        s._batch_prefills = False
        for rid, header_rank in (("a", 1), ("b", 1), ("c", 1), ("d", 0)):
            ecr = ip.process_inputs(rid,
                                    {"prompt_token_ids": list(range(1, 200))},
                                    SamplingParams(max_tokens=8),
                                    supported_tasks=("generate", ),
                                    data_parallel_rank=header_rank)
            ecr = MsgpackDecoder(EngineCoreRequest).decode(
                MsgpackEncoder().encode(ecr))
            s.add_request(
                Request.from_engine_core_request(ecr, lib._block_hasher()))
        assert s.assigned_dp_rank == {"a": 1, "b": 1, "c": 1, "d": 0}
    finally:
        s.shutdown()
