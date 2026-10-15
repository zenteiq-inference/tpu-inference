# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Load-aware DP rank routing (``DP_SCHED_ROUTING=load``).

* the pure picker (``dp_rank_picker``);
* ``DPScheduler`` wiring with mocked IPC (the default stays ``legacy`` and
  never sends the new command);
* real forked rank workers: ``GET_RANK_LOAD`` and the skew mechanism;
* the REAL vLLM schedulers: ``GET_RANK_LOAD`` agrees with vLLM's own
  ``make_stats``, and the prefix-sticky skew of the legacy picker is
  reproduced and bounded by the load-aware one.

CPU only: ``VLLM_TARGET_DEVICE=empty JAX_PLATFORMS=cpu``.
"""
import os
import random
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import Request

from tests.core import dp_testlib as lib
from tpu_inference.core.sched import dp_rank_picker as rp
from tpu_inference.core.sched import dp_scheduler as dps

BS = 128  # gemma4-dsdg block size
NB = 2223  # gemma4-dsdg blocks per rank


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DP_SCHED_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return lib.write_tiny_llama(tmp_path_factory.mktemp("tiny-llama"))


def st(running=0, waiting=0, pending=0, kv=0.0, nb=NB, max_seqs=128, routed=0):
    return dict(running=running,
                waiting=waiting,
                pending_prefill_tokens=pending,
                kv_usage=kv,
                num_blocks=nb,
                max_seqs=max_seqs,
                routed_total=routed)


# ===========================================================================
# 1. the pure picker
# ===========================================================================
class TestPurePicker:

    def test_no_hit_picks_less_loaded_rank_not_rank0(self):
        state = [st(running=60, kv=0.50), st(running=40, kv=0.35)]
        assert rp.pick_rank([0, 0], state, BS) == 1

    def test_no_hit_idle_ranks_alternate(self):
        state = [st(), st()]
        picks = [rp.pick_rank([0, 0], state, BS) for _ in range(40)]
        assert 15 <= picks.count(0) <= 25, picks  # rotates, never all rank 0

    def test_tie_break_prefers_fewest_routed_before_rotation(self):
        state = [st(routed=10), st(routed=3)]
        for _ in range(10):
            assert rp.pick_rank([0, 0], state, BS) == 1

    @pytest.mark.parametrize("n", [2, 3, 4, 8])
    def test_idle_ranks_rotate_at_every_dp_size(self, n):
        state = [st() for _ in range(n)]
        picks = {rp.pick_rank([0] * n, state, BS) for _ in range(n * 10)}
        assert picks == set(range(n))

    def test_affinity_kept_when_balanced(self):
        state = [st(running=30, kv=0.40), st(running=30, kv=0.40)]
        info = {}
        assert rp.pick_rank([0, 5 * BS], state, BS, info=info) == 1
        assert info["reason"] in ("affinity", "agree")

    def test_affinity_kept_when_hit_rank_slightly_hotter(self):
        state = [st(running=30, kv=0.40), st(running=36, kv=0.45)]
        info = {}
        assert rp.pick_rank([0, 5 * BS], state, BS, info=info) == 1
        assert info["reason"] == "affinity"

    def test_small_advantage_ignored(self):
        state = [st(running=20, kv=0.30), st(running=22, kv=0.32)]
        info = {}
        assert rp.pick_rank([0, 1 * BS], state, BS, info=info) == 0
        assert info["reason"] == "small_advantage"

    def test_advantage_is_over_the_cooler_rank_not_over_zero(self):
        state = [st(running=20, kv=0.30), st(running=30, kv=0.40)]
        info = {}
        assert rp.pick_rank([9 * BS, 10 * BS], state, BS, info=info) == 0
        assert info["reason"] == "small_advantage"

    def test_equal_hits_use_load_not_rank0(self):
        state = [st(running=50, kv=0.50), st(running=20, kv=0.30)]
        assert rp.pick_rank([4 * BS, 4 * BS], state, BS) == 1

    def test_kv_gap_overrides_affinity(self):
        state = [st(running=30, kv=0.30), st(running=30, kv=0.45)]
        info = {}
        assert rp.pick_rank([0, 8 * BS], state, BS, info=info) == 0
        assert info["reason"] == "kv_gap"

    def test_kv_gap_boundary_is_inclusive(self):
        state = [st(running=30, kv=0.30), st(running=30, kv=0.40)]  # gap 0.10
        info = {}
        assert rp.pick_rank([0, 8 * BS], state, BS, info=info) == 1
        assert info["reason"] == "affinity"

    def test_queue_gap_overrides_affinity(self):
        state = [st(running=20, kv=0.30), st(running=29, kv=0.30)]  # gap 9 > 8
        info = {}
        assert rp.pick_rank([0, 8 * BS], state, BS, info=info) == 0
        assert info["reason"] == "queue_gap"

    def test_queue_gap_boundary_is_inclusive(self):
        state = [st(running=20, kv=0.30), st(running=28, kv=0.30)]  # gap 8
        assert rp.pick_rank([0, 8 * BS], state, BS) == 1

    def test_waiting_counts_in_the_queue_gap(self):
        state = [
            st(running=40, waiting=0, kv=0.40),
            st(running=40, waiting=12, kv=0.40)
        ]
        info = {}
        assert rp.pick_rank([0, 8 * BS], state, BS, info=info) == 0
        assert info["reason"] == "queue_gap"

    def test_pending_gap_overrides_affinity(self):
        state = [
            st(running=20, kv=0.30, pending=0),
            st(running=20, kv=0.30, pending=40000)
        ]
        # num_blocks=0 disables the KV projection so only the pending gap decides
        state[0]["num_blocks"] = state[1]["num_blocks"] = 0
        info = {}
        assert rp.pick_rank([0, 8 * BS], state, BS, info=info) == 0
        assert info["reason"] == "pending_gap"

    def test_saturated_hit_rank_overrides_even_inside_the_gaps(self):
        state = [
            st(running=121, kv=0.60, max_seqs=128),
            st(running=127, waiting=1, kv=0.62, max_seqs=128)
        ]
        info = {}
        assert rp.pick_rank([0, 8 * BS], state, BS, info=info) == 0
        assert info["reason"] == "hit_rank_saturated"

    def test_both_saturated_keeps_affinity(self):
        state = [
            st(running=128, waiting=3, kv=0.95),
            st(running=128, waiting=3, kv=0.95)
        ]
        info = {}
        assert rp.pick_rank([0, 8 * BS], state, BS, info=info) == 1
        assert info["reason"] in ("affinity", "agree")

    def test_kv_saturation_demotes_a_full_rank_in_load_order(self):
        state = [st(running=10, kv=0.95), st(running=60, kv=0.30)]
        assert rp.pick_rank([0, 0], state, BS) == 1

    def test_queued_prefill_projects_onto_kv_so_a_burst_spreads(self):
        state = [st(), st()]
        counts = [0, 0]
        for _ in range(60):
            r = rp.pick_rank([0, 0], state, BS)
            counts[r] += 1
            state[r]["waiting"] += 1
            state[r]["pending_prefill_tokens"] += 1275
            state[r]["routed_total"] += 1
        assert abs(counts[0] - counts[1]) <= 2, counts

    @pytest.mark.parametrize("n", [3, 4, 8])
    def test_hot_rank_is_avoided_at_every_dp_size(self, n):
        """One rank at 0.92 KV holds the only prefix hit; the rest are idle."""
        state = [st(running=100, kv=0.92)] + [st() for _ in range(n - 1)]
        hits = [8 * BS] + [0] * (n - 1)
        info = {}
        rank = rp.pick_rank(hits, state, BS, info=info)
        assert rank != 0, info
        assert info["reason"] in ("kv_gap", "queue_gap", "hit_rank_saturated")

    def test_short_hits_list_is_padded(self):
        assert rp.pick_rank([0], [st(), st(running=9)], BS) == 0

    def test_empty_state_raises(self):
        with pytest.raises(ValueError):
            rp.pick_rank([], [], BS)

    def test_missing_keys_default_to_zero(self):
        assert rp.pick_rank([0, 0], [{}, {"running": 5}], BS) == 0

    def test_info_out_param(self):
        info = {}
        rp.pick_rank([0, 3 * BS], [st(), st()], BS, info=info)
        assert set(info) >= {"reason", "rank", "hit_rank", "load_rank", "hits"}

    def test_config_from_env(self):
        cfg = rp.PickerConfig.from_env({
            "DP_SCHED_AFFINITY_MIN_BLOCKS": "4",
            "DP_SCHED_AFFINITY_MAX_KV_GAP": "0.2",
            "DP_SCHED_AFFINITY_MAX_QUEUE_GAP": "16",
        })
        assert (cfg.min_affinity_blocks, cfg.max_kv_gap,
                cfg.max_queue_gap) == (4, 0.2, 16)
        assert cfg.bad_env == ()

    def test_config_bad_env_falls_back_and_is_reported(self):
        cfg = rp.PickerConfig.from_env({
            "DP_SCHED_AFFINITY_MIN_BLOCKS": "two",
            "DP_SCHED_KV_BUCKET": "",
        })
        assert cfg.min_affinity_blocks == 2 and cfg.kv_bucket == 0.05
        assert cfg.bad_env == ("DP_SCHED_AFFINITY_MIN_BLOCKS", )

    @pytest.mark.parametrize("name,raw", [
        ("DP_SCHED_AFFINITY_MAX_KV_GAP", "nan"),
        ("DP_SCHED_KV_SATURATION", "inf"),
        ("DP_SCHED_KV_BUCKET", "1e-320"),
        ("DP_SCHED_AFFINITY_MAX_PENDING_GAP_TOKENS", "10000000000"),
    ])
    def test_config_rejects_values_that_would_disable_or_crash_the_picker(
            self, name, raw):
        cfg = rp.PickerConfig.from_env({name: raw})
        assert cfg.bad_env == (name, )
        assert cfg == rp.PickerConfig(bad_env=(name, ))

    def test_custom_min_blocks_changes_the_decision(self):
        state = [st(running=20, kv=0.30), st(running=22, kv=0.32)]
        hit = 3 * BS
        assert rp.pick_rank([0, hit], state, BS) == 1  # 3 >= 2
        cfg = rp.PickerConfig(min_affinity_blocks=4)
        assert rp.pick_rank([0, hit], state, BS, cfg=cfg) == 0  # 3 < 4

    def test_random_states_always_return_a_valid_rank(self):
        rng = random.Random(7)
        for _ in range(2000):
            n = rng.choice([2, 2, 2, 3, 4, 8])
            state = [
                st(running=rng.randint(0, 140),
                   waiting=rng.randint(0, 30),
                   pending=rng.randint(0, 80000),
                   kv=rng.random(),
                   routed=rng.randint(0, 50)) for _ in range(n)
            ]
            hits = [
                rng.choice([0, 0, BS, 2 * BS, 7 * BS, 20 * BS])
                for _ in range(n)
            ]
            assert 0 <= rp.pick_rank(hits, state, BS) < n


# ===========================================================================
# 2. DPScheduler wiring with mocked IPC
# ===========================================================================
def _mk_config(max_seqs=128, caching=True, dp=2):
    cfg = MagicMock()
    cfg.sharding_config = MagicMock()
    cfg.sharding_config.total_dp_size = dp
    cfg.scheduler_config = MagicMock()
    cfg.scheduler_config._original_scheduler_cls = Scheduler
    cfg.scheduler_config.max_num_seqs = max_seqs
    cfg.scheduler_config.max_num_batched_tokens = 65536
    cfg.scheduler_config.async_scheduling = False
    cfg.cache_config = MagicMock()
    cfg.cache_config.enable_prefix_caching = caching
    cfg.cache_config.prefix_caching_hash_algo = "sha256"
    cfg.cache_config.mamba_num_blocks = None
    return cfg


def _mock_ctx():
    ctx = MagicMock()
    ctx.Pipe = MagicMock(return_value=(MagicMock(), MagicMock()))
    ctx.Process = MagicMock(return_value=MagicMock())
    return ctx


def _make(monkeypatch, env=None, caching=True, log_stats=True, block_size=BS):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    kvc = MagicMock(spec=KVCacheConfig)
    kvc.num_blocks = 2 * NB
    with patch(
            "tpu_inference.core.sched.dp_scheduler._scheduler_worker_process"
    ), patch("multiprocessing.get_context", return_value=_mock_ctx()), patch(
            "vllm.v1.core.kv_cache_utils.init_none_hash"):
        return dps.DPScheduler(vllm_config=_mk_config(caching=caching),
                               kv_cache_config=kvc,
                               structured_output_manager=MagicMock(),
                               block_size=block_size,
                               log_stats=log_stats)


def _load(running=0, waiting=0, pending=0, kv=0.0):
    return dict(running=running,
                waiting=waiting,
                pending_prefill_tokens=pending,
                kv_usage=kv,
                num_blocks=NB)


def _request():
    request = MagicMock(spec=Request)
    request.sampling_params = None
    return request


class TestDPSchedulerRouting:

    def test_default_is_legacy_and_never_sends_the_new_command(
            self, monkeypatch):
        s = _make(monkeypatch)
        assert s._rank_routing == "legacy"
        s._send_command = MagicMock()
        s._get_result = MagicMock(side_effect=[10, 25])  # as the upstream test
        assert s._find_best_rank_for_request(_request()) == 1
        sent = [c.args[1] for c in s._send_command.call_args_list]
        assert dps.SchedulerCommand.GET_RANK_LOAD not in sent

    def test_invalid_mode_falls_back_to_legacy(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "turbo"})
        assert s._rank_routing == "legacy"

    def test_mode_is_case_and_space_tolerant(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": " Load "})
        assert s._rank_routing == "load"

    def test_load_mode_pipelines_probe_then_load_and_picks(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "load"})
        assert s._rank_routing == "load"
        s._send_command = MagicMock()
        # rank 1 has a 5-block hit but is 0.2 KV hotter -> override to rank 0
        s._get_result = MagicMock(side_effect=[
            0,
            5 * BS,  # PROBE r0, r1
            _load(running=30, kv=0.30),
            _load(running=30, kv=0.50),  # LOAD r0, r1
        ])
        assert s._find_best_rank_for_request(_request()) == 0
        sent = [(c.args[0], c.args[1]) for c in s._send_command.call_args_list]
        C = dps.SchedulerCommand
        assert sent == [(0, C.PROBE_COMPUTED_BLOCKS),
                        (1, C.PROBE_COMPUTED_BLOCKS), (0, C.GET_RANK_LOAD),
                        (1, C.GET_RANK_LOAD)]
        got = [(c.args[0], c.args[1]) for c in s._get_result.call_args_list]
        assert got == sent
        assert s._route_reasons == {"kv_gap": 1}

    def test_load_mode_keeps_affinity_when_cool(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "load"})
        s._send_command = MagicMock()
        s._get_result = MagicMock(side_effect=[
            0, 5 * BS,
            _load(running=30, kv=0.40),
            _load(running=33, kv=0.42)
        ])
        assert s._find_best_rank_for_request(_request()) == 1
        assert s._route_reasons == {"affinity": 1}

    def test_load_mode_without_prefix_caching_sends_only_load(
            self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "load"}, caching=False)
        s._send_command = MagicMock()
        s._get_result = MagicMock(
            side_effect=[_load(running=50, kv=0.5),
                         _load(running=10, kv=0.2)])
        assert s._find_best_rank_for_request(_request()) == 1
        sent = [c.args[1] for c in s._send_command.call_args_list]
        assert sent == [dps.SchedulerCommand.GET_RANK_LOAD] * 2

    def test_env_thresholds_reach_the_picker(self, monkeypatch):
        s = _make(monkeypatch, {
            "DP_SCHED_ROUTING": "load",
            "DP_SCHED_AFFINITY_MIN_BLOCKS": "6"
        })
        s._send_command = MagicMock()
        s._get_result = MagicMock(side_effect=[
            0, 5 * BS,
            _load(running=30, kv=0.40),
            _load(running=30, kv=0.40)
        ])
        s._find_best_rank_for_request(_request())
        assert s._route_reasons == {"small_advantage": 1}

    def test_bad_threshold_env_is_reported_not_fatal(self, monkeypatch):
        with patch.object(dps.logger, "warning") as warning:
            s = _make(monkeypatch, {"DP_SCHED_AFFINITY_MAX_KV_GAP": "nan"})
        assert s._picker_cfg.max_kv_gap == 0.10
        assert any("DP_SCHED_AFFINITY_MAX_KV_GAP" in str(c)
                   for c in warning.call_args_list)

    def test_max_seqs_is_passed_into_the_state(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "load"}, caching=False)
        assert s._max_seqs_per_rank == 128
        s._send_command = MagicMock()
        s._get_result = MagicMock(side_effect=[_load(), _load()])
        with patch.object(dps, "pick_rank", return_value=0) as pk:
            s._find_best_rank_for_request(_request())
        loads = pk.call_args.args[1]
        assert all(x["max_seqs"] == 128 for x in loads)
        assert [x["routed_total"] for x in loads] == [0, 0]

    def test_route_and_forward_counts_routed_per_rank(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "load"})
        s._find_best_rank_for_request = MagicMock(side_effect=[1, 1, 0])
        s._send_command = MagicMock()
        s._get_result = MagicMock(return_value=None)
        for i in range(3):
            req = MagicMock(spec=Request)
            req.request_id = "r%d" % i
            s._route_and_forward_request(req)
        assert s._rank_routed_total == [1, 2]

    def test_a_pinned_rank_beats_load_routing_and_is_counted(
            self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "load"})
        s._send_command = MagicMock()
        s._get_result = MagicMock()
        request = _request()
        request.sampling_params = SimpleNamespace(extra_args={"dp_rank": 1})
        assert s._find_best_rank_for_request(request) == 1
        s._send_command.assert_not_called()
        assert s._route_reasons == {"hint": 1}


# ===========================================================================
# 3. over REAL pipes and REAL forked workers (stub per-rank scheduler)
# ===========================================================================
class _FakeKV:
    enable_caching = True

    def __init__(self):
        self.usage = 0.0
        self.block_pool = SimpleNamespace(num_gpu_blocks=NB)
        self.cached = set()
        outer = self

        class _Coord:

            def find_longest_cache_hit(self, block_hashes, max_len):
                return None, (5 * BS if block_hashes in outer.cached else 0)

        self.coordinator = _Coord()


class FakeScheduler:
    """A request's .block_hashes is its template id; a rank 'caches' a template
    once it has been added there."""

    def __init__(self, **kw):
        self.running = []
        self.waiting = deque()
        self.skipped_waiting = deque()
        self.kv_cache_manager = _FakeKV()

    def add_request(self, request):
        self.waiting.append(request)
        self.kv_cache_manager.cached.add(request.block_hashes)
        self.kv_cache_manager.usage = min(
            0.99, 0.002 * (len(self.waiting) + len(self.running)))

    def shutdown(self):
        pass


def _fake_req(i, template="T"):
    return SimpleNamespace(request_id="q%d" % i,
                           num_tokens=1275,
                           num_prompt_tokens=1275,
                           num_computed_tokens=0,
                           block_hashes=template,
                           skip_reading_prefix_cache=False,
                           sampling_params=None)


def _make_real(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg = _mk_config()
    cfg.scheduler_config._original_scheduler_cls = FakeScheduler
    kvc = MagicMock(spec=KVCacheConfig)
    kvc.num_blocks = 2 * NB
    with patch("vllm.v1.core.kv_cache_utils.init_none_hash"):
        s = dps.DPScheduler(vllm_config=cfg,
                            kv_cache_config=kvc,
                            structured_output_manager=MagicMock(),
                            block_size=BS,
                            log_stats=True)
    s._batch_prefills = False
    return s


def _skew_run(monkeypatch, env, n=80):
    s = _make_real(monkeypatch, env)
    try:
        # template T is cached on rank 0 only (as if rank 0 served it first)
        seed = _fake_req(-1)
        s._send_command(0, dps.SchedulerCommand.ADD_REQUEST, seed)
        s._get_result(0, dps.SchedulerCommand.ADD_REQUEST)
        for i in range(n):
            s.add_request(_fake_req(i))
        per_rank = [0, 0]
        for r in s.assigned_dp_rank.values():
            per_rank[r] += 1
        return per_rank, dict(s._route_reasons)
    finally:
        s.shutdown()


class TestOverRealPipes:

    def test_legacy_pins_a_whole_template_to_one_rank(self, monkeypatch):
        per_rank, _ = _skew_run(monkeypatch, {})
        assert per_rank == [80, 0]  # the skew mechanism

    def test_load_mode_bounds_the_skew(self, monkeypatch):
        per_rank, reasons = _skew_run(monkeypatch,
                                      {"DP_SCHED_ROUTING": "load"})
        assert sum(per_rank) == 80
        assert abs(per_rank[0] - per_rank[1]) <= 12, (per_rank, reasons)
        assert min(per_rank) >= 30, per_rank
        assert reasons.get("affinity", 0) > 0  # still used while balanced
        assert reasons.get("queue_gap", 0) + reasons.get("kv_gap", 0) > 0

    def test_load_mode_no_hits_balances_exactly_and_is_not_rank0_biased(
            self, monkeypatch):
        s = _make_real(monkeypatch, {"DP_SCHED_ROUTING": "load"})
        try:
            for i in range(40):
                s.add_request(_fake_req(i, template="u%d" % i))  # distinct
            per_rank = [0, 0]
            for r in s.assigned_dp_rank.values():
                per_rank[r] += 1
            assert per_rank == [20, 20], per_rank
            first4 = [s.assigned_dp_rank["q%d" % i] for i in range(4)]
            assert first4 != [0, 0, 0, 0]
        finally:
            s.shutdown()

    def test_get_rank_load_reports_waiting_skipped_and_pending(
            self, monkeypatch):
        s = _make_real(monkeypatch, {"DP_SCHED_ROUTING": "load"})
        try:
            C = dps.SchedulerCommand
            for i in range(3):
                s._send_command(1, C.ADD_REQUEST, _fake_req(i, "z"))
                s._get_result(1, C.ADD_REQUEST)
            s._send_command(1, C.GET_RANK_LOAD)
            got = s._get_result(1, C.GET_RANK_LOAD)
            assert got["running"] == 0 and got["waiting"] == 3
            assert got["pending_prefill_tokens"] == 3 * 1275
            assert got["num_blocks"] == NB
            assert got["kv_usage"] == pytest.approx(0.006)
        finally:
            s.shutdown()


def _worker_with(sched):
    """Run the real worker loop in a forked process around ``sched``."""
    import multiprocessing

    import cloudpickle
    ctx = multiprocessing.get_context("fork")
    in_p, in_c = ctx.Pipe()
    out_p, out_c = ctx.Pipe()

    class _Cls:

        def __new__(cls, **kw):
            return sched

    proc = ctx.Process(target=dps._scheduler_worker_process,
                       args=(0, in_c, out_c, None, None, None, BS, BS, None,
                             False, True, _Cls))
    proc.start()
    return proc, in_p, out_p, cloudpickle


def test_worker_counts_skipped_waiting_which_upstream_commands_drop():
    sched = SimpleNamespace(running=[
        SimpleNamespace(num_prompt_tokens=1000, num_computed_tokens=400),
        SimpleNamespace(num_prompt_tokens=500, num_computed_tokens=500)
    ],
                            waiting=deque(
                                [SimpleNamespace(num_prompt_tokens=2000)]),
                            skipped_waiting=deque([
                                SimpleNamespace(num_prompt_tokens=3000),
                                SimpleNamespace(num_prompt_tokens=100)
                            ]),
                            kv_cache_manager=SimpleNamespace(
                                usage=0.61,
                                block_pool=SimpleNamespace(num_gpu_blocks=NB)),
                            shutdown=lambda: None)
    proc, in_p, out_p, cp = _worker_with(sched)
    try:
        C = dps.SchedulerCommand
        in_p.send_bytes(cp.dumps((C.GET_RANK_LOAD, None)))
        got = cp.loads(out_p.recv_bytes())
        assert got == dict(running=2,
                           waiting=3,
                           pending_prefill_tokens=600 + 2000 + 3100,
                           kv_usage=0.61,
                           num_blocks=NB)
        in_p.send_bytes(cp.dumps((C.GET_REQUEST_COUNTS, None)))
        assert cp.loads(out_p.recv_bytes()) == (2, 1)  # upstream drops skipped
    finally:
        proc.terminate()
        proc.join(5)


def test_worker_attribute_names_exist_in_vllm():
    """The worker code reads names off vLLM's Scheduler / KVCacheManager. Check
    them against the real vLLM source in use."""
    import inspect

    import vllm.v1.core.block_pool as bp
    import vllm.v1.core.kv_cache_manager as km
    import vllm.v1.core.sched.scheduler as sc
    src = inspect.getsource(sc.Scheduler.__init__)
    for name in ("self.waiting =", "self.skipped_waiting =", "self.running"):
        assert name in src, name
    assert isinstance(km.KVCacheManager.usage, property)
    assert "self.block_pool = " in inspect.getsource(
        km.KVCacheManager.__init__)
    assert "self.num_gpu_blocks = num_gpu_blocks" in inspect.getsource(
        bp.BlockPool.__init__)
    assert "preempted_req_ids" in inspect.getsource(SchedulerOutput)
    assert "num_skipped_waiting_reqs" in inspect.getsource(SchedulerStats)


# ===========================================================================
# 4. the REAL vLLM schedulers (CPU, no model)
# ===========================================================================
TEMPLATE_BLOCKS = 6  # shared prefix: 96 tokens


def _rank_load(s, rank):
    s._send_command(rank, dps.SchedulerCommand.GET_RANK_LOAD)
    return s._get_result(rank, dps.SchedulerCommand.GET_RANK_LOAD)


def _template_req(i):
    return lib.make_request("q%d" % i,
                            prefix_len=TEMPLATE_BLOCKS * lib.BLOCK,
                            body_len=100,
                            seed=i,
                            max_tokens=64,
                            prefix_token=7)


def _drive(s, n=120, per_step=2, dp=2):
    """Arrivals of ``per_step`` template requests per scheduler step."""
    s._batch_prefills = True  # production default
    for i in range(0, n, per_step):
        for j in range(per_step):
            s.add_request(_template_req(i + j))
        s.schedule()
    s.schedule()
    per_rank = [0] * dp
    for r in s.assigned_dp_rank.values():
        per_rank[r] += 1
    return per_rank


class TestRealSchedulerEndToEnd:

    @pytest.mark.parametrize("sched_cls", [AsyncScheduler, Scheduler])
    def test_get_rank_load_matches_the_real_scheduler(self, model_dir,
                                                      monkeypatch, sched_cls):
        monkeypatch.setenv("DP_SCHED_ROUTING", "load")
        _, s = lib.make_real_dp_scheduler(model_dir,
                                          2,
                                          pool_blocks_per_rank=4000,
                                          max_num_seqs=128,
                                          scheduler_cls=sched_cls)
        try:
            assert _rank_load(s, 0) == dict(running=0,
                                            waiting=0,
                                            pending_prefill_tokens=0,
                                            kv_usage=0.0,
                                            num_blocks=4000)
            n = 6
            for i in range(n):
                s.add_request(_template_req(i))
            s.schedule()  # moves them waiting -> running and allocates KV
            l0, l1 = _rank_load(s, 0), _rank_load(s, 1)
            assert l0["running"] + l1["running"] == n
            assert l0["waiting"] + l1["waiting"] == 0
            assert l0["kv_usage"] > 0 and l1["kv_usage"] > 0
            assert l0["num_blocks"] == l1["num_blocks"] == 4000
            # usage and counts agree with what vLLM itself reports per rank
            s._send_command(0, dps.SchedulerCommand.MAKE_STATS, (None, None))
            st0 = s._get_result(0, dps.SchedulerCommand.MAKE_STATS)
            assert st0.num_running_reqs == l0["running"]
            assert st0.kv_cache_usage == pytest.approx(l0["kv_usage"])
        finally:
            s.shutdown()

    def test_legacy_sends_every_template_request_to_rank0_once_both_cached_it(
            self, model_dir, monkeypatch):
        _, s = lib.make_real_dp_scheduler(model_dir,
                                          2,
                                          pool_blocks_per_rank=4000,
                                          max_num_seqs=128)
        try:
            per_rank = _drive(s, n=120)
            l0, l1 = _rank_load(s, 0), _rank_load(s, 1)
            assert per_rank[0] >= 115 and per_rank[1] <= 5, per_rank
            assert l1["kv_usage"] < l0["kv_usage"] / 10
        finally:
            s.shutdown()

    def test_load_mode_spreads_the_same_traffic(self, model_dir, monkeypatch):
        monkeypatch.setenv("DP_SCHED_ROUTING", "load")
        _, s = lib.make_real_dp_scheduler(model_dir,
                                          2,
                                          pool_blocks_per_rank=4000,
                                          max_num_seqs=128)
        try:
            per_rank = _drive(s, n=120)
            l0, l1 = _rank_load(s, 0), _rank_load(s, 1)
            assert abs(per_rank[0] - per_rank[1]) <= 8, per_rank
            assert abs(l0["kv_usage"] - l1["kv_usage"]) < 0.02
        finally:
            s.shutdown()

    @pytest.mark.parametrize("dp", [3, 4])
    def test_the_skew_and_its_bound_hold_at_dp_3_and_4(self, model_dir,
                                                       monkeypatch, dp):
        """Same template traffic at DP 3 and 4: legacy still piles it on one
        rank, load mode spreads it over all of them."""
        results = {}
        for mode in ("legacy", "load"):
            monkeypatch.setenv("DP_SCHED_ROUTING", mode)
            _, s = lib.make_real_dp_scheduler(model_dir,
                                              dp,
                                              pool_blocks_per_rank=4000,
                                              max_num_seqs=128)
            try:
                results[mode] = _drive(s, n=120, dp=dp)
            finally:
                s.shutdown()
        assert max(results["legacy"]) >= 110, results
        assert max(results["load"]) - min(results["load"]) <= 12, results
        assert min(results["load"]) >= 120 // dp - 12, results

    def test_load_mode_still_uses_affinity_when_it_is_cheap(
            self, model_dir, monkeypatch):
        """Template cached on rank 0 only, ranks equally idle -> follow it."""
        monkeypatch.setenv("DP_SCHED_ROUTING", "load")
        _, s = lib.make_real_dp_scheduler(model_dir,
                                          2,
                                          pool_blocks_per_rank=4000,
                                          max_num_seqs=128)
        try:
            s._batch_prefills = False
            seed = _template_req(10_000)
            s._send_command(0, dps.SchedulerCommand.ADD_REQUEST, seed)
            s._get_result(0, dps.SchedulerCommand.ADD_REQUEST)
            s.assigned_dp_rank[seed.request_id] = 0
            s.schedule()  # rank 0 allocates and caches the template blocks
            s.add_request(_template_req(1))
            assert s.assigned_dp_rank["q1"] == 0
            assert s._route_reasons.get("affinity", 0) + s._route_reasons.get(
                "agree", 0) == 1
        finally:
            s.shutdown()
