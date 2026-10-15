# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""KV aggregation (DP_SCHED_KV_AGGREGATE) and per-rank DP metrics.

* the reduction itself, at DP 2/3/4/8;
* ``DPScheduler._combine_scheduler_stats`` / ``make_stats`` with the env;
* end to end through vLLM's REAL ``PrometheusStatLogger`` and the REAL
  DPScheduler (real rank workers), showing the exported
  ``vllm:kv_cache_usage_perc`` as mean vs max next to the per-rank gauges;
* the cross-process path the per-rank series need in production (EngineCore
  forked after prometheus_client froze its in-memory value class, API server
  scraping ``PROMETHEUS_MULTIPROC_DIR``), in a fresh interpreter;
* per-rank preemptions against vLLM's own preemption events.

CPU only: ``VLLM_TARGET_DEVICE=empty JAX_PLATFORMS=cpu``.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY, CollectorRegistry
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats

from tests.core import dp_testlib as lib
from tpu_inference import envs
from tpu_inference.core.sched import dp_metrics
from tpu_inference.core.sched.dp_metrics import (DPRankMetrics,
                                                 aggregate_kv_usage,
                                                 count_preemptions)
from tpu_inference.core.sched.dp_scheduler import DPScheduler, SchedulerCommand

DP_SIZES = [2, 3, 4, 8]
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return lib.write_tiny_llama(tmp_path_factory.mktemp("tiny-llama"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("DP_SCHED_KV_AGGREGATE", "DP_SCHED_RANK_METRICS",
                 "DP_SCHED_RANK_METRICS_INTERVAL_S", "DP_SCHED_ROUTING"):
        monkeypatch.delenv(name, raising=False)


def _sample(registry, name, **labels):
    """Value of one exposition sample, or None."""
    for family in registry.collect():
        for sample in family.samples:
            if sample.name == name and all(
                    sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


# ---------------------------------------------------------------------------
# 1. the reduction and the env
# ---------------------------------------------------------------------------
class TestAggregate:

    @pytest.mark.parametrize("dp", DP_SIZES)
    def test_one_full_rank_mean_vs_max(self, dp):
        per_rank = [0.919] + [0.0] * (dp - 1)
        assert aggregate_kv_usage(per_rank,
                                  "mean") == pytest.approx(0.919 / dp)
        assert aggregate_kv_usage(per_rank, "max") == pytest.approx(0.919)

    def test_none_counts_as_an_idle_rank(self):
        assert aggregate_kv_usage([0.8, None], "mean") == pytest.approx(0.4)
        assert aggregate_kv_usage([None, 0.5], "max") == 0.5
        assert aggregate_kv_usage([None, None], "max") == 0.0

    def test_empty(self):
        assert aggregate_kv_usage([], "mean") == 0.0
        assert aggregate_kv_usage([], "max") == 0.0

    def test_mean_is_the_historical_arithmetic(self):
        values = [0.31, 0.62, 0.05, 0.99]
        assert aggregate_kv_usage(values, "mean") == sum(values) / len(values)


class TestEnv:

    def test_defaults(self):
        assert envs.DP_SCHED_KV_AGGREGATE == "mean"
        assert envs.DP_SCHED_RANK_METRICS is True
        assert envs.DP_SCHED_RANK_METRICS_INTERVAL_S == 1.0

    @pytest.mark.parametrize("raw", ["max", "MAX", "Mean"])
    def test_choices_are_case_insensitive(self, monkeypatch, raw):
        monkeypatch.setenv("DP_SCHED_KV_AGGREGATE", raw)
        assert envs.DP_SCHED_KV_AGGREGATE.lower() in ("mean", "max")

    def test_invalid_value_fails_fast(self, monkeypatch):
        monkeypatch.setenv("DP_SCHED_KV_AGGREGATE", "median")
        with pytest.raises(ValueError, match="DP_SCHED_KV_AGGREGATE"):
            envs.DP_SCHED_KV_AGGREGATE

    def test_metrics_switch_and_interval(self, monkeypatch):
        monkeypatch.setenv("DP_SCHED_RANK_METRICS", "0")
        monkeypatch.setenv("DP_SCHED_RANK_METRICS_INTERVAL_S", "0.25")
        assert envs.DP_SCHED_RANK_METRICS is False
        assert envs.DP_SCHED_RANK_METRICS_INTERVAL_S == 0.25


# ---------------------------------------------------------------------------
# 2. DPScheduler stats combination with mocked IPC
# ---------------------------------------------------------------------------
def _mocked_scheduler(dp, monkeypatch, env=None):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    cfg = MagicMock()
    cfg.sharding_config = MagicMock()
    cfg.sharding_config.total_dp_size = dp
    cfg.scheduler_config = MagicMock()
    cfg.scheduler_config._original_scheduler_cls = Scheduler
    cfg.scheduler_config.max_num_seqs = 8
    cfg.scheduler_config.max_num_batched_tokens = 1024
    cfg.scheduler_config.async_scheduling = False
    cfg.cache_config = MagicMock()
    cfg.cache_config.enable_prefix_caching = False
    cfg.cache_config.mamba_num_blocks = None
    cfg.model_config.served_model_name = "dp-test-model"
    ctx = MagicMock()
    ctx.Pipe = MagicMock(return_value=(MagicMock(), MagicMock()))
    ctx.Process = MagicMock(return_value=MagicMock())
    kv_cache_config = MagicMock(spec=KVCacheConfig)
    kv_cache_config.num_blocks = 100 * dp
    with patch(
            "tpu_inference.core.sched.dp_scheduler._scheduler_worker_process"
    ), patch("multiprocessing.get_context", return_value=ctx):
        return DPScheduler(vllm_config=cfg,
                           kv_cache_config=kv_cache_config,
                           structured_output_manager=MagicMock(),
                           block_size=16,
                           log_stats=True)


def _stats(run, wait, kv):
    return SchedulerStats(num_running_reqs=run,
                          num_waiting_reqs=wait,
                          kv_cache_usage=kv,
                          prefix_cache_stats=PrefixCacheStats())


class TestCombineStats:

    @pytest.mark.parametrize("dp", DP_SIZES)
    def test_default_is_the_mean_and_counts_are_sums(self, dp, monkeypatch):
        s = _mocked_scheduler(dp, monkeypatch)
        per_rank = [_stats(10, 2, 0.92)] + [_stats(1, 0, 0.0)] * (dp - 1)
        out = s._combine_scheduler_stats(per_rank)
        assert out.kv_cache_usage == pytest.approx(0.92 / dp)
        assert out.num_running_reqs == 10 + (dp - 1)
        assert out.num_waiting_reqs == 2

    @pytest.mark.parametrize("dp", DP_SIZES)
    def test_max_exports_the_hottest_rank(self, dp, monkeypatch):
        s = _mocked_scheduler(dp, monkeypatch,
                              {"DP_SCHED_KV_AGGREGATE": "max"})
        per_rank = [_stats(10, 2, 0.1)] * (dp - 1) + [_stats(10, 2, 0.92)]
        out = s._combine_scheduler_stats(per_rank)
        assert out.kv_cache_usage == pytest.approx(0.92)
        assert out.num_running_reqs == 10 * dp  # counts stay sums
        assert out.num_waiting_reqs == 2 * dp

    def test_rank_without_stats_counts_as_idle_in_mean(self, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch)
        out = s._combine_scheduler_stats([_stats(1, 0, 0.8), None])
        assert out.kv_cache_usage == pytest.approx(0.4)

    def test_nothing_to_combine(self, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch)
        assert s._combine_scheduler_stats([None, None]) is None

    @pytest.mark.parametrize("mode,expected", [("mean", 0.5), ("max", 0.8)])
    def test_make_stats_follows_the_same_env(self, monkeypatch, mode,
                                             expected):
        s = _mocked_scheduler(2, monkeypatch, {"DP_SCHED_KV_AGGREGATE": mode})
        s._send_command = MagicMock()
        s._get_result = MagicMock(
            side_effect=[_stats(3, 1, 0.8),
                         _stats(2, 0, 0.2)])
        out = s.make_stats()
        assert out.kv_cache_usage == pytest.approx(expected)
        assert (out.num_running_reqs, out.num_waiting_reqs) == (5, 1)

    def test_make_stats_missing_rank_counts_as_idle(self, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch)
        s._send_command = MagicMock()
        s._get_result = MagicMock(side_effect=[_stats(3, 1, 0.8), None])
        assert s.make_stats().kv_cache_usage == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 3. DPRankMetrics on its own (isolated registries)
# ---------------------------------------------------------------------------
class _Clock:

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class TestDPRankMetrics:

    def _metrics(self, dp=2, registry=None, interval_s=1.0, clock=None):
        registry = registry or CollectorRegistry()
        return registry, DPRankMetrics(model_name="m",
                                       dp_size=dp,
                                       registry=registry,
                                       interval_s=interval_s,
                                       clock=clock or _Clock())

    def test_series_exist_at_zero_for_every_rank(self):
        registry, _ = self._metrics(dp=3)
        for rank in range(3):
            labels = dict(model_name="m", engine="0", dp_rank=str(rank))
            assert _sample(registry,
                           "tpu_inference:dp_rank_kv_cache_usage_perc",
                           **labels) == 0.0
            assert _sample(registry,
                           "tpu_inference:dp_rank_num_preemptions_total",
                           **labels) == 0.0

    def test_observe_stats_sets_all_three_gauges(self):
        registry, m = self._metrics()
        m.observe_stats([_stats(7, 3, 0.9), None])
        labels = dict(model_name="m", engine="0")
        assert _sample(registry,
                       "tpu_inference:dp_rank_kv_cache_usage_perc",
                       dp_rank="0",
                       **labels) == pytest.approx(0.9)
        assert _sample(registry,
                       "tpu_inference:dp_rank_num_requests_running",
                       dp_rank="0",
                       **labels) == 7
        assert _sample(registry,
                       "tpu_inference:dp_rank_num_requests_waiting",
                       dp_rank="0",
                       **labels) == 3
        # a rank that reported nothing keeps its last value (here 0)
        assert _sample(registry,
                       "tpu_inference:dp_rank_kv_cache_usage_perc",
                       dp_rank="1",
                       **labels) == 0.0

    def test_gauge_updates_are_throttled_and_counters_are_not(self):
        clock = _Clock()
        registry, m = self._metrics(interval_s=1.0, clock=clock)
        labels = dict(model_name="m", engine="0", dp_rank="0")
        m.observe_stats([_stats(1, 0, 0.1), None])
        clock.now += 0.5
        m.observe_stats([_stats(2, 0, 0.2), None])  # inside the interval
        assert _sample(registry, "tpu_inference:dp_rank_num_requests_running",
                       **labels) == 1
        clock.now += 0.6
        m.observe_stats([_stats(3, 0, 0.3), None])
        assert _sample(registry, "tpu_inference:dp_rank_num_requests_running",
                       **labels) == 3
        m.add_preemptions([2, 0])
        m.add_preemptions([1, 4])
        m.add_routed(0)
        m.add_routed(1)
        m.add_routed(1)
        assert _sample(registry, "tpu_inference:dp_rank_num_preemptions_total",
                       **labels) == 3
        assert _sample(registry,
                       "tpu_inference:dp_rank_num_preemptions_total",
                       model_name="m",
                       engine="0",
                       dp_rank="1") == 4
        assert _sample(registry,
                       "tpu_inference:dp_rank_requests_routed_total",
                       model_name="m",
                       engine="0",
                       dp_rank="1") == 2

    def test_zero_interval_publishes_every_step(self):
        registry, m = self._metrics(interval_s=0)
        for run in (1, 2, 3):
            m.observe_stats([_stats(run, 0, 0.0), None])
            assert _sample(registry,
                           "tpu_inference:dp_rank_num_requests_running",
                           dp_rank="0") == run

    def test_two_instances_share_one_set_of_series(self):
        registry = CollectorRegistry()
        a = DPRankMetrics(model_name="m", dp_size=2, registry=registry)
        b = DPRankMetrics(model_name="m", dp_size=2, registry=registry)
        a.add_routed(0)
        b.add_routed(0)
        assert _sample(registry,
                       "tpu_inference:dp_rank_requests_routed_total",
                       dp_rank="0") == 2

    def test_a_metrics_failure_never_raises(self):
        registry = CollectorRegistry()
        m = DPRankMetrics(model_name="m", dp_size=2, registry=registry)
        m.observe_stats([object(), object()])  # AttributeError inside
        assert not m.enabled
        m.observe_stats([_stats(1, 0, 0.1)] * 2)
        m.add_preemptions([1, 1])
        m.add_routed(0)

    def test_name_clash_disables_instead_of_raising(self):
        from prometheus_client import Gauge
        registry = CollectorRegistry()
        Gauge("tpu_inference:dp_rank_kv_cache_usage_perc",
              "squatter",
              registry=registry)
        m = DPRankMetrics(model_name="m", dp_size=2, registry=registry)
        assert not m.enabled
        m.add_routed(0)

    def test_count_preemptions(self):
        outs = [
            SimpleNamespace(preempted_req_ids={"a", "b"}),
            SimpleNamespace(preempted_req_ids=None),
            SimpleNamespace()
        ]
        assert count_preemptions(outs) == [2, 0, 0]

    def test_model_name_falls_back_when_not_a_string(self):
        cfg = SimpleNamespace(model_config=SimpleNamespace(
            served_model_name=MagicMock()))
        m = DPRankMetrics.from_vllm_config(cfg, 2, 1.0)
        assert m.enabled


class TestEnsureMultiprocessValueClass:
    """The value-class switch must only happen when it can work."""

    @staticmethod
    def _freeze_mutex(monkeypatch):
        import prometheus_client.values as values

        class MutexValue:  # what prometheus_client froze before the fork
            pass

        monkeypatch.setattr(values, "ValueClass", MutexValue)
        return values, MutexValue

    def test_no_directory_no_switch(self, monkeypatch):
        values, mutex = self._freeze_mutex(monkeypatch)
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        monkeypatch.delenv("prometheus_multiproc_dir", raising=False)
        assert dp_metrics.ensure_multiprocess_value_class() is False
        assert values.ValueClass is mutex

    @pytest.mark.parametrize(
        "name", ["PROMETHEUS_MULTIPROC_DIR", "prometheus_multiproc_dir"])
    def test_usable_directory_switches_once(self, monkeypatch, tmp_path, name):
        values, mutex = self._freeze_mutex(monkeypatch)
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        monkeypatch.delenv("prometheus_multiproc_dir", raising=False)
        monkeypatch.setenv(name, str(tmp_path))
        assert dp_metrics.ensure_multiprocess_value_class() is True
        switched = values.ValueClass
        assert switched is not mutex and switched.__name__ == "MmapedValue"
        assert dp_metrics.ensure_multiprocess_value_class() is True
        assert values.ValueClass is switched  # not switched again

    def test_missing_directory_leaves_the_process_alone(
            self, monkeypatch, tmp_path):
        values, mutex = self._freeze_mutex(monkeypatch)
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR",
                           str(tmp_path / "does-not-exist"))
        assert dp_metrics.ensure_multiprocess_value_class() is False
        assert values.ValueClass is mutex

    def test_unwritable_directory_leaves_the_process_alone(
            self, monkeypatch, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root can write anywhere")
        values, mutex = self._freeze_mutex(monkeypatch)
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(locked))
            assert dp_metrics.ensure_multiprocess_value_class() is False
            assert values.ValueClass is mutex
        finally:
            locked.chmod(0o700)


# ---------------------------------------------------------------------------
# 4. DPScheduler wiring (mocked IPC)
# ---------------------------------------------------------------------------
class TestDPSchedulerWiring:

    def test_switch_off(self, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch, {"DP_SCHED_RANK_METRICS": "0"})
        assert s._rank_metrics is None
        s._send_command = MagicMock()
        s._get_result = MagicMock(return_value=None)
        request = MagicMock()
        request.request_id = "x"
        s._find_best_rank_for_request = MagicMock(return_value=1)
        s._route_and_forward_request(request)  # must not touch metrics
        assert s.assigned_dp_rank["x"] == 1

    def test_routed_counter_follows_the_chosen_rank(self, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch)
        s._rank_metrics = MagicMock()
        s._find_best_rank_for_request = MagicMock(side_effect=[1, 0, 1])
        s._send_command = MagicMock()
        s._get_result = MagicMock(return_value=None)
        for i in range(3):
            request = MagicMock()
            request.request_id = f"q{i}"
            s._route_and_forward_request(request)
        assert [c.args[0] for c in s._rank_metrics.add_routed.call_args_list
                ] == [1, 0, 1]

    def test_schedule_reports_per_rank_preemptions(self, monkeypatch):
        s = _mocked_scheduler(2, monkeypatch)
        s._rank_metrics = MagicMock()

        def out(preempted):
            o = MagicMock()
            o.preempted_req_ids = preempted
            o.scheduled_new_reqs = []
            o.num_scheduled_tokens = {}
            o.total_num_scheduled_tokens = 0
            o.finished_req_ids = set()
            o.scheduled_cached_reqs = SimpleNamespace(req_ids=[],
                                                      resumed_req_ids=[],
                                                      new_token_ids=[],
                                                      all_token_ids={},
                                                      new_block_ids=[],
                                                      num_computed_tokens=[],
                                                      num_output_tokens=[])
            o.scheduled_spec_decode_tokens = {}
            o.scheduled_encoder_inputs = {}
            o.num_common_prefix_blocks = []
            o.kv_connector_metadata = None
            return o

        s._send_command = MagicMock()
        s._collect_results_unordered = MagicMock(
            return_value=[out({"a", "b"}), out(None)])
        with patch.object(s, "_combine_scheduler_outputs"):
            s.schedule()
        s._rank_metrics.add_preemptions.assert_called_once_with([2, 0])


# ---------------------------------------------------------------------------
# 5. end to end with the real PrometheusStatLogger and a real DPScheduler
# ---------------------------------------------------------------------------
def _hot_rank_traffic(scheduler, n=12):
    """Everything pinned to rank 0 (via the rank hint), every other rank idle:
    the production symptom, built deterministically."""
    scheduler._batch_prefills = False
    for i in range(n):
        scheduler.add_request(
            lib.make_request(f"r{i}",
                             seed=i,
                             prefix_len=64,
                             body_len=300,
                             extra_args={"dp_rank": 0}))
    stats = None
    for _ in range(4):
        stats = lib.last_stats(lib.step(scheduler)) or stats
    return stats


def _rank_kv(scheduler, rank):
    scheduler._send_command(rank, SchedulerCommand.MAKE_STATS, (None, None))
    return scheduler._get_result(rank,
                                 SchedulerCommand.MAKE_STATS).kv_cache_usage


@pytest.mark.parametrize("dp", [2, 3, 4, 8])
@pytest.mark.parametrize("mode", ["mean", "max"])
def test_exported_gauge_vs_per_rank_gauges(model_dir, monkeypatch, dp, mode):
    monkeypatch.setenv("DP_SCHED_KV_AGGREGATE", mode)
    monkeypatch.setenv("DP_SCHED_RANK_METRICS_INTERVAL_S", "0")
    vc, s = lib.make_real_dp_scheduler(model_dir, dp, pool_blocks_per_rank=200)
    try:
        stats = _hot_rank_traffic(s)
        per_rank_kv = [_rank_kv(s, r) for r in range(dp)]
        assert per_rank_kv[0] > 0.2 and all(v == 0.0 for v in per_rank_kv[1:])

        prom = PrometheusStatLogger(vc, engine_indexes=[0])
        prom.record(scheduler_stats=stats, iteration_stats=None, engine_idx=0)
        model = vc.model_config.served_model_name
        exported = _sample(REGISTRY,
                           "vllm:kv_cache_usage_perc",
                           model_name=model,
                           engine="0")
        expected = (max(per_rank_kv) if mode == "max" else sum(per_rank_kv) /
                    dp)
        assert exported == pytest.approx(expected)
        if mode == "mean":
            assert exported == pytest.approx(per_rank_kv[0] / dp)
            assert exported < per_rank_kv[0]  # the full rank is hidden

        # per-rank series: the hot rank is visible whatever the aggregate says
        base = dict(model_name=model, engine="0")
        for rank in range(dp):
            assert _sample(REGISTRY,
                           "tpu_inference:dp_rank_kv_cache_usage_perc",
                           dp_rank=str(rank),
                           **base) == pytest.approx(per_rank_kv[rank])
        # the per-rank counts add up to the exported vLLM ones
        running = sum(
            _sample(REGISTRY,
                    "tpu_inference:dp_rank_num_requests_running",
                    dp_rank=str(r),
                    **base) for r in range(dp))
        waiting = sum(
            _sample(REGISTRY,
                    "tpu_inference:dp_rank_num_requests_waiting",
                    dp_rank=str(r),
                    **base) for r in range(dp))
        assert running == _sample(REGISTRY, "vllm:num_requests_running",
                                  **base)
        assert waiting == _sample(REGISTRY, "vllm:num_requests_waiting",
                                  **base)
        assert _sample(REGISTRY,
                       "tpu_inference:dp_rank_requests_routed_total",
                       dp_rank="0",
                       **base) >= 12
    finally:
        s.shutdown()


def test_per_rank_preemptions_match_vllms_own_events(model_dir, monkeypatch):
    """A pool far too small for the traffic: requests get preempted on the
    hot rank only. Our per-rank counter must equal the PREEMPTED events vLLM
    itself emits for those requests."""
    monkeypatch.setenv("DP_SCHED_RANK_METRICS_INTERVAL_S", "0")
    vc, s = lib.make_real_dp_scheduler(model_dir,
                                       2,
                                       pool_blocks_per_rank=36,
                                       max_num_seqs=64)
    try:
        s._batch_prefills = False
        for i in range(8):
            s.add_request(
                lib.make_request(f"r{i}",
                                 seed=i,
                                 prefix_len=0,
                                 body_len=80,
                                 max_tokens=60,
                                 extra_args={"dp_rank": 0}))
        events = 0
        for _ in range(3000):
            if s.get_num_unfinished_requests() == 0:
                break
            for engine_outputs in lib.step(s).values():
                for out in engine_outputs.outputs:
                    events += sum(1 for e in (out.events or ())
                                  if e.type == EngineCoreEventType.PREEMPTED)
        assert s.get_num_unfinished_requests() == 0, "traffic must drain"
        assert events > 0, "the scenario must actually preempt"
        base = dict(model_name=vc.model_config.served_model_name, engine="0")
        counted0 = _sample(REGISTRY,
                           "tpu_inference:dp_rank_num_preemptions_total",
                           dp_rank="0",
                           **base)
        counted1 = _sample(REGISTRY,
                           "tpu_inference:dp_rank_num_preemptions_total",
                           dp_rank="1",
                           **base)
        # the registry is process-wide: compare increments of this test only
        assert counted1 == _PREEMPT_BASELINE.get(1, 0)
        assert counted0 - _PREEMPT_BASELINE.get(0, 0) == events
    finally:
        s.shutdown()


_PREEMPT_BASELINE: dict = {}


@pytest.fixture(autouse=True)
def _snapshot_preemption_counters():
    _PREEMPT_BASELINE.clear()
    for rank in (0, 1):
        value = _sample(REGISTRY,
                        "tpu_inference:dp_rank_num_preemptions_total",
                        dp_rank=str(rank))
        if value is not None:
            _PREEMPT_BASELINE[rank] = value
    yield


# ---------------------------------------------------------------------------
# 6. the cross-process path, in a fresh interpreter
# ---------------------------------------------------------------------------
def _run_probe(model_dir, mode, dp):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT] +
        [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "dp_metrics_fork_probe.py")
    proc = subprocess.run(
        [sys.executable, "-B", probe, REPO_ROOT, model_dir, mode,
         str(dp)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _series(lines, name):
    out = {}
    for ln in lines:
        if ln.startswith(name + "{"):
            labels = dict(
                kv.split("=", 1)
                for kv in ln[ln.index("{") + 1:ln.index("}")].split(","))
            out[labels["dp_rank"].strip('"')] = float(ln.rsplit(" ", 1)[1])
    return out


@pytest.mark.parametrize("dp", [2, 4])
def test_per_rank_series_reach_the_api_server_registry(model_dir, dp):
    """EngineCore forked after prometheus_client froze MutexValue; the API
    server renders PROMETHEUS_MULTIPROC_DIR. Only because DPRankMetrics
    switches the value class do the per-rank series arrive."""
    got = _run_probe(model_dir, "forced", dp)
    assert got["start_method"] == "fork"
    assert got["child"]["value_class_before"] == "MutexValue"
    assert got["child"]["value_class_after"] == "MmapedValue"
    kv = _series(got["metrics"], "tpu_inference:dp_rank_kv_cache_usage_perc")
    running = _series(got["metrics"],
                      "tpu_inference:dp_rank_num_requests_running")
    routed = _series(got["metrics"],
                     "tpu_inference:dp_rank_requests_routed_total")
    assert sorted(kv) == [str(r) for r in range(dp)]
    assert kv["0"] > 0.1 and all(kv[str(r)] == 0.0 for r in range(1, dp))
    assert running["0"] == got["child"]["combined_running"] > 0
    assert routed["0"] == 10 and all(routed[str(r)] == 0 for r in range(1, dp))
    # the mean the engine exports is the hot rank diluted by the idle ones
    assert got["child"]["combined_kv"] == pytest.approx(kv["0"] / dp, rel=1e-6)


def test_without_the_switch_the_series_would_be_invisible(model_dir):
    """Control for the test above: same scenario, value-class switch disabled."""
    got = _run_probe(model_dir, "stock", 2)
    assert got["child"]["value_class_after"] == "MutexValue"
    assert got["metrics"] == []
