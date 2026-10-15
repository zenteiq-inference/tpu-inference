# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""The per-rank ``DPRANK`` log line.

``update_from_output`` is the one place the per-rank ``SchedulerStats`` still
exist: right after it they are merged (running/waiting summed, KV reduced) and
the per-rank picture is gone. The DPRANK line is how an operator sees one rank
full while the exported gauge reads half of it, with legacy or load routing,
and without Prometheus.

CPU only: ``VLLM_TARGET_DEVICE=empty JAX_PLATFORMS=cpu``.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats

from tests.core import dp_testlib as lib
from tpu_inference.core.sched import dp_scheduler as dps


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DP_SCHED_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return lib.write_tiny_llama(tmp_path_factory.mktemp("tiny-llama"))


def _make(monkeypatch, env=None, **kwargs):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return lib.make_mocked_dp_scheduler(**kwargs)


def _stats(run, wait, kv, skip=0):
    return SchedulerStats(num_running_reqs=run,
                          num_waiting_reqs=wait,
                          num_skipped_waiting_reqs=skip,
                          kv_cache_usage=kv)


def _line(info):
    return info.call_args.args[0] % info.call_args.args[1:]


class TestDPRankLog:

    def test_logs_one_line_with_both_ranks(self, monkeypatch):
        s = _make(monkeypatch)
        s._schedule_step_count = 42
        with patch.object(dps.logger, "info") as info:
            s._maybe_log_rank_stats(
                [_stats(53, 7, 0.98, skip=2),
                 _stats(21, 0, 0.12)])
        assert info.call_count == 1
        line = _line(info)
        assert line.startswith("DPRANK step=42 routing=legacy ")
        for frag in ("r0_run=53", "r0_wait=7", "r0_skip=2", "r0_kv=0.9800",
                     "r1_run=21", "r1_wait=0", "r1_kv=0.1200", "kv_max=0.9800",
                     "kv_mean=0.5500", "r0_preempt=0", "r1_routed=0",
                     "held=0"):
            assert frag in line, (frag, line)
        # logfmt-parsable: no spaces inside values
        assert all("=" in tok for tok in line.split()[1:]), line

    @pytest.mark.parametrize("dp", [2, 3, 4, 8])
    def test_every_rank_has_a_field_at_any_dp_size(self, monkeypatch, dp):
        s = _make(monkeypatch, dp=dp)
        with patch.object(dps.logger, "info") as info:
            s._maybe_log_rank_stats([_stats(r, 0, r / 10) for r in range(dp)])
        line = _line(info)
        for r in range(dp):
            assert f"r{r}_run={r} " in line and f"r{r}_routed=0" in line
        assert f"kv_max={(dp - 1) / 10:.4f}" in line

    def test_routing_mode_and_reasons_are_in_the_line(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_ROUTING": "load"})
        s._route_reasons.update({"kv_gap": 3, "affinity": 9})
        s._rank_routed_total[:] = [5, 7]
        with patch.object(dps.logger, "info") as info:
            s._maybe_log_rank_stats([_stats(1, 0, .1), _stats(1, 0, .1)])
        line = _line(info)
        assert " routing=load " in line
        assert "route_affinity=9 route_kv_gap=3" in line
        assert "r0_routed=5" in line and "r1_routed=7" in line

    def test_interval_gates_and_zero_disables(self, monkeypatch):
        s = _make(monkeypatch, {"DP_SCHED_RANK_LOG_INTERVAL_S": "3600"})
        with patch.object(dps.logger, "info") as info:
            s._maybe_log_rank_stats([_stats(1, 0, .1), _stats(1, 0, .1)])
            s._maybe_log_rank_stats([_stats(1, 0, .1), _stats(1, 0, .1)])
        assert info.call_count == 1
        s = _make(monkeypatch, {"DP_SCHED_RANK_LOG_INTERVAL_S": "0"})
        with patch.object(dps.logger, "info") as info:
            s._maybe_log_rank_stats([_stats(1, 0, .1), _stats(1, 0, .1)])
        assert info.call_count == 0

    @pytest.mark.parametrize("raw", ["ten", "nan", ""])
    def test_bad_interval_env_falls_back_to_10s(self, monkeypatch, raw):
        s = _make(monkeypatch, {"DP_SCHED_RANK_LOG_INTERVAL_S": raw})
        assert s._rank_log_interval_s == 10.0

    def test_none_stats_and_garbage_never_raise(self, monkeypatch):
        s = _make(monkeypatch)
        with patch.object(dps.logger, "info") as info:
            s._maybe_log_rank_stats([None, _stats(1, 0, .1)])
        assert "r0_kv=NA" in _line(info)
        s = _make(monkeypatch)
        s._maybe_log_rank_stats([object(), object()])  # swallowed

    def test_startup_line_reports_flush_timeout_and_interval(
            self, monkeypatch):
        with patch.object(dps.logger, "info") as info:
            _make(
                monkeypatch, {
                    "DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS": "1000",
                    "DP_SCHED_RANK_LOG_INTERVAL_S": "5",
                })
        lines = [c.args[0] % c.args[1:] for c in info.call_args_list]
        assert any("flush timeout=1000 ms" in x and "interval=5.0 s" in x
                   for x in lines), lines

    def test_schedule_counts_preemptions_per_rank(self, monkeypatch):
        s = _make(monkeypatch)

        def out(preempted):
            o = MagicMock(spec=SchedulerOutput)
            o.scheduled_new_reqs = []
            o.num_scheduled_tokens = {}
            o.total_num_scheduled_tokens = 0
            o.finished_req_ids = set()
            o.scheduled_cached_reqs = CachedRequestData(req_ids=[],
                                                        resumed_req_ids=[],
                                                        new_token_ids=[],
                                                        all_token_ids={},
                                                        new_block_ids=[],
                                                        num_computed_tokens=[],
                                                        num_output_tokens=[])
            o.scheduled_spec_decode_tokens = {}
            o.scheduled_encoder_inputs = {}
            o.num_common_prefix_blocks = []
            o.preempted_req_ids = preempted
            return o

        s._send_command = MagicMock()
        s._collect_results_unordered = MagicMock(side_effect=[
            [out({"a", "b"}), out(None)],
            [out(set()), out({"c"})],
        ])
        s.schedule()
        s.schedule()
        assert s._rank_preempt_total == [2, 1]

    def test_update_from_output_logs_per_rank_and_keeps_the_combined_stats(
            self, monkeypatch):
        s = _make(monkeypatch)
        s._rank_log_last = 0.0
        s.cached_schedulers_output.append([MagicMock(), MagicMock()])
        s._split_model_output_by_rank = MagicMock(return_value=[None, None])
        s._send_command = MagicMock()
        r0 = {0: EngineCoreOutputs(scheduler_stats=_stats(60, 9, 0.99))}
        r1 = {0: EngineCoreOutputs(scheduler_stats=_stats(30, 0, 0.11))}
        s._collect_results_unordered = MagicMock(return_value=[r0, r1])
        so = MagicMock()
        so.finished_req_ids = set()
        with patch.object(dps.logger, "info") as info:
            out = s.update_from_output(so, MagicMock())
        line = _line(info)
        assert "r0_kv=0.9900" in line and "r1_kv=0.1100" in line
        assert "r0_run=60" in line and "r1_run=30" in line
        # the exported (combined) stats are untouched: sum, sum, MEAN
        cs = out[0].scheduler_stats
        assert (cs.num_running_reqs, cs.num_waiting_reqs) == (90, 9)
        assert cs.kv_cache_usage == pytest.approx(0.55)


def test_real_scheduler_line_shows_the_hot_rank(model_dir, monkeypatch):
    """Real DPScheduler and vLLM schedulers: traffic pinned to rank 0 shows up
    as a hot rank 0 and an idle rank 1 in the DPRANK line, while the combined
    stats leaving the engine carry only the diluted mean."""
    monkeypatch.setenv("DP_SCHED_RANK_LOG_INTERVAL_S", "0.0001")
    _, s = lib.make_real_dp_scheduler(model_dir, 2, pool_blocks_per_rank=200)
    lines = []
    original = dps.logger.info
    dps.logger.info = lambda fmt, *a, **k: (lines.append(
        fmt % a) if fmt.startswith("DPRANK") else original(fmt, *a, **k))
    try:
        s._batch_prefills = False
        for i in range(12):
            s.add_request(
                lib.make_request(f"r{i}",
                                 seed=i,
                                 prefix_len=64,
                                 body_len=300,
                                 extra_args={"dp_rank": 0}))
        stats = None
        for _ in range(4):
            stats = lib.last_stats(lib.step(s)) or stats
    finally:
        dps.logger.info = original
        s.shutdown()
    assert lines, "no DPRANK line was logged"
    last = lines[-1]
    fields = dict(tok.split("=", 1) for tok in last.split()[1:])
    assert int(fields["r0_run"]) + int(fields["r0_wait"]) == 12
    assert int(fields["r1_run"]) == 0 and int(fields["r1_wait"]) == 0
    assert float(fields["r0_kv"]) > 0.2 and float(fields["r1_kv"]) == 0.0
    assert int(fields["r0_routed"]) == 12 and int(fields["r1_routed"]) == 0
    assert int(fields["route_hint"]) == 12
    assert float(fields["kv_mean"]) == pytest.approx(float(fields["kv_max"]) /
                                                     2,
                                                     rel=1e-3)
    assert stats.kv_cache_usage == pytest.approx(float(fields["kv_mean"]),
                                                 rel=1e-3)
