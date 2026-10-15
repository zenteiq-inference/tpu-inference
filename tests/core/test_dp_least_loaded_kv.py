# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Resident-KV term of the least_loaded picker (DP_SCHED_LEAST_LOADED_KV_WEIGHT).

``least_loaded`` scores a rank by pending prefill + uncached prompt tokens. It
never asks what the rank already holds in KV, and a prefix hit lowers the
score of the rank that cached the template. When requests that share one
template arrive spread out, pending prefill drains between arrivals, so that
rank stays the cheapest while its KV fills with decode state and the other
ranks idle (it preempts while the mean KV gauge reads about half). The weight
adds resident KV tokens to the score; 0, the default, is the unmodified
upstream behaviour.

CPU only: ``VLLM_TARGET_DEVICE=empty JAX_PLATFORMS=cpu``.
"""
import os
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request

from tests.core import dp_testlib as lib
from tpu_inference import envs
from tpu_inference.core.sched import dp_scheduler as dps

C = dps.SchedulerCommand
BLOCK = 16


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DP_SCHED_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return lib.write_tiny_llama(tmp_path_factory.mktemp("tiny-llama"))


# ---------------------------------------------------------------------------
# 1. the env
# ---------------------------------------------------------------------------
class TestEnv:

    def test_default_is_off(self):
        assert envs.DP_SCHED_LEAST_LOADED_KV_WEIGHT == 0.0

    @pytest.mark.parametrize("raw,expected", [("0", 0.0), ("0.05", 0.05),
                                              ("1", 1.0), (" 2.5 ", 2.5),
                                              ("", 0.0)])
    def test_valid(self, monkeypatch, raw, expected):
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", raw)
        assert envs.DP_SCHED_LEAST_LOADED_KV_WEIGHT == expected

    @pytest.mark.parametrize("raw", ["-1", "nan", "inf", "-inf", "heavy"])
    def test_invalid_fails_fast(self, monkeypatch, raw):
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", raw)
        with pytest.raises(ValueError,
                           match="DP_SCHED_LEAST_LOADED_KV_WEIGHT"):
            envs.DP_SCHED_LEAST_LOADED_KV_WEIGHT


# ---------------------------------------------------------------------------
# 2. the scoring, with mocked IPC
# ---------------------------------------------------------------------------
def _request(num_tokens=400):
    request = MagicMock(spec=Request)
    request.num_tokens = num_tokens
    request.sampling_params = None
    return request


def _answers(scheduler, *, cached, pending, usage, num_blocks=1000):
    """Per-rank answers to every query least_loaded sends."""
    scheduler._send_command = MagicMock()

    def answer(rank, command=None):
        return {
            C.PROBE_COMPUTED_BLOCKS: cached[rank],
            C.GET_PENDING_PREFILL_TOKENS: pending[rank],
            C.GET_REQUEST_COUNTS: (0, 0),
            C.GET_MIN_REMAINING_OUTPUT: 5,
            C.GET_KV_USAGE: (usage[rank], num_blocks),
        }[command]

    scheduler._get_result = MagicMock(side_effect=answer)


def _sent(scheduler):
    return [c.args[1] for c in scheduler._send_command.call_args_list]


class TestScoring:

    def test_off_never_asks_for_kv_and_is_upstream(self, monkeypatch):
        s = lib.make_mocked_dp_scheduler(2, block_size=BLOCK)
        _answers(s, cached=[320, 0], pending=[0, 0], usage=[0.9, 0.0])
        # rank 0 holds the template (320 cached tokens): upstream picks it even
        # at 90% KV
        assert s._find_best_rank_for_request(_request()) == 0
        assert C.GET_KV_USAGE not in _sent(s)

    def test_a_hot_rank_loses_its_prefix_hit(self, monkeypatch):
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", "1.0")
        s = lib.make_mocked_dp_scheduler(2, block_size=BLOCK)
        # rank 0: 90% of 1000 blocks x 16 = 14400 resident tokens, hit saves 320
        _answers(s, cached=[320, 0], pending=[0, 0], usage=[0.9, 0.0])
        assert s._find_best_rank_for_request(_request()) == 1
        assert _sent(s).count(C.GET_KV_USAGE) == 2

    def test_equal_residency_keeps_the_prefix_hit(self, monkeypatch):
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", "1.0")
        s = lib.make_mocked_dp_scheduler(2, block_size=BLOCK)
        _answers(s, cached=[320, 0], pending=[0, 0], usage=[0.4, 0.4])
        assert s._find_best_rank_for_request(_request()) == 0

    def test_flip_point(self, monkeypatch):
        """Resident gap 2000 tokens, the hit saves 320: it survives while
        gap x weight < 320, i.e. weight < 0.16."""
        usage = [2000 / (1000 * BLOCK), 0.0]
        for weight, expected in ((0.15, 0), (0.17, 1)):
            monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", str(weight))
            s = lib.make_mocked_dp_scheduler(2, block_size=BLOCK)
            _answers(s, cached=[320, 0], pending=[0, 0], usage=usage)
            assert s._find_best_rank_for_request(_request()) == expected

    @pytest.mark.parametrize("dp", [2, 3, 4, 8])
    def test_picks_the_emptiest_rank_at_any_dp_size(self, monkeypatch, dp):
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", "1.0")
        s = lib.make_mocked_dp_scheduler(dp, block_size=BLOCK)
        usage = [0.5 + 0.01 * r for r in range(dp)]
        usage[dp - 2] = 0.1  # the emptiest
        _answers(s, cached=[0] * dp, pending=[0] * dp, usage=usage)
        assert s._find_best_rank_for_request(_request()) == dp - 2

    def test_pending_prefill_still_counts(self, monkeypatch):
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", "1.0")
        s = lib.make_mocked_dp_scheduler(2, block_size=BLOCK)
        _answers(s, cached=[0, 0], pending=[50000, 0], usage=[0.0, 0.0])
        assert s._find_best_rank_for_request(_request()) == 1

    def test_round_robin_and_hints_never_query_kv(self, monkeypatch):
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", "1.0")
        monkeypatch.setenv("DP_SCHED_ROUTING", "round_robin")
        s = lib.make_mocked_dp_scheduler(2, block_size=BLOCK)
        s._send_command = MagicMock()
        s._get_result = MagicMock()
        assert [s._pick_rank_for_request(_request())
                for _ in range(3)] == [0, 1, 0]
        hinted = _request()
        hinted.sampling_params = SimpleNamespace(extra_args={"dp_rank": 1})
        assert s._pick_rank_for_request(hinted) == 1
        s._send_command.assert_not_called()


# ---------------------------------------------------------------------------
# 3. the real thing
# ---------------------------------------------------------------------------
def test_get_kv_usage_matches_the_real_scheduler(model_dir):
    _, s = lib.make_real_dp_scheduler(model_dir, 2, pool_blocks_per_rank=200)
    try:
        s._batch_prefills = False
        s._send_command(0, C.GET_KV_USAGE)
        assert s._get_result(0, C.GET_KV_USAGE) == (0.0, 200)
        for i in range(4):
            s.add_request(
                lib.make_request(f"r{i}",
                                 seed=i,
                                 prefix_len=0,
                                 body_len=300,
                                 extra_args={"dp_rank": 0}))
        lib.step(s)
        s._send_command(0, C.GET_KV_USAGE)
        usage, num_blocks = s._get_result(0, C.GET_KV_USAGE)
        s._send_command(0, C.MAKE_STATS, (None, None))
        stats = s._get_result(0, C.MAKE_STATS)
        assert num_blocks == 200
        assert usage == pytest.approx(stats.kv_cache_usage) and usage > 0.3
        s._send_command(1, C.GET_KV_USAGE)
        assert s._get_result(1, C.GET_KV_USAGE) == (0.0, 200)
    finally:
        s.shutdown()


def _trickle(model_dir, weight, monkeypatch, n=32):
    """One request per 3 steps, every request = the same 320-token template +
    its own 48 tokens (a single-template campaign as one pod sees it)."""
    if weight:
        monkeypatch.setenv("DP_SCHED_LEAST_LOADED_KV_WEIGHT", str(weight))
    _, s = lib.make_real_dp_scheduler(model_dir,
                                      2,
                                      pool_blocks_per_rank=200,
                                      max_num_seqs=64)
    preemptions = Counter()
    peak = [0.0, 0.0]

    def run(steps):
        for _ in range(steps):
            for outputs in lib.step(s).values():
                for out in outputs.outputs:
                    for event in out.events or ():
                        if event.type == EngineCoreEventType.PREEMPTED:
                            preemptions[s.assigned_dp_rank[
                                out.request_id]] += 1

    try:
        for i in range(n):
            s.add_request(
                lib.make_request(f"t{i}",
                                 prefix_len=320,
                                 body_len=48,
                                 seed=i,
                                 max_tokens=120))
            run(3)
            for rank in range(2):
                s._send_command(rank, C.MAKE_STATS, (None, None))
                kv = s._get_result(rank, C.MAKE_STATS).kv_cache_usage
                peak[rank] = max(peak[rank], kv)
        routed = lib.routed_per_rank(s)
        run(150)
        return routed, preemptions, peak
    finally:
        s.shutdown()


def test_trickle_of_one_template_upstream_vs_weighted(model_dir, monkeypatch):
    """The production signature on the real scheduler: upstream least_loaded
    sends nearly the whole campaign to the rank that cached the template, which
    fills (peak KV 1.0) and preempts while the other rank peaks near 0.16; the
    weighted picker splits it and nobody preempts."""
    routed, preemptions, peak = _trickle(model_dir, 0, monkeypatch)
    assert routed.get(0, 0) >= 26 and peak[0] >= 0.99, (routed, peak)
    assert preemptions[0] > 0 and peak[1] < 0.3, (preemptions, peak)

    routed, preemptions, peak = _trickle(model_dir, 1.0, monkeypatch)
    assert abs(routed[0] - routed[1]) <= 4, routed
    assert not preemptions, preemptions
    assert max(peak) < 0.9, peak
