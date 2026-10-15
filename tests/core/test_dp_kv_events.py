# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""KV-cache events with the SPMD DPScheduler: one publisher identity per rank.

Every rank worker builds a stock vLLM ``Scheduler`` from the same config, and
vLLM derives the KV-event publisher's identity (ZMQ port offset, the
``data_parallel_rank`` tag on each batch) from
``parallel_config.data_parallel_index``, which is 0 for all of them under SPMD
DP. With a binding endpoint N-1 rank workers died with "Address already in
use"; with a connecting one the ranks were indistinguishable. These tests use
real ZMQ sockets and a real DPScheduler with real rank workers.

CPU only: ``VLLM_TARGET_DEVICE=empty JAX_PLATFORMS=cpu``.
"""
import multiprocessing
import socket
import time
from types import SimpleNamespace

import cloudpickle
import msgspec
import pytest
import zmq
from vllm.config import KVEventsConfig
from vllm.distributed.kv_events import KVEventBatch

from tests.core import dp_testlib as lib
from tpu_inference.core.sched import dp_scheduler as dps
from tpu_inference.core.sched.dp_scheduler import (SchedulerCommand,
                                                   _vllm_config_for_rank)

DP_SIZES = [2, 3, 4]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return lib.write_tiny_llama(tmp_path_factory.mktemp("tiny-llama"))


@pytest.fixture(autouse=True)
def _no_routing_env(monkeypatch):
    monkeypatch.delenv("DP_SCHED_ROUTING", raising=False)


def _free_port_block(n: int) -> int:
    """First port of ``n`` consecutive free TCP ports (low range, so it cannot
    collide with ephemeral ports)."""
    import random
    rng = random.Random()
    for _ in range(200):
        base = rng.randrange(21000, 29000)
        socks = []
        try:
            for i in range(n):
                s = socket.socket()
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                s.bind(("127.0.0.1", base + i))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port block")


def _kv_config(endpoint, replay_endpoint=None):
    return KVEventsConfig(enable_kv_cache_events=True,
                          publisher="zmq",
                          endpoint=endpoint,
                          replay_endpoint=replay_endpoint,
                          topic="kv")


# ---------------------------------------------------------------------------
# 1. the per-rank config
# ---------------------------------------------------------------------------
class _ParallelConfig:

    def __init__(self):
        self.data_parallel_index = 0
        self.tensor_parallel_size = 8


class _VllmConfig:

    def __init__(self):
        self.parallel_config = _ParallelConfig()
        self.cache_config = object()


class TestConfigForRank:

    @pytest.mark.parametrize("rank", [0, 1, 3, 7])
    def test_index_is_the_rank_on_a_copy(self, rank):
        original = _VllmConfig()
        rank_config = _vllm_config_for_rank(original, rank)
        assert rank_config.parallel_config.data_parallel_index == rank
        # the caller's config is not touched
        assert original.parallel_config.data_parallel_index == 0
        assert rank_config is not original
        assert rank_config.parallel_config is not original.parallel_config
        # everything else is shared, not copied deep
        assert rank_config.cache_config is original.cache_config
        assert rank_config.parallel_config.tensor_parallel_size == 8

    def test_ranks_do_not_see_each_other(self):
        original = _VllmConfig()
        configs = [_vllm_config_for_rank(original, r) for r in range(4)]
        assert [c.parallel_config.data_parallel_index
                for c in configs] == [0, 1, 2, 3]

    @pytest.mark.parametrize("config", [None, SimpleNamespace()])
    def test_configs_without_parallel_config_pass_through(self, config):
        assert _vllm_config_for_rank(config, 1) is config

    def test_real_vllm_config(self, model_dir):
        vc = lib.make_vllm_config(model_dir)
        rank_config = _vllm_config_for_rank(vc, 2)
        assert rank_config.parallel_config.data_parallel_index == 2
        assert vc.parallel_config.data_parallel_index == 0
        assert rank_config.model_config is vc.model_config
        assert rank_config.kv_events_config is vc.kv_events_config

    def test_copy_failure_keeps_the_worker_starting(self):

        class Unreadable:

            def __copy__(self):
                raise RuntimeError("no copy")

        config = SimpleNamespace(parallel_config=Unreadable())
        assert _vllm_config_for_rank(config, 1) is config


class TestEndpointLog:

    def test_logs_where_the_rank_publishes(self):
        from unittest.mock import patch
        scheduler = SimpleNamespace(get_kv_event_publisher_config=lambda: SimpleNamespace(
            endpoint="tcp://10.0.0.5:5558", replay_endpoint=None))
        with patch.object(dps.logger, "info") as info:
            dps._log_kv_events_endpoint(scheduler, 1)
        assert info.call_count == 1
        assert info.call_args.args[1:] == (1, "tcp://10.0.0.5:5558", None)

    @pytest.mark.parametrize("scheduler", [
        SimpleNamespace(),  # older vLLM: no such method
        SimpleNamespace(get_kv_event_publisher_config=lambda: None),
    ])
    def test_silent_without_a_publisher(self, scheduler):
        from unittest.mock import patch
        with patch.object(dps.logger, "info") as info:
            dps._log_kv_events_endpoint(scheduler, 0)
        info.assert_not_called()

    def test_never_raises(self):

        def boom():
            raise RuntimeError("no")

        dps._log_kv_events_endpoint(
            SimpleNamespace(get_kv_event_publisher_config=boom), 0)


def test_each_worker_builds_its_scheduler_with_its_own_index():
    """The real worker loop, in a forked process per rank."""

    class IndexProbe:
        """Stands in for vLLM's Scheduler: reports the index it was given."""

        def __init__(self, vllm_config, **kwargs):
            self.index = vllm_config.parallel_config.data_parallel_index

        def get_num_unfinished_requests(self):
            return self.index

        def shutdown(self):
            pass

    ctx = multiprocessing.get_context("fork")
    seen = []
    for rank in range(4):
        in_p, in_c = ctx.Pipe()
        out_p, out_c = ctx.Pipe()
        proc = ctx.Process(target=dps._scheduler_worker_process,
                           args=(rank, in_c, out_c, _VllmConfig(), None, None,
                                 16, 16, None, False, True, IndexProbe))
        proc.start()
        try:
            in_p.send_bytes(
                cloudpickle.dumps(
                    (SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS, None)))
            seen.append(cloudpickle.loads(out_p.recv_bytes()))
        finally:
            proc.terminate()
            proc.join(5)
    assert seen == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 2. real ZMQ, real DPScheduler
# ---------------------------------------------------------------------------
def _subscribe(ports):
    ctx = zmq.Context.instance()
    socks = {}
    for port in ports:
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(f"tcp://127.0.0.1:{port}")
        socks[port] = sock
    return socks


def _publish_and_collect(scheduler, per_rank_ports, dp, timeout_s=15.0):
    """Pin requests to every rank, run steps, return for each port the set of
    ``data_parallel_rank`` tags and event kinds seen on it."""
    decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
    socks = _subscribe(per_rank_ports)
    seen = {port: {"tags": set(), "kinds": set()} for port in per_rank_ports}
    try:
        time.sleep(1.0)  # let the SUB connections establish (slow joiner)
        scheduler._batch_prefills = False
        deadline = time.time() + timeout_s
        round_no = 0
        while time.time() < deadline and not all(
                v["kinds"] for v in seen.values()):
            for rank in range(dp):
                scheduler.add_request(
                    lib.make_request(f"r{round_no}-{rank}",
                                     seed=round_no * 10 + rank,
                                     prefix_len=0,
                                     body_len=150,
                                     max_tokens=4,
                                     extra_args={"dp_rank": rank}))
            round_no += 1
            for _ in range(4):
                lib.step(scheduler)
            for port, sock in socks.items():
                while sock.poll(50):
                    _topic, _seq, payload = sock.recv_multipart()
                    batch = decoder.decode(payload)
                    seen[port]["tags"].add(batch.data_parallel_rank)
                    seen[port]["kinds"].update(
                        type(e).__name__ for e in batch.events)
    finally:
        for sock in socks.values():
            sock.close(linger=0)
    return seen


@pytest.mark.parametrize("dp", DP_SIZES)
def test_binding_endpoint_gives_every_rank_its_own_port_and_tag(
        model_dir, dp):
    """The default / llm-d style wiring: tcp://*:BASE. Rank r publishes on
    BASE + r and tags its batches r; no worker dies."""
    base = _free_port_block(dp)
    _, s = lib.make_real_dp_scheduler(
        model_dir, dp, kv_events_config=_kv_config(f"tcp://*:{base}"))
    try:
        assert all(p.is_alive() for p in s.processes)
        assert s.get_num_unfinished_requests() == 0  # the engine's idle poll
        seen = _publish_and_collect(s, [base + r for r in range(dp)], dp)
        for rank in range(dp):
            got = seen[base + rank]
            assert got["tags"] == {rank}, (rank, seen)
            assert "BlockStored" in got["kinds"], (rank, seen)
        assert all(p.is_alive() for p in s.processes)
    finally:
        s.shutdown()


def test_replay_endpoint_is_per_rank_too(model_dir):
    """A replay ROUTER always binds; it used to take the port for rank 0 and
    kill the rest."""
    dp = 2
    base = _free_port_block(dp)
    replay = _free_port_block(dp)
    _, s = lib.make_real_dp_scheduler(model_dir,
                                      dp,
                                      kv_events_config=_kv_config(
                                          f"tcp://*:{base}",
                                          f"tcp://*:{replay}"))
    try:
        assert all(p.is_alive() for p in s.processes)
        assert s.get_num_unfinished_requests() == 0
        seen = _publish_and_collect(s, [base, base + 1], dp)
        assert seen[base]["tags"] == {0} and seen[base + 1]["tags"] == {1}
    finally:
        s.shutdown()


def test_connecting_endpoint_keeps_ranks_distinguishable(model_dir):
    """A collector that binds and publishers that connect (vLLM's rule for
    endpoints without a wildcard). Previously all ranks connected to the same
    port and every batch was tagged 0; now rank r connects to BASE + r."""
    dp = 2
    base = _free_port_block(dp)
    ctx = zmq.Context.instance()
    subs = {}
    for r in range(dp):  # external subscribers that bind
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.bind(f"tcp://127.0.0.1:{base + r}")
        subs[base + r] = sock
    _, s = lib.make_real_dp_scheduler(
        model_dir, dp, kv_events_config=_kv_config(f"tcp://127.0.0.1:{base}"))
    try:
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        s._batch_prefills = False
        tags = {port: set() for port in subs}
        deadline = time.time() + 15
        round_no = 0
        while time.time() < deadline and not all(tags.values()):
            for rank in range(dp):
                s.add_request(
                    lib.make_request(f"r{round_no}-{rank}",
                                     seed=round_no * 10 + rank,
                                     prefix_len=0,
                                     body_len=150,
                                     max_tokens=4,
                                     extra_args={"dp_rank": rank}))
            round_no += 1
            for _ in range(4):
                lib.step(s)
            for port, sock in subs.items():
                while sock.poll(50):
                    _t, _q, payload = sock.recv_multipart()
                    tags[port].add(decoder.decode(payload).data_parallel_rank)
        assert tags == {base: {0}, base + 1: {1}}
    finally:
        s.shutdown()
        for sock in subs.values():
            sock.close(linger=0)


def test_without_the_fix_binding_workers_die(model_dir, monkeypatch):
    """Control: with the per-rank index disabled the bug comes back exactly as
    reported: workers 1..N-1 die and the engine's next scheduler call fails."""
    monkeypatch.setattr(dps, "_vllm_config_for_rank",
                        lambda vllm_config, rank: vllm_config)
    dp = 3
    base = _free_port_block(1)
    _, s = lib.make_real_dp_scheduler(
        model_dir, dp, kv_events_config=_kv_config(f"tcp://*:{base}"))
    try:
        time.sleep(1.0)
        alive = [p.is_alive() for p in s.processes]
        assert alive == [True, False, False]
        with pytest.raises((RuntimeError, BrokenPipeError, OSError)):
            s.get_num_unfinished_requests()
    finally:
        s.shutdown()


def test_events_off_changes_nothing(model_dir):
    """No kv_events_config: ranks still build, nothing binds."""
    _, s = lib.make_real_dp_scheduler(model_dir, 2)
    try:
        assert all(p.is_alive() for p in s.processes)
        assert s.get_num_unfinished_requests() == 0
    finally:
        s.shutdown()
