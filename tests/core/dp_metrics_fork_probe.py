# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Subprocess probe for tests/core/test_dp_metrics.py (not a test itself).

Reproduces how per-rank DP metrics have to travel in production:

* ``vllm serve --data-parallel-size N`` imports ``vllm.v1.metrics.prometheus``
  (and with it ``prometheus_client``) at CLI import time, which freezes
  ``prometheus_client.values.ValueClass`` to the in-memory ``MutexValue``;
* only then does ``run_multi_api_server`` call
  ``setup_multiprocess_prometheus()`` (N API servers is the default at DP N);
* the EngineCore, which hosts the DPScheduler, is forked from that process.

The parent here plays the API server: after the EngineCore child is done it
renders ``/metrics`` from ``get_prometheus_registry()``, the exact registry
vLLM's ``/metrics`` endpoint serves. The child runs a REAL DPScheduler with
real rank workers.

usage: dp_metrics_fork_probe.py <repo_root> <model_dir> <forced|stock> <dp>
Prints one JSON object: {"child": {...}, "metrics": [<exposition lines>]}.
"""
import json
import sys

repo_root, model_dir, mode, dp = sys.argv[1], sys.argv[2], sys.argv[3], int(
    sys.argv[4])
sys.path.insert(0, repo_root)

import vllm.v1.metrics.prometheus as vprom  # noqa: E402  (serve.py does this at import)
from vllm.utils.system_utils import get_mp_context  # noqa: E402

vprom.setup_multiprocess_prometheus()  # serve.py: run_multi_api_server


def engine_core(conn):
    from tests.core import dp_testlib as lib
    from tpu_inference.core.sched import dp_metrics
    if mode == "stock":
        # what a scheduler that does not switch the value class would see
        dp_metrics.ensure_multiprocess_value_class = lambda: True
    import prometheus_client.values as pv
    before = pv.ValueClass.__name__
    _, s = lib.make_real_dp_scheduler(model_dir, dp, pool_blocks_per_rank=300)
    try:
        s._batch_prefills = False
        # everything pinned to rank 0, rank 1.. idle: the hot-rank picture
        for i in range(10):
            s.add_request(
                lib.make_request(f"r{i}",
                                 seed=i,
                                 prefix_len=64,
                                 body_len=300,
                                 extra_args={"dp_rank": 0}))
        stats = None
        for _ in range(4):
            stats = lib.last_stats(lib.step(s)) or stats
        s._rank_metrics._last_publish = float("-inf")  # defeat the throttle
        stats = lib.last_stats(lib.step(s)) or stats
        conn.send({
            "value_class_before": before,
            "value_class_after": pv.ValueClass.__name__,
            "combined_running": stats.num_running_reqs,
            "combined_kv": stats.kv_cache_usage,
        })
    finally:
        s.shutdown()


if __name__ == "__main__":
    ctx = get_mp_context()
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=engine_core, args=(child_conn, ))
    proc.start()
    child = parent_conn.recv()
    proc.join()
    from prometheus_client import generate_latest
    text = generate_latest(vprom.get_prometheus_registry()).decode()
    lines = [
        ln for ln in text.splitlines()
        if ln.startswith("tpu_inference:dp_rank_")
    ]
    print(
        json.dumps({
            "child": child,
            "metrics": lines,
            "start_method": ctx.get_start_method()
        }))
