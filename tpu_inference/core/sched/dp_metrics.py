# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""KV-usage aggregation and per-rank Prometheus metrics for ``DPScheduler``.

Why this exists
---------------
Each DP rank owns its own slice of the KV pool and its own vLLM scheduler, and
a request is bound to one rank for life. But the ranks are merged into ONE
``SchedulerStats`` before it leaves the engine: running and waiting are summed
and ``kv_cache_usage`` is the MEAN of the ranks. vLLM then exports a single
``vllm:kv_cache_usage_perc{engine="0"}``. With one rank full and the other
idle that gauge reads ~0.5, while the full rank is already preempting. The
mean is the right number for the pod's pooled capacity and the wrong number
for admission, because ranks cannot borrow blocks from each other.

Two independent tools, both off the hot path:

``DP_SCHED_KV_AGGREGATE=mean|max``
    How the per-rank KV usage is reduced into the exported gauge. ``mean``
    (default) is the historical behaviour; ``max`` exports the hottest rank.

per-rank metrics (``DPRankMetrics``, ``DP_SCHED_RANK_METRICS``, default on)
    ``tpu_inference:dp_rank_kv_cache_usage_perc``,
    ``tpu_inference:dp_rank_num_requests_running``,
    ``tpu_inference:dp_rank_num_requests_waiting`` (gauges),
    ``tpu_inference:dp_rank_num_preemptions_total`` and
    ``tpu_inference:dp_rank_requests_routed_total`` (counters), all with
    ``model_name``, ``engine`` and ``dp_rank`` labels.

How per-rank metrics reach ``/metrics``
---------------------------------------
vLLM's stat loggers run in the API-server process and only see the merged
``SchedulerStats``, so the per-rank numbers are published from the EngineCore
process, where ``DPScheduler`` lives, through prometheus_client's
multiprocess mode: the EngineCore writes its values into
``PROMETHEUS_MULTIPROC_DIR`` and the API server's ``/metrics`` merges that
directory (``vllm.v1.metrics.prometheus.get_prometheus_registry``).

vLLM turns multiprocess mode on whenever it runs more than one API server,
which is the default for every ``--data-parallel-size >= 2`` launch
(``--api-server-count`` defaults to the DP size). But
``prometheus_client.values.ValueClass`` is chosen when ``prometheus_client``
is first imported, which ``vllm serve`` does *before* it sets the directory,
and the EngineCore is forked from that process. So the EngineCore keeps the
in-memory ``MutexValue`` and its metrics never reach disk. ``DPRankMetrics``
therefore switches ``ValueClass`` to ``MultiProcessValue()`` in the EngineCore
when the directory is set and the in-memory class is still in place.

With a single API server vLLM does not set the directory and the per-rank
series stay local to the EngineCore. Export ``PROMETHEUS_MULTIPROC_DIR`` for
the pod (before the process starts) to get them in that configuration too.
"""
from __future__ import annotations

import os
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

METRIC_NAMESPACE = "tpu_inference"
_LABELS = ("model_name", "engine", "dp_rank")


def aggregate_kv_usage(per_rank: Sequence[Optional[float]],
                       mode: str) -> float:
    """Reduce per-rank KV usage (``None`` = rank reported nothing = 0.0).

    ``mean`` divides by the number of ranks, so a silent rank counts as idle;
    this is the arithmetic the scheduler always used.
    """
    values = [0.0 if v is None else float(v) for v in per_rank]
    if not values:
        return 0.0
    if mode == "max":
        return max(values)
    return sum(values) / len(values)


def _multiprocess_dir() -> Optional[str]:
    """PROMETHEUS_MULTIPROC_DIR (or the legacy lowercase name), if it is set."""
    return (os.environ.get("PROMETHEUS_MULTIPROC_DIR")
            or os.environ.get("prometheus_multiproc_dir") or None)


def ensure_multiprocess_value_class() -> bool:
    """Make metrics created from now on in THIS process land in the
    multiprocess directory. Returns True when a usable directory is set
    (whether or not a switch was needed). See the module docstring for why it
    is needed.

    A set-but-unusable directory (missing or not writable) leaves everything
    as it is: switching would make every metric created afterwards in this
    process fail, not just ours.
    """
    directory = _multiprocess_dir()
    if directory is None:
        return False
    if not (os.path.isdir(directory) and os.access(directory, os.W_OK)):
        logger.warning(
            "PROMETHEUS_MULTIPROC_DIR=%s is not a writable directory; "
            "per-rank DP metrics stay in this process only.", directory)
        return False
    import prometheus_client.values as values
    if getattr(values.ValueClass, "__name__", "") == "MutexValue":
        values.ValueClass = values.MultiProcessValue()
        logger.info(
            "Per-rank DP metrics: switched prometheus_client to "
            "multiprocess values (%s).", directory)
    return True


@dataclass
class _Families:
    kv_usage: Any
    running: Any
    waiting: Any
    preemptions: Any
    routed: Any


# One set of metric objects per registry: DPScheduler may be constructed more
# than once in a process (tests, engine restarts) and a metric name can only be
# registered once.
_FAMILIES: "weakref.WeakKeyDictionary[Any, _Families]" = (
    weakref.WeakKeyDictionary())


def _make_gauge(name: str, doc: str, registry: Any):
    from prometheus_client import Gauge
    try:
        # newest value wins; the EngineCore is the only writer
        return Gauge(name,
                     doc,
                     labelnames=_LABELS,
                     registry=registry,
                     multiprocess_mode="mostrecent")
    except ValueError as e:
        if "multiprocess mode" not in str(e).lower():
            raise
        return Gauge(name,
                     doc,
                     labelnames=_LABELS,
                     registry=registry,
                     multiprocess_mode="max")


def _get_families(registry: Any) -> _Families:
    from prometheus_client import REGISTRY, Counter
    registry = REGISTRY if registry is None else registry
    families = _FAMILIES.get(registry)
    if families is None:
        ns = METRIC_NAMESPACE
        families = _Families(
            kv_usage=_make_gauge(
                f"{ns}:dp_rank_kv_cache_usage_perc",
                "KV-cache usage (1 = 100%) of one DP rank's block pool.",
                registry),
            running=_make_gauge(f"{ns}:dp_rank_num_requests_running",
                                "Number of requests running on one DP rank.",
                                registry),
            waiting=_make_gauge(f"{ns}:dp_rank_num_requests_waiting",
                                "Number of requests waiting on one DP rank.",
                                registry),
            preemptions=Counter(
                f"{ns}:dp_rank_num_preemptions",
                "Cumulative number of preemptions on one DP rank.",
                labelnames=_LABELS,
                registry=registry),
            routed=Counter(
                f"{ns}:dp_rank_requests_routed",
                "Cumulative number of requests routed to one DP rank.",
                labelnames=_LABELS,
                registry=registry),
        )
        _FAMILIES[registry] = families
    return families


class DPRankMetrics:
    """Per-rank gauges and counters, updated by ``DPScheduler``.

    Never raises: a metrics problem is logged once and the metric dropped; it
    must not take the scheduler down.
    """

    def __init__(self,
                 *,
                 model_name: str,
                 dp_size: int,
                 engine: str = "0",
                 registry: Any = None,
                 interval_s: float = 1.0,
                 force_multiprocess: Optional[bool] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.dp_size = dp_size
        self._interval_s = interval_s
        self._clock = clock
        self._last_publish = float("-inf")
        self._failed = False
        self._kv = self._run = self._wait = None
        self._preempt: List[Any] = []
        self._routed: List[Any] = []
        try:
            # Only the default registry is scraped through the multiprocess
            # directory; a caller-supplied registry is its own business.
            if force_multiprocess is None:
                force_multiprocess = registry is None
            if force_multiprocess:
                ensure_multiprocess_value_class()
            fam = _get_families(registry)
            kv, run, wait, pre, routed = [], [], [], [], []
            for rank in range(dp_size):
                labels = dict(model_name=model_name,
                              engine=engine,
                              dp_rank=str(rank))
                kv.append(fam.kv_usage.labels(**labels))
                run.append(fam.running.labels(**labels))
                wait.append(fam.waiting.labels(**labels))
                pre.append(fam.preemptions.labels(**labels))
                routed.append(fam.routed.labels(**labels))
            self._kv, self._run, self._wait = kv, run, wait
            self._preempt, self._routed = pre, routed
        except Exception as e:
            self._disable("cannot register per-rank DP metrics", e)

    @classmethod
    def from_vllm_config(cls, vllm_config: Any, dp_size: int,
                         interval_s: float) -> "DPRankMetrics":
        name = getattr(getattr(vllm_config, "model_config", None),
                       "served_model_name", None)
        if not isinstance(name, str):
            name = "unknown"
        return cls(model_name=name, dp_size=dp_size, interval_s=interval_s)

    @property
    def enabled(self) -> bool:
        return not self._failed

    def _disable(self, what: str, err: Exception) -> None:
        if not self._failed:
            logger.warning("%s: %s; per-rank DP metrics are off.", what, err)
        self._failed = True

    def observe_stats(self, rank_stats: Sequence[Any]) -> None:
        """Publish per-rank running/waiting/KV from each rank's SchedulerStats
        (``None`` entries are skipped). Throttled to ``interval_s``."""
        if self._failed:
            return
        now = self._clock()
        if self._interval_s > 0 and now - self._last_publish < self._interval_s:
            return
        self._last_publish = now
        try:
            for rank, stats in enumerate(rank_stats[:self.dp_size]):
                if stats is None:
                    continue
                self._kv[rank].set(float(stats.kv_cache_usage))
                self._run[rank].set(stats.num_running_reqs)
                self._wait[rank].set(stats.num_waiting_reqs)
        except Exception as e:
            self._disable("cannot update per-rank DP gauges", e)

    def add_preemptions(self, per_rank_counts: Sequence[int]) -> None:
        if self._failed:
            return
        try:
            for rank, n in enumerate(per_rank_counts[:self.dp_size]):
                if n:
                    self._preempt[rank].inc(n)
        except Exception as e:
            self._disable("cannot update per-rank DP preemption counter", e)

    def add_routed(self, rank: int) -> None:
        if self._failed:
            return
        try:
            self._routed[rank].inc()
        except Exception as e:
            self._disable("cannot update per-rank DP routed counter", e)


def count_preemptions(rank_outputs: Sequence[Any]) -> List[int]:
    """Per-rank number of requests preempted in this scheduling step.

    ``SchedulerOutput.preempted_req_ids`` is per rank here; the combined
    DPSchedulerOutput does not carry it, so it has to be read before merging.
    """
    counts: List[int] = []
    for out in rank_outputs:
        try:
            counts.append(len(getattr(out, "preempted_req_ids", None) or ()))
        except Exception:
            counts.append(0)
    return counts
