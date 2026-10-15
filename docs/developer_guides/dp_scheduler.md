# Data-parallel scheduling (SPMD DP)

With SPMD data parallelism the DP ranks live inside one engine. A
`DPScheduler` (`tpu_inference/core/sched/dp_scheduler.py`) owns one stock vLLM
scheduler per rank, each in its own worker process with its own slice of the KV
pool, and binds every request to one rank for its whole life. It is used when
the DP size (`data_parallel_size`, or attention DP) is above 1 and DP is not
handled by vLLM itself (`TPU_MULTIPROCESS_DP=1` on trees that have it).

Because the ranks share one engine, a few things that vLLM does per engine need
a tpu-inference answer. This page lists them.

## Pinning a request to a rank

vLLM reads the `X-data-parallel-rank` HTTP header into the request and range
checks it against `parallel_config.data_parallel_size`, which tpu-inference
pins to 1 under SPMD DP. Without help, rank 0 is accepted and then ignored and
every rank >= 1 is an HTTP 400.

tpu-inference wraps `InputProcessor.process_inputs` (installed while the
`VllmConfig` is built, see `core/sched/dp_rank_hint.py`) so that:

* the header is validated against the real DP size, `[0, total_dp_size)`; a
  rank outside it is an HTTP 400 that states that range;
* a valid rank travels in `SamplingParams.extra_args["dp_rank"]`, which already
  survives the trip to the scheduler, and `DPScheduler` routes the request to
  that rank before any picker runs;
* `"vllm_xargs": {"dp_rank": N}` in the request body works the same way for
  clients that cannot set headers (not range-checked at the API; a missing,
  malformed or out-of-range value is ignored by the scheduler and the request
  is routed normally).

A pinned request waits on its rank if that rank is full; nothing moves it. Any
client that can reach the server can pin traffic to one rank, so strip the
header at the gateway if that is not wanted.

## KV usage and per-rank metrics

Running and waiting requests are summed across ranks and `kv_cache_usage` is
reduced to one number before it leaves the engine, so vLLM exports a single
`vllm:kv_cache_usage_perc{engine="0"}`. Each rank owns `1/N` of the blocks and
cannot borrow from the others, so with one rank full and the others idle the
default (the mean) reads `1/N` of the hot rank while that rank is preempting.

* `DP_SCHED_KV_AGGREGATE=mean|max` (default `mean`) selects the reduction for
  the exported gauge. `max` exports the hottest rank.
* `DP_SCHED_RANK_METRICS` (default on) and `DP_SCHED_RANK_METRICS_INTERVAL_S`
  (default 1.0) control these series, all labelled `model_name`, `engine`,
  `dp_rank`:

  | metric | type |
  |---|---|
  | `tpu_inference:dp_rank_kv_cache_usage_perc` | gauge |
  | `tpu_inference:dp_rank_num_requests_running` | gauge |
  | `tpu_inference:dp_rank_num_requests_waiting` | gauge |
  | `tpu_inference:dp_rank_num_preemptions_total` | counter |
  | `tpu_inference:dp_rank_requests_routed_total` | counter |

  The per-rank running and waiting gauges add up to `vllm:num_requests_running`
  and `vllm:num_requests_waiting`. They are written by the EngineCore process
  into `PROMETHEUS_MULTIPROC_DIR`, which the API server's `/metrics` merges.
  vLLM enables that directory whenever it runs more than one API server, the
  default for every `--data-parallel-size >= 2` launch. With a single API
  server, export `PROMETHEUS_MULTIPROC_DIR` for the process to see them on
  `/metrics`. The gauges need vLLM's stats on (not `--disable-log-stats`),
  because they come from the rank schedulers' `SchedulerStats`.

## KV events

Every rank publishes its own KV events. Rank `r` uses `data_parallel_index = r`,
so with a ZMQ endpoint `tcp://*:BASE` it binds `BASE + r` (the replay endpoint
likewise) and tags each batch `data_parallel_rank = r`, the same convention as
vLLM-native DP. A collector that binds while the publishers connect must listen
on `BASE .. BASE + dp_size - 1`. vLLM's `/kv_event_sources` carries a single
config per engine and cannot describe one publisher per rank; with a recent
vLLM each rank logs where it publishes at startup.

## Rank routing and the DPRANK log

Default routing (`DP_SCHED_ROUTING=legacy`): the rank with the longest prefix
cache hit wins and load is ignored on a hit (ties go to rank 0); on a miss, the
rank with the fewest pending prefill tokens. Requests that share a template can
therefore pile onto the rank that cached it first.

`DP_SCHED_ROUTING=load` (`core/sched/dp_rank_picker.py`) keeps prefix affinity
only while the hit rank is not meaningfully hotter than the least-loaded rank
and otherwise picks the less loaded one; it needs one extra pipelined worker
query per rank. Thresholds:

| env | default | meaning |
|---|---|---|
| `DP_SCHED_AFFINITY_MIN_BLOCKS` | 2 | minimum hit advantage, in blocks |
| `DP_SCHED_AFFINITY_MAX_KV_GAP` | 0.10 | hit rank may be this much hotter in KV usage |
| `DP_SCHED_AFFINITY_MAX_QUEUE_GAP` | 8 | ... in running + waiting requests |
| `DP_SCHED_AFFINITY_MAX_PENDING_GAP_TOKENS` | 32768 | ... in queued prefill tokens |
| `DP_SCHED_KV_BUCKET` / `DP_SCHED_KV_SATURATION` | 0.05 / 0.92 | load-order bucket width / saturation line |

A bad value is reported and falls back to its default. A client-pinned rank
always wins.

Every `DP_SCHED_RANK_LOG_INTERVAL_S` seconds (default 10, 0 disables) the
scheduler logs one line, whatever the routing mode:

```
DPRANK step=N routing=load held=0 r0_run=53 r0_wait=7 r0_skip=0 r0_kv=0.9800 r0_preempt=4 r0_routed=210 r1_run=21 ... kv_max=0.9800 kv_mean=0.5500 route_affinity=88 route_kv_gap=3
```

`preempt`, `routed` and `route_*` are totals since start.
