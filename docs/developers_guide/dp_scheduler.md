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

## Rank routing

`DP_SCHED_ROUTING=least_loaded` (default) sends a request to the rank with the
smallest pending prefill + uncached prompt tokens; `round_robin` cycles. Neither
asks what a rank already holds in KV. When requests that share one template
arrive spread out, pending prefill drains between arrivals and the rank that
cached the template stays the cheapest while its KV fills and the others idle.

`DP_SCHED_LEAST_LOADED_KV_WEIGHT` (float >= 0, default 0 = off) adds
`weight x resident KV tokens` to a rank's load in `least_loaded`. A prefix hit
still wins while (resident-KV gap in tokens) x weight is below the tokens the
hit saves: 1.0 balances resident KV almost exactly, a small weight keeps
affinity until the ranks are far apart. It costs one more pipelined worker query
per rank, only when the weight is above 0. A client-pinned rank always wins.
