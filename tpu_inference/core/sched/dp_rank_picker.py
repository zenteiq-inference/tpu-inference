# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Load-aware DP rank picker for ``DPScheduler`` (pure functions, no I/O).

Why this exists
---------------
``DPScheduler._find_best_rank_for_request`` routes each new request like this:

    probe every rank for a prefix hit
    if any hit > 0: the rank with the longest hit wins (strict '>', so ties go
                    to rank 0) and LOAD IS IGNORED
    else:           the rank with the fewest pending prefill tokens (ties -> 0)

The assignment is sticky for the request's life, so a template that one rank
cached first keeps landing on that rank while the other rank idles, and the
resident decode state of a rank (the thing that fills its KV pool and causes
preemptions) is never looked at. The exported ``vllm:kv_cache_usage_perc`` is
the mean of the ranks, which hides the hot one.

This picker keeps prefix affinity only when it is worth it and the hit rank is
not meaningfully hotter; otherwise it picks the less-loaded rank. It is used
when ``DP_SCHED_ROUTING=load`` (default ``legacy`` keeps the old behaviour).

Interface
---------
    pick_rank(request_probe_hits, rank_state, block_size) -> int

    request_probe_hits : list[int]  prefix-cache hit length in TOKENS per rank, i.e.
                         exactly what SchedulerCommand.PROBE_COMPUTED_BLOCKS returns
                         (0 when prefix caching is off or nothing is cached)
    rank_state         : list[dict] one dict per rank. Every key is optional and
                         defaults to 0:
        running                 int    len(scheduler.running)
        waiting                 int    len(waiting) + len(skipped_waiting)
        pending_prefill_tokens  int    prompt tokens still owed (running remainder
                                       + every waiting prompt)
        kv_usage                float  0..1, KVCacheManager.usage of that rank's pool
        num_blocks              int    that rank's pool size (blocks); lets the picker
                                       project queued prefill onto KV so a burst routed
                                       within one scheduler step still spreads
        max_seqs                int    per-rank max_num_seqs (0 = unknown)
        routed_total            int    requests routed to the rank so far (tie-break)
    block_size         : int  tokens per KV block (the scheduler block size)

    Optional keyword args: cfg (PickerConfig), info (dict, filled with the decision).

Algorithm
---------
1. load order: lexicographic (saturated, projected-KV bucket, running+waiting,
   pending prefill tokens), then fewest routed so far, then a ROTATING cursor.
   Nothing ever ties to rank 0 by default.
     saturated  = running+waiting >= max_seqs, or projected KV >= kv_saturation
     projected  = kv_usage + pending_prefill_tokens / (num_blocks * block_size)
     bucket     = int(projected / kv_bucket)  (default 0.05, so 0.41 vs 0.44 tie
                  and the queue length decides)
2. no hit anywhere        -> least-loaded rank.
3. hit rank == least-loaded rank -> that rank.
4. otherwise keep affinity ONLY IF ALL hold, else take the least-loaded rank:
     hit advantage over the least-loaded rank >= min_affinity_blocks (default 2)
     hit rank is not saturated while the least-loaded rank is not
     kv_usage(hit) - kv_usage(low)                 <= max_kv_gap       (0.10)
     (running+waiting)(hit) - (running+waiting)(low) <= max_queue_gap  (8)
     pending_prefill(hit) - pending_prefill(low)     <= max_pending_gap_tokens (32768)

Environment (``PickerConfig.from_env``; a bad value falls back to its default
and is listed in ``cfg.bad_env`` instead of crashing the engine):

    DP_SCHED_AFFINITY_MIN_BLOCKS               2      minimum hit advantage, in blocks
    DP_SCHED_AFFINITY_MAX_KV_GAP               0.10   hit rank may be this much hotter in kv_usage
    DP_SCHED_AFFINITY_MAX_QUEUE_GAP            8      ... in running+waiting requests
    DP_SCHED_AFFINITY_MAX_PENDING_GAP_TOKENS   32768  ... in queued prefill tokens
    DP_SCHED_KV_BUCKET                         0.05   load-order bucket width
    DP_SCHED_KV_SATURATION                     0.92   saturation line
"""
import os
from dataclasses import dataclass

_RR = [0]  # rotating tie-break cursor (module state, one engine process)


@dataclass(frozen=True)
class PickerConfig:
    min_affinity_blocks: int = 2
    max_kv_gap: float = 0.10
    max_queue_gap: int = 8
    max_pending_gap_tokens: int = 32768
    kv_bucket: float = 0.05
    kv_saturation: float = 0.92
    bad_env: tuple = ()

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        spec = (
            ("DP_SCHED_AFFINITY_MIN_BLOCKS", "min_affinity_blocks", int),
            ("DP_SCHED_AFFINITY_MAX_KV_GAP", "max_kv_gap", float),
            ("DP_SCHED_AFFINITY_MAX_QUEUE_GAP", "max_queue_gap", int),
            ("DP_SCHED_AFFINITY_MAX_PENDING_GAP_TOKENS",
             "max_pending_gap_tokens", int),
            ("DP_SCHED_KV_BUCKET", "kv_bucket", float),
            ("DP_SCHED_KV_SATURATION", "kv_saturation", float),
        )
        kwargs = {}
        bad = []
        for env_name, field_name, cast in spec:
            raw = env.get(env_name)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                value = cast(str(raw).strip())
            except ValueError:
                bad.append(env_name)
                continue
            # Reject nan, inf and absurd magnitudes (they parse as floats but turn
            # the picker into legacy affinity or crash it), and a kv_bucket so small
            # that int(projected / kv_bucket) overflows. kv_bucket 0 = no bucketing.
            if not abs(value) < 1e9 or (field_name == "kv_bucket"
                                        and value != 0 and not value >= 1e-4):
                bad.append(env_name)
                continue
            kwargs[field_name] = value
        return cls(bad_env=tuple(bad), **kwargs)


def _num(state, key, default=0):
    value = state.get(key, default)
    return default if value is None else value


def _proj_kv(state, block_size):
    """kv_usage plus queued prefill expressed as a fraction of the rank's pool."""
    kv = float(_num(state, "kv_usage", 0.0))
    capacity = int(_num(state, "num_blocks", 0)) * block_size
    if capacity > 0:
        kv += float(_num(state, "pending_prefill_tokens", 0)) / capacity
    return kv


def _queue(state):
    return int(_num(state, "running", 0)) + int(_num(state, "waiting", 0))


def _saturated(state, block_size, cfg):
    max_seqs = int(_num(state, "max_seqs", 0))
    if max_seqs > 0 and _queue(state) >= max_seqs:
        return True
    return _proj_kv(state, block_size) >= cfg.kv_saturation


def _load_key(state, block_size, cfg):
    proj = _proj_kv(state, block_size)
    bucket = int(proj / cfg.kv_bucket) if cfg.kv_bucket > 0 else proj
    return (_saturated(state, block_size, cfg), bucket, _queue(state),
            int(_num(state, "pending_prefill_tokens", 0)))


def pick_rank(request_probe_hits, rank_state, block_size, cfg=None, info=None):
    """Return the DP rank for a new request. See the module docstring."""
    n = len(rank_state)
    if n == 0:
        raise ValueError("rank_state is empty")
    cfg = cfg if cfg is not None else PickerConfig()
    hits = [int(h) for h in list(request_probe_hits)[:n]]
    hits += [0] * (n - len(hits))
    block_size = max(1, int(block_size))

    cursor = _RR[0]
    _RR[0] = cursor + 1
    keys = [_load_key(rank_state[r], block_size, cfg) for r in range(n)]

    def order(r):
        return (keys[r], int(_num(rank_state[r], "routed_total",
                                  0)), (r - cursor) % n)

    load_rank = min(range(n), key=order)
    best_hit = max(hits)
    rank = load_rank
    reason = "load"
    hit_rank = -1
    if best_hit > 0:
        hit_rank = min((r for r in range(n) if hits[r] == best_hit), key=order)
        if hit_rank == load_rank:
            reason = "agree"
        else:
            adv_blocks = (hits[hit_rank] - hits[load_rank]) // block_size
            hs, ls = rank_state[hit_rank], rank_state[load_rank]
            if adv_blocks < cfg.min_affinity_blocks:
                reason = "small_advantage"
            elif (_saturated(hs, block_size, cfg)
                  and not _saturated(ls, block_size, cfg)):
                reason = "hit_rank_saturated"
            elif (float(_num(hs, "kv_usage", 0.0)) -
                  float(_num(ls, "kv_usage", 0.0)) > cfg.max_kv_gap +
                  1e-9):  # epsilon: 0.40 - 0.30 > 0.10 in floats
                reason = "kv_gap"
            elif _queue(hs) - _queue(ls) > cfg.max_queue_gap:
                reason = "queue_gap"
            elif (int(_num(hs, "pending_prefill_tokens", 0)) -
                  int(_num(ls, "pending_prefill_tokens", 0))
                  > cfg.max_pending_gap_tokens):
                reason = "pending_gap"
            else:
                rank = hit_rank
                reason = "affinity"
    if info is not None:
        info.update(reason=reason,
                    rank=rank,
                    hit_rank=hit_rank,
                    load_rank=load_rank,
                    hits=hits)
    return rank
