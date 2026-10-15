# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""CPU helpers: a REAL ``DPScheduler`` over REAL vLLM schedulers, no TPU.

Used by the DP rank-hint / metrics / KV-event / picker tests. Everything on
the scheduling side is the production code: ``DPScheduler``, forked rank
worker processes, vLLM's ``Scheduler``/``AsyncScheduler`` with real KV block
pools and prefix-cache hashing, real ``Request`` objects. Only the model
forward pass is absent, which does not matter for routing and bookkeeping.

The model is a 2-layer Llama *config* written to a temp dir (no weights, no
tokenizer, no network).
"""
import contextlib
import json
import os
from types import SimpleNamespace
from unittest import mock

import torch
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

BLOCK = 16

TINY_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16",
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def write_tiny_llama(path) -> str:
    """Write a config-only Llama model dir; returns its path."""
    path = str(path)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(TINY_LLAMA, f)
    return path


@contextlib.contextmanager
def _platform_hook_off():
    """Keep VllmConfig() independent of the platform that happens to be active
    (CPU test venv: none; TPU host: TpuPlatform adjusts block sizes etc.)."""
    from vllm.platforms import current_platform
    with mock.patch.object(current_platform,
                           "check_and_update_config",
                           lambda *a, **k: None):
        yield


def make_vllm_config(model_dir,
                     *,
                     parallel_dp: int = 1,
                     max_num_seqs: int = 64,
                     max_num_batched_tokens: int = 4096,
                     kv_events_config=None,
                     enable_prefix_caching: bool = True) -> VllmConfig:
    model_config = ModelConfig(model=model_dir,
                               tokenizer=model_dir,
                               skip_tokenizer_init=True,
                               dtype="bfloat16",
                               max_model_len=4096)
    cache_config = CacheConfig(block_size=BLOCK,
                               gpu_memory_utilization=0.5,
                               enable_prefix_caching=enable_prefix_caching)
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=4096,
        is_encoder_decoder=False)
    kwargs = {}
    if kv_events_config is not None:
        kwargs["kv_events_config"] = kv_events_config
    with _platform_hook_off():
        return VllmConfig(model_config=model_config,
                          cache_config=cache_config,
                          scheduler_config=scheduler_config,
                          parallel_config=ParallelConfig(
                              data_parallel_size=parallel_dp),
                          device_config=DeviceConfig(device="cpu"),
                          **kwargs)


def apply_tpu_sharding(vllm_config: VllmConfig):
    """What ``TpuPlatform.check_and_update_config`` does for DP: the real
    ``ShardingConfigManager`` (which pins parallel_config.data_parallel_size
    to 1 under SPMD DP) and the real DPScheduler selection hook."""
    from tpu_inference.core.sched.dp_scheduler import \
        update_vllm_config_for_dp_scheduler
    from tpu_inference.layers.common.sharding import ShardingConfigManager
    vllm_config.sharding_config = ShardingConfigManager.from_vllm_config(
        vllm_config)
    update_vllm_config_for_dp_scheduler(vllm_config)
    return vllm_config


def make_real_dp_scheduler(model_dir,
                           dp: int,
                           *,
                           pool_blocks_per_rank: int = 400,
                           max_num_seqs: int = 64,
                           kv_events_config=None,
                           log_stats: bool = True,
                           scheduler_cls=AsyncScheduler,
                           vllm_config=None):
    """Build a real DPScheduler with ``dp`` forked rank workers.

    Returns ``(vllm_config, scheduler)``. The caller must ``shutdown()`` it.
    """
    from tpu_inference.core.sched import dp_scheduler as dps
    pool_blocks = pool_blocks_per_rank * dp
    vc = vllm_config or make_vllm_config(model_dir,
                                         max_num_seqs=max_num_seqs,
                                         kv_events_config=kv_events_config)
    vc.cache_config.num_gpu_blocks = pool_blocks // dp
    if not hasattr(vc, "sharding_config"):
        vc.sharding_config = SimpleNamespace(total_dp_size=dp)
    vc.scheduler_config._original_scheduler_cls = scheduler_cls
    spec = FullAttentionSpec(block_size=BLOCK,
                             num_kv_heads=2,
                             head_size=16,
                             dtype=torch.bfloat16)
    kv_cache_config = KVCacheConfig(
        num_blocks=pool_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)
        ])
    scheduler = dps.DPScheduler(
        vllm_config=vc,
        kv_cache_config=kv_cache_config,
        structured_output_manager=SimpleNamespace(
            should_advance=lambda request: False,
            accept_tokens=lambda request, tokens: True,
            validate_tokens=lambda request, tokens: tokens),
        block_size=BLOCK,
        log_stats=log_stats)
    return vc, scheduler


_HASHER = None


def _block_hasher():
    global _HASHER
    if _HASHER is None:
        hash_fn = get_hash_fn_by_name("sha256")
        init_none_hash(hash_fn)
        _HASHER = get_request_block_hasher(BLOCK, hash_fn)
    return _HASHER


def make_request(request_id: str,
                 *,
                 prefix_len: int = 96,
                 body_len: int = 200,
                 seed: int = 0,
                 max_tokens: int = 64,
                 prefix_token: int = 7,
                 extra_args=None) -> Request:
    """A request whose first ``prefix_len`` tokens are shared (same
    ``prefix_token``) and whose body is unique per ``seed``."""
    prompt = ([prefix_token] * prefix_len) + [
        1000 + (seed * 13 + j) % 900 for j in range(body_len)
    ]
    return Request(request_id=request_id,
                   prompt_token_ids=prompt,
                   sampling_params=SamplingParams(max_tokens=max_tokens,
                                                  ignore_eos=True,
                                                  extra_args=extra_args),
                   pooling_params=None,
                   block_hasher=_block_hasher())


def step(scheduler):
    """One engine step: schedule on every rank, fake a 1-token sample for each
    scheduled request, update_from_output. Returns DPScheduler's
    ``{client_index: EngineCoreOutputs}``; ``last_stats(out)`` extracts the
    combined SchedulerStats."""
    out = scheduler.schedule()
    ids = list(out.num_scheduled_tokens)
    runner_output = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i
                         for i, r in enumerate(ids)},
        sampled_token_ids=[[5] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[None] * len(ids))
    return scheduler.update_from_output(out, runner_output)


def last_stats(outputs):
    stats = None
    for engine_outputs in outputs.values():
        if engine_outputs.scheduler_stats is not None:
            stats = engine_outputs.scheduler_stats
    return stats


def routed_per_rank(scheduler) -> dict:
    counts: dict = {}
    for rank in scheduler.assigned_dp_rank.values():
        counts[rank] = counts.get(rank, 0) + 1
    return dict(sorted(counts.items()))
