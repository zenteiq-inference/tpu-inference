# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Per-request DP-rank hints for the SPMD ``DPScheduler``.

Why this exists
---------------
vLLM lets a client pin a request to one data-parallel rank with the
``X-data-parallel-rank`` HTTP header. The header becomes
``EngineCoreRequest.data_parallel_rank`` and is range-checked against
``parallel_config.data_parallel_size`` in ``InputProcessor.process_inputs``.

With tpu-inference's SPMD DP the DP ranks live *inside* one engine, driven by
``DPScheduler``. ``ShardingConfigManager.from_vllm_config`` therefore forces
``parallel_config.data_parallel_size = 1`` in every process that builds a
config (API servers included), so vLLM's own check accepts only rank 0 and
answers HTTP 400 for rank >= 1. Rank 0 is accepted but then dropped:
``Request.from_engine_core_request`` does not copy the field and
``DPScheduler`` had no hint to read.

What this module does (tpu-inference side only, no vLLM change)
---------------------------------------------------------------
* ``install_input_processor_hook`` wraps ``InputProcessor.process_inputs``.
  When a ``DPScheduler`` drives the config it validates the rank against the
  real DP size (``sharding_config.total_dp_size``), passes ``None`` to vLLM's
  own check, and writes the rank into ``SamplingParams.extra_args["dp_rank"]``
  of the returned ``EngineCoreRequest``. ``extra_args`` is the one per-request
  field that already survives the trip to ``Scheduler.add_request`` unchanged
  (it is the channel behind the OpenAI ``vllm_xargs`` body field), so no
  ``Request`` patching is needed in the EngineCore process.
* ``get_dp_rank_hint`` is what ``DPScheduler`` calls to read the hint back.

A hint can also be sent directly as ``"vllm_xargs": {"dp_rank": 1}`` in the
request body. That path skips the API-level range check, which is why the
scheduler re-validates and falls back to its normal picker when the hint is
missing, malformed or out of range.

Wrapper activation requires ``total_dp_size > 1`` *and* a ``DPScheduler`` as
the configured scheduler. With vLLM-native DP (``TPU_MULTIPROCESS_DP=1``) the
header is validated and honoured by vLLM itself and the wrapper steps aside.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Optional

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Key under SamplingParams.extra_args (and PoolingParams.extra_kwargs).
DP_RANK_HINT_KEY = "dp_rank"

_HOOK_MARKER = "_tpu_inference_dp_rank_hint_hook"
_warned: set = set()


def _warn_once(key: Any, msg: str, *args: Any) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def dp_scheduler_rank_count(vllm_config: Any) -> int:
    """Number of ``DPScheduler`` ranks serving ``vllm_config``, else 1.

    Returns 1 (hint handling off) unless the platform selected the SPMD
    ``DPScheduler``: with a plain vLLM scheduler, or with vLLM-native DP,
    vLLM's own rank handling is the right one.
    """
    sharding = getattr(vllm_config, "sharding_config", None)
    size = getattr(sharding, "total_dp_size", 1)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 1:
        return 1
    scheduler_cls = getattr(getattr(vllm_config, "scheduler_config", None),
                            "scheduler_cls", None)
    if isinstance(scheduler_cls, str):
        return size if scheduler_cls.endswith("DPScheduler") else 1
    from tpu_inference.core.sched.dp_scheduler import DPScheduler
    if isinstance(scheduler_cls, type) and issubclass(scheduler_cls,
                                                      DPScheduler):
        return size
    return 1


def _as_rank(value: Any) -> Optional[int]:
    """Strict int (JSON numbers arrive as int); bool and float are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def get_dp_rank_hint(request: Any, dp_size: int) -> Optional[int]:
    """Return the DP rank ``request`` asks for, or None.

    None means "no usable hint": nothing was requested, or the value is not an
    int in ``[0, dp_size)``. A bad value is logged once and ignored so a
    malformed ``vllm_xargs`` entry can never fail a request in the scheduler.
    """
    params = getattr(request, "sampling_params", None)
    extras = getattr(params, "extra_args", None)
    if not isinstance(extras, dict):
        params = getattr(request, "pooling_params", None)
        extras = getattr(params, "extra_kwargs", None)
    if not isinstance(extras, dict) or DP_RANK_HINT_KEY not in extras:
        return None
    raw = extras[DP_RANK_HINT_KEY]
    if raw is None:
        return None
    rank = _as_rank(raw)
    if rank is None:
        _warn_once(("type", type(raw).__name__),
                   "Ignoring %s=%r in request extra args: not an integer.",
                   DP_RANK_HINT_KEY, raw)
        return None
    if not 0 <= rank < dp_size:
        _warn_once(("range", rank, dp_size),
                   "Ignoring %s=%d in request extra args: out of range "
                   "[0, %d).", DP_RANK_HINT_KEY, rank, dp_size)
        return None
    return rank


def _attach_hint(engine_core_request: Any, rank: int) -> None:
    """Store ``rank`` where the scheduler will find it."""
    params = getattr(engine_core_request, "sampling_params", None)
    if params is not None and hasattr(params, "extra_args"):
        extras = dict(params.extra_args or {})
        extras[DP_RANK_HINT_KEY] = rank
        params.extra_args = extras
        return
    params = getattr(engine_core_request, "pooling_params", None)
    if params is not None and hasattr(params, "extra_kwargs"):
        extras = dict(params.extra_kwargs or {})
        extras[DP_RANK_HINT_KEY] = rank
        params.extra_kwargs = extras
        return
    _warn_once(
        "no-params",
        "X-data-parallel-rank was validated but the request has no "
        "params object to carry it; routing it with the normal picker.")


def _range_error(rank: int, num_ranks: int) -> Exception:
    message = f"data_parallel_rank {rank} is out of range [0, {num_ranks})."
    try:  # newer vLLM maps this type to HTTP 400 with the offending parameter
        from vllm.exceptions import VLLMValidationError
    except ImportError:
        return ValueError(message)
    return VLLMValidationError(message,
                               parameter="data_parallel_rank",
                               value=rank)


def install_input_processor_hook() -> bool:
    """Wrap ``InputProcessor.process_inputs`` (idempotent).

    Must run before an ``InputProcessor`` is constructed in the process:
    recent vLLM binds ``process_inputs_async`` to the bound method in
    ``InputProcessor.__init__``. ``update_vllm_config_for_dp_scheduler`` calls
    this from ``TpuPlatform.check_and_update_config``, which runs while the
    ``VllmConfig`` is built, i.e. before any ``InputProcessor`` exists.

    Returns True when the hook is active, False when it could not be installed
    (vLLM layout differs); in that case the header behaves as before.
    """
    try:
        from vllm.v1.engine.input_processor import InputProcessor
    except Exception as e:  # ImportError, or a heavy transitive import failing
        _warn_once(
            "import", "DP rank hint hook not installed: cannot import vLLM "
            "InputProcessor (%s).", e)
        return False

    original = InputProcessor.process_inputs
    if getattr(original, _HOOK_MARKER, False):
        return True
    signature = inspect.signature(original)
    if "data_parallel_rank" not in signature.parameters:
        _warn_once(
            "signature", "DP rank hint hook not installed: InputProcessor."
            "process_inputs has no data_parallel_rank parameter.")
        return False

    @functools.wraps(original)
    def process_inputs(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError:
            # Not our error to report; vLLM raises it with its own message.
            return original(self, *args, **kwargs)
        rank = bound.arguments.get("data_parallel_rank")
        if rank is None:
            return original(self, *args, **kwargs)
        num_ranks = dp_scheduler_rank_count(self.vllm_config)
        if num_ranks <= 1:
            return original(self, *args, **kwargs)
        if _as_rank(rank) is None or not 0 <= rank < num_ranks:
            raise _range_error(rank, num_ranks)
        # vLLM's own check compares against parallel_config.data_parallel_size,
        # which the SPMD override pinned to 1. Hand it None and carry the
        # validated rank in the request instead.
        bound.arguments["data_parallel_rank"] = None
        engine_core_request = original(*bound.args, **bound.kwargs)
        _attach_hint(engine_core_request, rank)
        return engine_core_request

    setattr(process_inputs, _HOOK_MARKER, True)
    process_inputs.__wrapped_original__ = original  # for tests / uninstall
    InputProcessor.process_inputs = process_inputs
    logger.info(
        "Installed DP rank hint hook: X-data-parallel-rank is honoured "
        "by the SPMD DPScheduler.")
    return True


def uninstall_input_processor_hook() -> None:
    """Undo ``install_input_processor_hook`` (used by tests)."""
    from vllm.v1.engine.input_processor import InputProcessor
    current = InputProcessor.process_inputs
    original = getattr(current, "__wrapped_original__", None)
    if original is not None:
        InputProcessor.process_inputs = original
