from __future__ import annotations

import importlib
import importlib.metadata
import warnings
import math
import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn


AttentionBackend = Literal["auto", "fa4", "fa3", "fa2", "legacy"]
ResolvedAttentionBackend = Literal["fa4", "fa3", "fa2", "legacy"]
FA4BiasGradientStrategy = Literal["eager", "deferred"]
FA4KeyBiasLayout = Literal["per_key", "group_scalar", "split_lse"]
FA4SplitLSEMixBackend = Literal["eager", "inductor"]

_FLASH_BACKEND_ORDER: tuple[ResolvedAttentionBackend, ...] = ("fa4", "fa3", "fa2")
_BACKEND_IDS = {"legacy": 0, "fa2": 2, "fa3": 3, "fa4": 4}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _split_heads(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    if value.ndim != 3 or value.shape[-1] % num_heads:
        raise ValueError(
            f"attention tensor must be (B,S,H*D), got {tuple(value.shape)} with H={num_heads}"
        )
    return value.reshape(value.shape[0], value.shape[1], num_heads, -1)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"headed attention tensor must be rank four, got {tuple(value.shape)}")
    return value.reshape(value.shape[0], value.shape[1], value.shape[2] * value.shape[3])


def _padded_bias_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_log_bias: torch.Tensor,
    *,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
        raise ValueError("q/k/v shapes are invalid")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q/k shape mismatch: q={tuple(q.shape)} k={tuple(k.shape)}")
    if key_log_bias.numel() != k.shape[1]:
        raise ValueError("one key_log_bias value is required per key")
    batch, query_count, hidden_dim = q.shape
    key_count = k.shape[1]
    head_dim = hidden_dim // num_heads
    padded_head_dim = ((head_dim + 1 + 7) // 8) * 8
    extra_dim = padded_head_dim - head_dim
    qk_scale = (float(padded_head_dim) / float(head_dim)) ** 0.25
    q_heads = _split_heads(q, num_heads)
    k_heads = _split_heads(k, num_heads)
    v_heads = _split_heads(v, num_heads)
    q_extra = torch.ones((batch, query_count, num_heads, extra_dim), dtype=q.dtype, device=q.device)
    k_extra = torch.zeros((batch, key_count, num_heads, extra_dim), dtype=k.dtype, device=k.device)
    k_extra[..., 0] = key_log_bias.to(dtype=k.dtype, device=k.device).reshape(
        1, key_count, 1
    ).expand(batch, key_count, num_heads) * math.sqrt(float(padded_head_dim))
    v_extra = torch.zeros_like(k_extra, dtype=v.dtype)
    return (
        _merge_heads(torch.cat([q_heads * qk_scale, q_extra], dim=-1)),
        _merge_heads(torch.cat([k_heads * qk_scale, k_extra], dim=-1)),
        _merge_heads(torch.cat([v_heads, v_extra], dim=-1)),
        head_dim,
    )


def attention_with_key_log_bias(
    attention: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_log_bias: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    q_padded, k_padded, v_padded, head_dim = _padded_bias_inputs(
        q, k, v, key_log_bias, num_heads=num_heads
    )
    output = attention(q_padded, k_padded, v_padded)
    batch, query_count, _ = q.shape
    padded_head_dim = output.shape[-1] // num_heads
    return output.reshape(batch, query_count, num_heads, padded_head_dim)[..., :head_dim].reshape(
        batch, query_count, num_heads * head_dim
    )


class _InjectScalarBiasGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        output: torch.Tensor,
        action_bias: torch.Tensor,
        output_derivative: torch.Tensor,
    ) -> torch.Tensor:
        del action_bias
        ctx.save_for_backward(output_derivative)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (output_derivative,) = ctx.saved_tensors
        accumulation_dtype = (
            torch.float64
            if torch.float64 in {grad_output.dtype, output_derivative.dtype}
            else torch.float32
        )
        bias_grad = (
            (grad_output.to(accumulation_dtype) * output_derivative.to(accumulation_dtype))
            .sum()
            .reshape(())
        )
        return grad_output, bias_grad, None


def inject_scalar_bias_gradient(
    output: torch.Tensor,
    action_bias: torch.Tensor,
    output_derivative: torch.Tensor,
) -> torch.Tensor:

    return _InjectScalarBiasGradient.apply(output, action_bias, output_derivative)


class _InjectDeferredScalarBiasGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        output: torch.Tensor,
        action_bias: torch.Tensor,
        query: torch.Tensor,
        action_key: torch.Tensor,
        action_value: torch.Tensor,
        joint_lse: torch.Tensor,
        effective_action_bias: torch.Tensor,
    ) -> torch.Tensor:
        del action_bias
        ctx.save_for_backward(
            output,
            query,
            action_key,
            action_value,
            joint_lse,
            effective_action_bias,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            output,
            query,
            action_key,
            action_value,
            joint_lse,
            effective_action_bias,
        ) = ctx.saved_tensors
        from flash_attn.cute.interface import flash_attn_func

        with torch.no_grad():
            action_output, action_lse = flash_attn_func(
                query.detach(),
                action_key.detach(),
                action_value.detach(),
                causal=False,
                return_lse=True,
            )
            action_mass = torch.exp(
                action_lse.float()
                + effective_action_bias.detach().float()
                - joint_lse.detach().float()
            ).clamp_(0.0, 1.0)
            output_derivative = action_mass.transpose(1, 2).unsqueeze(-1) * (
                action_output.float() - output.detach().float()
            )
            bias_grad = (
                (grad_output.float() * output_derivative).sum(dtype=torch.float32).reshape(())
            )
        return grad_output, bias_grad, None, None, None, None, None


def inject_deferred_scalar_bias_gradient(
    output: torch.Tensor,
    action_bias: torch.Tensor,
    query: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    joint_lse: torch.Tensor,
    effective_action_bias: torch.Tensor,
) -> torch.Tensor:

    return _InjectDeferredScalarBiasGradient.apply(
        output,
        action_bias,
        query,
        action_key,
        action_value,
        joint_lse,
        effective_action_bias,
    )


def _legacy_plain_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    qh = _split_heads(q, num_heads).transpose(1, 2)
    kh = _split_heads(k, num_heads).transpose(1, 2)
    vh = _split_heads(v, num_heads).transpose(1, 2)
    output = torch.nn.functional.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        dropout_p=0.0,
        is_causal=False,
    )
    return _merge_heads(output.transpose(1, 2))


def _fa4_plain_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    from flash_attn.cute.interface import flash_attn_func

    output = flash_attn_func(
        _split_heads(q, num_heads),
        _split_heads(k, num_heads),
        _split_heads(v, num_heads),
        causal=False,
    )
    if isinstance(output, tuple):
        output = output[0]
    return _merge_heads(output)


def _split_lse_mix_eager(
    video_output: torch.Tensor,
    action_output: torch.Tensor,
    video_lse: torch.Tensor,
    action_lse: torch.Tensor,
    effective_bias: torch.Tensor,
) -> torch.Tensor:
    action_mass = (
        torch.sigmoid(action_lse.float() + effective_bias.float() - video_lse.float())
        .transpose(1, 2)
        .unsqueeze(-1)
    )
    output = video_output.float() + action_mass * (action_output.float() - video_output.float())
    return output.to(video_output.dtype)


_INDUCTOR_SPLIT_LSE_MIX = None


def _split_lse_mix(
    video_output: torch.Tensor,
    action_output: torch.Tensor,
    video_lse: torch.Tensor,
    action_lse: torch.Tensor,
    effective_bias: torch.Tensor,
    *,
    backend: FA4SplitLSEMixBackend,
) -> torch.Tensor:
    if backend == "eager":
        return _split_lse_mix_eager(
            video_output,
            action_output,
            video_lse,
            action_lse,
            effective_bias,
        )
    if backend != "inductor":
        raise ValueError(f"unsupported split-LSE mix backend {backend!r}")
    global _INDUCTOR_SPLIT_LSE_MIX
    if _INDUCTOR_SPLIT_LSE_MIX is None:
        _INDUCTOR_SPLIT_LSE_MIX = torch.compile(
            _split_lse_mix_eager,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
    return _INDUCTOR_SPLIT_LSE_MIX(
        video_output,
        action_output,
        video_lse,
        action_lse,
        effective_bias,
    )


def _fa4_split_lse_attention(
    q: torch.Tensor,
    video_k: torch.Tensor,
    video_v: torch.Tensor,
    action_k: torch.Tensor,
    action_v: torch.Tensor,
    action_bias: torch.Tensor,
    *,
    num_heads: int,
    split_lse_mix_backend: FA4SplitLSEMixBackend,
) -> torch.Tensor:

    if video_k.shape != video_v.shape or action_k.shape != action_v.shape:
        raise ValueError("split-LSE K/V shapes must match within each key group")
    if not video_k.shape[1] or not action_k.shape[1]:
        raise ValueError("split-LSE requires non-empty video and action key groups")
    if (
        q.shape[0] != video_k.shape[0]
        or q.shape[0] != action_k.shape[0]
        or q.shape[2] != video_k.shape[2]
        or q.shape[2] != action_k.shape[2]
    ):
        raise ValueError("split-LSE query and grouped K/V batch/hidden shapes must align")

    from flash_attn.cute.interface import flash_attn_func

    qh = _split_heads(q, num_heads)
    video_output, video_lse = flash_attn_func(
        qh,
        _split_heads(video_k, num_heads),
        _split_heads(video_v, num_heads),
        causal=False,
        return_lse=True,
    )
    action_output, action_lse = flash_attn_func(
        qh,
        _split_heads(action_k, num_heads),
        _split_heads(action_v, num_heads),
        causal=False,
        return_lse=True,
    )

    bias_fp32 = action_bias.float()
    rounded_bias = action_bias.to(dtype=q.dtype, device=q.device).float()
    effective_bias = bias_fp32 + (rounded_bias - bias_fp32).detach()
    output = _split_lse_mix(
        video_output,
        action_output,
        video_lse,
        action_lse,
        effective_bias,
        backend=split_lse_mix_backend,
    )
    return _merge_heads(output)


def _fa4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    action_bias: torch.Tensor,
    key_log_bias: torch.Tensor,
    *,
    video_token_count: int,
    num_heads: int,
    bias_gradient_strategy: FA4BiasGradientStrategy,
    key_bias_layout: FA4KeyBiasLayout,
    split_lse_mix_backend: FA4SplitLSEMixBackend,
) -> torch.Tensor:
    if video_token_count != q.shape[1]:
        raise ValueError(
            "FA4 per-key score_mod requires one query per video token: "
            f"queries={q.shape[1]} video_tokens={video_token_count}"
        )
    if not 0 < video_token_count < k.shape[1]:
        raise ValueError("FA4 joint attention requires non-empty video and action keys")
    from flash_attn.cute.interface import flash_attn_func

    from uniaction.model.fa4_score_mod import (
        action_group_bias_score_mod,
        action_group_bias_score_mod_backward,
        per_key_bias_score_mod,
        per_key_bias_score_mod_backward,
    )

    qh = _split_heads(q, num_heads)
    kh = _split_heads(k, num_heads)
    vh = _split_heads(v, num_heads)
    if key_bias_layout == "split_lse":
        return _fa4_split_lse_attention(
            q,
            k[:, :video_token_count],
            v[:, :video_token_count],
            k[:, video_token_count:],
            v[:, video_token_count:],
            action_bias,
            num_heads=num_heads,
            split_lse_mix_backend=split_lse_mix_backend,
        )

    effective_action_bias = action_bias.to(dtype=q.dtype, device=q.device)
    if key_bias_layout == "per_key":
        score_mod = per_key_bias_score_mod
        score_mod_bwd = per_key_bias_score_mod_backward
        aux_tensors = [key_log_bias.detach().contiguous()]
        aux_scalars = None
    elif key_bias_layout == "group_scalar":
        score_mod = action_group_bias_score_mod
        score_mod_bwd = action_group_bias_score_mod_backward
        aux_tensors = [effective_action_bias.detach().reshape(1).contiguous()]
        aux_scalars = None
    else:
        raise ValueError(f"unsupported FA4 key-bias layout {key_bias_layout!r}")
    output, joint_lse = flash_attn_func(
        qh,
        kh,
        vh,
        causal=False,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
        return_lse=True,
    )

    if action_bias.requires_grad and torch.is_grad_enabled():
        if bias_gradient_strategy == "deferred":
            output = inject_deferred_scalar_bias_gradient(
                output,
                action_bias,
                qh,
                kh[:, video_token_count:],
                vh[:, video_token_count:],
                joint_lse,
                effective_action_bias.detach(),
            )
            return _merge_heads(output)
        if bias_gradient_strategy != "eager":
            raise ValueError(f"unsupported FA4 bias-gradient strategy {bias_gradient_strategy!r}")

        with torch.no_grad():
            action_output, action_lse = flash_attn_func(
                qh.detach(),
                kh[:, video_token_count:].detach(),
                vh[:, video_token_count:].detach(),
                causal=False,
                return_lse=True,
            )
            action_mass = torch.exp(
                action_lse.float()
                + effective_action_bias.detach().float()
                - joint_lse.detach().float()
            ).clamp_(0.0, 1.0)
            output_derivative = action_mass.transpose(1, 2).unsqueeze(-1) * (
                action_output.float() - output.detach().float()
            )
        output = inject_scalar_bias_gradient(output, action_bias, output_derivative)

    return _merge_heads(output)


def _fa3_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    from flash_attn_3 import flash_attn_interface

    output = flash_attn_interface.flash_attn_func(
        _split_heads(q, num_heads),
        _split_heads(k, num_heads),
        _split_heads(v, num_heads),
        causal=False,
    )
    return _merge_heads(output)


def _fa2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    from flash_attn import flash_attn_func

    output = flash_attn_func(
        _split_heads(q, num_heads),
        _split_heads(k, num_heads),
        _split_heads(v, num_heads),
        dropout_p=0.0,
        causal=False,
    )
    return _merge_heads(output)


def _backend_attention_with_key_bias(
    backend: Literal["fa3", "fa2"],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_log_bias: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    q_padded, k_padded, v_padded, head_dim = _padded_bias_inputs(
        q, k, v, key_log_bias, num_heads=num_heads
    )
    if backend == "fa3":
        output = _fa3_attention(q_padded, k_padded, v_padded, num_heads=num_heads)
    else:
        output = _fa2_attention(q_padded, k_padded, v_padded, num_heads=num_heads)
    batch, query_count, _ = q.shape
    padded_head_dim = output.shape[-1] // num_heads
    return output.reshape(batch, query_count, num_heads, padded_head_dim)[..., :head_dim].reshape(
        batch, query_count, num_heads * head_dim
    )


@dataclass
class AttentionBackendRuntime:
    requested: AttentionBackend = "auto"
    allow_fallback: bool = True
    require_key_bias: bool = True
    selected: ResolvedAttentionBackend | None = None
    attempts: list[dict[str, str]] = field(default_factory=list)
    calls: int = 0
    plain_calls: int = 0
    key_bias_calls: int = 0
    _probing: bool = False
    fa4_bias_gradient: FA4BiasGradientStrategy = "eager"
    fa4_key_bias_layout: FA4KeyBiasLayout = "per_key"
    fa4_split_lse_mix_backend: FA4SplitLSEMixBackend = "eager"

    reference_count_learnable: bool | None = None
    _dynamo_boundary_installed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.requested not in {"auto", "fa4", "fa3", "fa2", "legacy"}:
            raise ValueError(f"unsupported attention backend {self.requested!r}")
        if self.requested == "legacy" and self.allow_fallback:
            self.allow_fallback = False
        if self.fa4_bias_gradient not in {"eager", "deferred"}:
            raise ValueError(f"unsupported FA4 bias-gradient strategy {self.fa4_bias_gradient!r}")
        if self.fa4_key_bias_layout not in {"per_key", "group_scalar", "split_lse"}:
            raise ValueError(f"unsupported FA4 key-bias layout {self.fa4_key_bias_layout!r}")
        if self.fa4_split_lse_mix_backend not in {"eager", "inductor"}:
            raise ValueError(
                f"unsupported FA4 split-LSE mix backend {self.fa4_split_lse_mix_backend!r}"
            )
        if self.fa4_split_lse_mix_backend != "eager" and self.fa4_key_bias_layout != "split_lse":
            raise ValueError(
                "the Inductor split-LSE mixer requires fa4_key_bias_layout='split_lse'"
            )

    def _candidates(self) -> tuple[ResolvedAttentionBackend, ...]:
        if self.requested == "legacy":
            return ("legacy",)
        if self.requested == "auto":
            candidates = _FLASH_BACKEND_ORDER
        else:
            start = _FLASH_BACKEND_ORDER.index(self.requested)
            candidates = _FLASH_BACKEND_ORDER[start:]
            if not self.allow_fallback:
                candidates = candidates[:1]
        if self.allow_fallback:
            return (*candidates, "legacy")
        return candidates

    def install_dynamo_boundary(self) -> None:

        if self._dynamo_boundary_installed:
            return
        self.plain_attention = torch.compiler.disable(self.plain_attention)
        self.attention = torch.compiler.disable(self.attention)
        self.split_lse_attention = torch.compiler.disable(self.split_lse_attention)
        self._dynamo_boundary_installed = True

    @staticmethod
    def _check_candidate_import(backend: ResolvedAttentionBackend) -> None:
        module = {
            "fa4": "flash_attn.cute",
            "fa3": "flash_attn_3.flash_attn_interface",
            "fa2": "flash_attn",
        }[backend]
        importlib.import_module(module)

    @staticmethod
    def _check_hardware(backend: ResolvedAttentionBackend, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError(f"{backend} requires CUDA, got {device}")
        if backend == "fa4":
            name = " ".join(torch.cuda.get_device_name(device).upper().split())
            if not any(model in name for model in ("H100", "H200", "RTX PRO 6000")):
                raise RuntimeError(
                    f"FA4 is not supported on {name}; supported GPUs: H100, H200, RTX PRO 6000"
                )
        major, minor = torch.cuda.get_device_capability(device)
        required_major = 9 if backend in {"fa4", "fa3"} else 8
        if major < required_major:
            raise RuntimeError(f"{backend} requires SM{required_major}0+, got SM{major}{minor}")

    def _probe(self, backend: ResolvedAttentionBackend, device: torch.device) -> None:
        self._check_hardware(backend, device)
        self._check_candidate_import(backend)
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(1234)
            q = torch.randn(1, 64, 4 * 128, device=device, dtype=torch.bfloat16).requires_grad_()
            k = torch.randn(1, 80, 4 * 128, device=device, dtype=torch.bfloat16).requires_grad_()
            v = torch.randn(1, 80, 4 * 128, device=device, dtype=torch.bfloat16).requires_grad_()
            bias = torch.tensor(1.25, device=device, dtype=torch.float32, requires_grad=True)
            key_bias = torch.cat(
                [
                    torch.zeros(64, device=device, dtype=torch.bfloat16),
                    torch.ones(16, device=device, dtype=torch.bfloat16) * bias,
                ]
            )
            self.selected = backend
            self._probing = True
            try:
                plain_output = self.plain_attention(
                    q,
                    k,
                    v,
                    num_heads=4,
                )
                outputs = [plain_output]
                gradient_inputs: tuple[torch.Tensor, ...] = (q, k, v)
                if self.require_key_bias:
                    outputs.append(
                        self.attention(
                            None,
                            q,
                            k,
                            v,
                            action_bias=bias,
                            key_log_bias=key_bias,
                            video_token_count=64,
                            num_heads=4,
                        )
                    )
                    gradient_inputs = (*gradient_inputs, bias)
                loss = sum(output.float().square().mean() for output in outputs)
                gradients = torch.autograd.grad(loss, gradient_inputs)
                if not all(bool(torch.isfinite(value).all()) for value in (*outputs, *gradients)):
                    raise FloatingPointError(f"{backend} probe produced non-finite values")
                if self.require_key_bias and float(gradients[-1].abs()) == 0.0:
                    raise RuntimeError(f"{backend} probe lost the learnable action-bias gradient")
                torch.cuda.synchronize(device)
            finally:
                self._probing = False

    def resolve(self, device: torch.device | str) -> ResolvedAttentionBackend:
        if self.selected is not None:
            return self.selected
        device = torch.device(device)
        if self.requested == "legacy":
            self.selected = "legacy"
            return self.selected
        errors: list[str] = []
        for backend in self._candidates():
            self.selected = None

            if backend == "legacy":
                self.attempts.append({"backend": backend, "status": "selected", "error": ""})
                self.selected = backend
                self.calls = 0
                self.plain_calls = 0
                self.key_bias_calls = 0
                self._report_fallback(backend)
                return backend
            try:
                self._probe(backend, device)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self.attempts.append({"backend": backend, "status": "failed", "error": message})
                errors.append(f"{backend}={message}")
                self.selected = None
                torch.cuda.empty_cache()
                if not self.allow_fallback:
                    break
                if int(os.environ.get("RANK", "0")) == 0:
                    warnings.warn(
                        f"{backend.upper()} unavailable: {error}. Trying the next attention backend.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            self.attempts.append({"backend": backend, "status": "selected", "error": ""})
            self.selected = backend
            self.calls = 0
            self.plain_calls = 0
            self.key_bias_calls = 0
            torch.cuda.empty_cache()
            self._report_fallback(backend)
            return backend
        raise RuntimeError("no requested attention backend passed its probe: " + "; ".join(errors))

    def _report_fallback(self, backend):
        if (
            any(attempt["status"] == "failed" for attempt in self.attempts)
            and int(os.environ.get("RANK", "0")) == 0
        ):
            label = "SDPA" if backend == "legacy" else backend.upper()
            warnings.warn(f"Attention fallback selected: {label}.", RuntimeWarning, stacklevel=3)

    def attention(
        self,
        legacy_attention: nn.Module | None,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        action_bias: torch.Tensor,
        key_log_bias: torch.Tensor,
        video_token_count: int,
        num_heads: int,
    ) -> torch.Tensor:
        if self.selected is None:
            raise RuntimeError("attention backend must be resolved before the first model forward")
        if not self._probing:
            self.calls += 1
            self.key_bias_calls += 1
        if self.selected == "fa4":
            return _fa4_attention(
                q,
                k,
                v,
                action_bias,
                key_log_bias,
                video_token_count=video_token_count,
                num_heads=num_heads,
                bias_gradient_strategy=self.fa4_bias_gradient,
                key_bias_layout=self.fa4_key_bias_layout,
                split_lse_mix_backend=self.fa4_split_lse_mix_backend,
            )
        if self.selected in {"fa3", "fa2"}:
            return _backend_attention_with_key_bias(
                self.selected,
                q,
                k,
                v,
                key_log_bias,
                num_heads=num_heads,
            )
        if legacy_attention is None:
            raise RuntimeError("legacy attention module is required for the legacy backend")
        return attention_with_key_log_bias(
            legacy_attention, q, k, v, key_log_bias, num_heads=num_heads
        )

    def split_lse_attention(
        self,
        q: torch.Tensor,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        action_k: torch.Tensor,
        action_v: torch.Tensor,
        *,
        action_bias: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:

        if self.selected != "fa4" or self.fa4_key_bias_layout != "split_lse":
            raise RuntimeError(
                "direct split-LSE attention requires selected FA4 with split_lse layout"
            )
        if not self._probing:
            self.calls += 1
            self.key_bias_calls += 1
        return _fa4_split_lse_attention(
            q,
            video_k,
            video_v,
            action_k,
            action_v,
            action_bias,
            num_heads=num_heads,
            split_lse_mix_backend=self.fa4_split_lse_mix_backend,
        )

    def plain_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        num_heads: int,
    ) -> torch.Tensor:
        if self.selected is None:
            raise RuntimeError("attention backend must be resolved before the first model forward")
        if not self._probing:
            self.calls += 1
            self.plain_calls += 1
        if self.selected == "fa4":
            return _fa4_plain_attention(q, k, v, num_heads=num_heads)
        if self.selected == "fa3":
            return _fa3_attention(q, k, v, num_heads=num_heads)
        if self.selected == "fa2":
            return _fa2_attention(q, k, v, num_heads=num_heads)
        return _legacy_plain_attention(q, k, v, num_heads=num_heads)

    def summary(self) -> dict[str, object]:
        split_lse_algorithm = (
            "split_plain_fa4_lse_mix_d128_learnable_scalar_gradient"
            f"_{self.fa4_split_lse_mix_backend}_mix"
        )
        fa4_biased_algorithm = {
            "per_key": "cutedsl_per_key_bias_d128_exact_learnable_scalar_gradient",
            "group_scalar": "cutedsl_action_group_scalar_bias_d128_exact_learnable_scalar_gradient",
            "split_lse": split_lse_algorithm,
        }[self.fa4_key_bias_layout]
        biased_algorithm = {
            "fa4": fa4_biased_algorithm,
            "fa3": "augmented_qk_bias_d136",
            "fa2": "augmented_qk_bias_d136",
            "legacy": "pytorch_sdpa_augmented_qk_bias_d136",
            None: "unresolved",
        }[self.selected]
        plain_algorithm = {
            "fa4": "native_fa4_d128",
            "fa3": "native_fa3_d128",
            "fa2": "native_fa2_d128",
            "legacy": "pytorch_sdpa_d128",
            None: "unresolved",
        }[self.selected]
        algorithm = (
            f"{plain_algorithm}+{biased_algorithm}" if self.require_key_bias else plain_algorithm
        )
        return {
            "summary_schema_version": 2,
            "requested": self.requested,
            "fallback_allowed": self.allow_fallback,
            "selected": self.selected,
            "selected_id": None if self.selected is None else _BACKEND_IDS[self.selected],
            "algorithm": algorithm,
            "coverage": (
                "wan_base_self_cross_and_action_joint"
                if self.require_key_bias
                else "wan_base_self_cross"
            ),
            "probe_requires_key_bias": self.require_key_bias,
            "learnable_reference_count": self.reference_count_learnable,
            "reference_count_learnable": self.reference_count_learnable,
            "action_key_bias_required": self.require_key_bias,
            "reference_count_gradient_supported": self.require_key_bias,
            "fa4_bias_gradient": self.fa4_bias_gradient,
            "fa4_key_bias_layout": self.fa4_key_bias_layout,
            "fa4_split_lse_mix_backend": self.fa4_split_lse_mix_backend,
            "calls": self.calls,
            "plain_calls": self.plain_calls,
            "key_bias_calls": self.key_bias_calls,
            "attempts": list(self.attempts),
            "rank": int(os.environ.get("RANK", "0")),
            "packages": {
                "torch": torch.__version__,
                "flash_attn_4": _distribution_version("flash-attn-4"),
                "flash_attn_3": _distribution_version("flash-attn-3"),
                "flash_attn_2": _distribution_version("flash-attn"),
            },
        }


class RuntimeAttentionModule(nn.Module):
    def __init__(self, runtime: AttentionBackendRuntime, num_heads: int) -> None:
        super().__init__()
        self.runtime = runtime
        self.num_heads = int(num_heads)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return self.runtime.plain_attention(q, k, v, num_heads=self.num_heads)
