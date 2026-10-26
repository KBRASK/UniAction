from __future__ import annotations

import math
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniaction.backbones.wan.dit import modulate, rope_apply, sinusoidal_embedding_1d
from uniaction.model.attention import (
    AttentionBackendRuntime,
)

if TYPE_CHECKING:
    from uniaction.config import ExperimentConfig


@dataclass(frozen=True)
class UnifiedActionConfig:
    l_max: int = 32
    num_frames: int = 81
    dit_dim: int = 5120
    num_dit_blocks: int = 40
    group_size: int = 4
    video_height: int = 480
    video_width: int = 832
    hand_vocab_size: int = 16
    camera_vocab_size: int = 16
    robot_vocab_size: int = 12
    force_start_token_id: int = 0
    force_endpoint_token_id: int = 3
    action_reference_count: float = 784.0

    torch_dtype: Any = torch.bfloat16

    def __post_init__(self) -> None:
        if self.l_max != 2 * self.hand_vocab_size:
            raise ValueError(
                f"l_max must encode {self.hand_vocab_size} hand UV pairs, got {self.l_max}"
            )
        if self.action_reference_count <= 0:
            raise ValueError("the model requires a positive N_ref")
        if (self.force_start_token_id, self.force_endpoint_token_id) != (0, 3):
            raise ValueError("the force vocabulary uses start=0 and endpoint=3")


class FP32AffineRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        if self.weight.dtype != torch.float32:
            self.weight.data = self.weight.data.float()
        if self.weight.grad is not None and self.weight.grad.dtype != torch.float32:
            self.weight.grad.data = self.weight.grad.data.float()
        return self

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        value_fp32 = value.float()
        normalized = value_fp32 * torch.rsqrt(
            value_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight.to(dtype)


class TaskUnifiedTokenBank(nn.Module):
    def __init__(self, cfg: UnifiedActionConfig) -> None:
        super().__init__()
        self.out_dim = cfg.dit_dim
        self.num_frames = cfg.num_frames
        self.l_max = cfg.l_max
        self.num_keypoints = cfg.hand_vocab_size
        self.force_start_token_id = cfg.force_start_token_id
        self.force_endpoint_token_id = cfg.force_endpoint_token_id
        self.task_unified_camera_vocab_size = cfg.camera_vocab_size
        self.task_unified_robot_vocab_size = cfg.robot_vocab_size
        scale = 1.0 / math.sqrt(float(cfg.dit_dim))

        self.force_direct_token = nn.Parameter(
            torch.randn(1, cfg.dit_dim, dtype=cfg.torch_dtype) * scale
        )
        self.force_endpoint_token = nn.Parameter(
            torch.randn(1, cfg.dit_dim, dtype=cfg.torch_dtype) * scale
        )
        self.hand_token_emb = nn.Parameter(
            torch.randn(1, cfg.hand_vocab_size, cfg.dit_dim, dtype=cfg.torch_dtype) * scale
        )
        self.camera_token_emb = nn.Parameter(
            torch.randn(1, cfg.camera_vocab_size, cfg.dit_dim, dtype=cfg.torch_dtype) * scale
        )
        self.robot_token_emb = nn.Parameter(
            torch.randn(1, cfg.robot_vocab_size, cfg.dit_dim, dtype=cfg.torch_dtype) * scale
        )

        self.hand_token_norm = FP32AffineRMSNorm(cfg.dit_dim)
        self.force_token_norm = FP32AffineRMSNorm(cfg.dit_dim)
        self.camera_token_norm = FP32AffineRMSNorm(cfg.dit_dim)
        self.robot_token_norm = FP32AffineRMSNorm(cfg.dit_dim)

    def _touch_all_parameters(self, tokens: torch.Tensor) -> torch.Tensor:

        zero = tokens.new_zeros(())
        for parameter in self.parameters():
            zero = zero + parameter.reshape(-1)[0].to(tokens.dtype) * 0.0
        return tokens + zero

    def forward_dense(
        self,
        action: torch.Tensor,
        action_time_mask: torch.Tensor | None,
        action_dim_mask: torch.Tensor | None,
        action_point_time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action.ndim != 3 or action.shape[1:] != (self.num_frames, self.l_max):
            raise ValueError(
                f"dense hand action must be (B,{self.num_frames},{self.l_max}), "
                f"got {tuple(action.shape)}"
            )
        batch, frames, _ = action.shape
        keypoints = self.num_keypoints
        uv = action.float().reshape(batch, frames, keypoints, 2).clamp(0.0, 1.0)
        tokens = self.hand_token_emb.to(action.device).unsqueeze(1)
        tokens = tokens.expand(batch, frames, keypoints, self.out_dim)
        tokens = self.hand_token_norm(tokens.reshape(batch, frames * keypoints, self.out_dim))
        tokens = self._touch_all_parameters(tokens)

        if action_time_mask is None:
            time_valid = torch.ones(batch, frames, dtype=torch.bool, device=action.device)
        else:
            time_valid = action_time_mask.to(device=action.device, dtype=torch.bool)
        if action_dim_mask is None:
            point_valid = torch.ones(batch, keypoints, dtype=torch.bool, device=action.device)
        else:
            point_valid = (
                action_dim_mask.to(device=action.device, dtype=torch.bool)
                .reshape(batch, keypoints, 2)
                .all(dim=-1)
            )
        valid = (time_valid.unsqueeze(-1) & point_valid.unsqueeze(1)).reshape(
            batch, frames * keypoints
        )
        if action_point_time_mask is not None:
            point_time_valid = action_point_time_mask.to(device=action.device, dtype=torch.bool)
            if tuple(point_time_valid.shape) != (batch, frames, keypoints):
                raise ValueError(
                    "action_point_time_mask must be "
                    f"{(batch, frames, keypoints)}, got {tuple(point_time_valid.shape)}"
                )
            if batch != 1 and not bool(point_time_valid.all()):
                raise ValueError(
                    "ragged dense point-time masking requires per-device batch size one"
                )
            valid = valid & point_time_valid.reshape(batch, frames * keypoints)
        tokens = tokens * valid.to(tokens.dtype).unsqueeze(-1)
        time_ids = (
            torch.arange(frames, device=action.device, dtype=torch.long)
            .view(1, frames, 1)
            .expand(batch, frames, keypoints)
            .reshape(batch, frames * keypoints)
        )
        uv = uv.reshape(batch, frames * keypoints, 2)

        if batch == 1:
            keep = valid[0]
            if not bool(keep.any()):
                raise ValueError("dense hand condition has no visible point-time tokens")
            tokens = tokens[:, keep]
            uv = uv[:, keep]
            valid = valid[:, keep]
            time_ids = time_ids[:, keep]
        return tokens, uv, valid, time_ids

    def _force_tokens(
        self,
        ids: torch.Tensor,
        time_ids: torch.Tensor,
        *,
        batch: int,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:

        if count != 2:
            raise ValueError(f"force requires exactly two tokens, got {count}")
        start = ids == self.force_start_token_id
        endpoint = ids == self.force_endpoint_token_id
        if not bool(((start.sum(dim=1) == 1) & (endpoint.sum(dim=1) == 1)).all()):
            raise ValueError(
                "force requires one start token (id=0) and one endpoint token "
                f"(id=3), got {ids.detach().cpu().tolist()}"
            )
        if not bool((time_ids.float() == 0).all()):
            raise ValueError("force start and endpoint RoPE must both use time zero")
        start_token = self.force_direct_token.to(device).view(1, 1, self.out_dim)
        endpoint_token = self.force_endpoint_token.to(device).view(1, 1, self.out_dim)
        tokens = torch.where(
            start.unsqueeze(-1),
            start_token.expand(batch, count, -1),
            endpoint_token.expand(batch, count, -1),
        )
        return self.force_token_norm(tokens)

    def forward_sparse(
        self,
        uv: torch.Tensor,
        time_ids: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        family: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if uv.ndim == 2:
            uv = uv.unsqueeze(0)
        if time_ids.ndim == 1:
            time_ids = time_ids.unsqueeze(0)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        batch, count, coordinate_dim = uv.shape
        if (
            coordinate_dim != 2
            or time_ids.shape != (batch, count)
            or token_ids.shape
            != (
                batch,
                count,
            )
        ):
            raise ValueError(
                "sparse action shapes must align: "
                f"uv={tuple(uv.shape)}, time={tuple(time_ids.shape)}, "
                f"ids={tuple(token_ids.shape)}"
            )
        device = uv.device
        ids = token_ids.to(device=device, dtype=torch.long)
        family = family.lower()

        if family == "force":
            tokens = self._force_tokens(ids, time_ids, batch=batch, count=count, device=device)
        else:
            families = {
                "camera": (
                    self.camera_token_emb,
                    self.camera_token_norm,
                    self.task_unified_camera_vocab_size,
                ),
                "robot": (
                    self.robot_token_emb,
                    self.robot_token_norm,
                    self.task_unified_robot_vocab_size,
                ),
            }
            if family not in families:
                raise ValueError(f"unsupported sparse token family {family!r}")
            table, norm, vocab_size = families[family]
            invalid = (ids < 0) | (ids >= vocab_size)
            if bool(invalid.any()):
                bad = torch.unique(ids[invalid]).detach().cpu().tolist()
                raise ValueError(f"{family} ids must be in [0,{vocab_size}), got {bad}")
            tokens = norm(table.to(device)[0][ids])

        tokens = self._touch_all_parameters(tokens)
        valid = torch.ones((batch, count), dtype=torch.bool, device=device)
        return (
            tokens,
            uv.to(device=device, dtype=torch.float32).clamp(0.0, 1.0),
            valid,
            time_ids.to(device=device, dtype=torch.float32),
        )


class ActionAdapter(nn.Module):
    def __init__(
        self,
        cfg: UnifiedActionConfig,
        attention_runtime: AttentionBackendRuntime | None = None,
        *,
        encoder_cls: type[TaskUnifiedTokenBank] = TaskUnifiedTokenBank,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.l_max = cfg.l_max
        self.num_frames = cfg.num_frames
        self.group_size = cfg.group_size
        self.selfattn_action_token_reference_count = cfg.action_reference_count
        self.action_reads_video = True
        self.selfattn_video_token_reference_count = 0
        self.encoder = encoder_cls(cfg)
        self.selfattn_action_log_reference_count = nn.Parameter(
            torch.tensor(math.log(float(cfg.action_reference_count)), dtype=torch.float32),
            requires_grad=False,
        )

        if attention_runtime is None:
            attention_runtime = AttentionBackendRuntime(
                requested="legacy",
                allow_fallback=False,
                selected="legacy",
            )
        self.attention_runtime = attention_runtime

    def _apply(self, fn, recurse: bool = True):
        value = self.selfattn_action_log_reference_count.detach().float().clone()
        grad = self.selfattn_action_log_reference_count.grad
        grad = None if grad is None else grad.detach().float().clone()
        super()._apply(fn, recurse=recurse)
        parameter = self.selfattn_action_log_reference_count
        parameter.data = value.to(device=parameter.device, dtype=torch.float32)
        if grad is not None:
            parameter.grad = grad.to(device=parameter.device, dtype=torch.float32)
        return self

    @property
    def expected_trainable_tensor_count(self) -> int:
        return 9

    def encode_sample(self, sample, *, device, dtype=torch.bfloat16):

        if "action_token_value" in sample:
            raise ValueError("continuous token values require an explicit value-capable adapter")
        return _encode_sample_condition(self, sample, device=device, dtype=dtype)


def build_action_adapter(
    config: ExperimentConfig,
    dit: nn.Module,
    *,
    attention_runtime: AttentionBackendRuntime | None = None,
    encoder_cls: type[TaskUnifiedTokenBank] = TaskUnifiedTokenBank,
    adapter_cls: type[ActionAdapter] = ActionAdapter,
) -> ActionAdapter:

    action = config.action
    return adapter_cls(
        UnifiedActionConfig(
            l_max=action.adapter_l_max,
            num_frames=config.model.num_frames,
            dit_dim=int(dit.dim),
            num_dit_blocks=len(dit.blocks),
            group_size=action.time_group_size,
            video_height=config.model.height,
            video_width=config.model.width,
            hand_vocab_size=action.hand_vocab_size,
            camera_vocab_size=action.camera_vocab_size,
            robot_vocab_size=action.robot_vocab_size,
            force_start_token_id=action.force_start_token_id,
            force_endpoint_token_id=action.force_endpoint_token_id,
            action_reference_count=action.reference_count_init,
            torch_dtype=torch.bfloat16,
        ),
        attention_runtime=attention_runtime,
        encoder_cls=encoder_cls,
    )


@dataclass
class _ActionState:
    tokens: torch.Tensor | None = None
    uv: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    time_ids: torch.Tensor | None = None
    freqs: torch.Tensor | None = None
    zero_t_mod: torch.Tensor | None = None
    cond_size: int = 0
    video_seq_len: int | None = None
    video_grid: tuple[int, int, int] | None = None


def infer_video_grid(
    seq_len: int,
    *,
    num_frames: int,
    group_size: int,
    video_height: int,
    video_width: int,
) -> tuple[int, int, int]:
    f_guess = max(1, int(math.ceil(float(num_frames) / float(group_size))))
    ratio = float(video_width) / max(float(video_height), 1.0)

    def score(f: int, h: int, w: int) -> float:
        return abs((w / max(h, 1)) - ratio) + 0.01 * abs(f - f_guess)

    candidates: list[tuple[float, int, int, int]] = []
    if seq_len % f_guess == 0:
        spatial = seq_len // f_guess
        h_guess = max(1, int(round(math.sqrt(spatial / max(ratio, 1e-6)))))
        for h in range(max(1, h_guess - 4), h_guess + 5):
            if spatial % h == 0:
                w = spatial // h
                candidates.append((score(f_guess, h, w), f_guess, h, w))
    max_f = min(seq_len, max(f_guess * 2 + 4, f_guess + 8))
    for f in range(1, max_f + 1):
        if seq_len % f:
            continue
        spatial = seq_len // f
        h_guess = max(1, int(round(math.sqrt(spatial / max(ratio, 1e-6)))))
        for h in range(max(1, h_guess - 4), h_guess + 5):
            if spatial % h == 0:
                w = spatial // h
                candidates.append((score(f, h, w), f, h, w))
    if not candidates:
        raise RuntimeError(f"cannot infer video grid from seq_len={seq_len}")
    _, f, h, w = min(candidates, key=lambda item: item[0])
    return f, h, w


def wan_rope_at_positions(
    positions: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    positions = positions.reshape(-1).to(torch.float64)
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device)[: dim // 2] / dim)
    )
    angles = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


def _action_freqs(
    adapter: ActionAdapter,
    state: _ActionState,
    video_freqs: torch.Tensor,
) -> torch.Tensor:
    if (
        state.freqs is not None
        and state.freqs.device == video_freqs.device
        and state.freqs.dtype == video_freqs.dtype
    ):
        return state.freqs
    if state.uv is None or state.time_ids is None:
        raise RuntimeError("action UV/time state is missing")
    video_seq_len = int(video_freqs.shape[0])
    f, h, w = infer_video_grid(
        video_seq_len,
        num_frames=adapter.num_frames,
        group_size=adapter.group_size,
        video_height=adapter.cfg.video_height,
        video_width=adapter.cfg.video_width,
    )
    state.video_seq_len = video_seq_len
    state.video_grid = (f, h, w)
    uv = state.uv[0].to(device=video_freqs.device, dtype=torch.float32)
    time_ids = state.time_ids[0].to(device=video_freqs.device, dtype=torch.float32)
    t_pos = (
        time_ids * (float(f - 1) / float(adapter.num_frames - 1))
        if adapter.num_frames > 1 and f > 1
        else torch.zeros_like(time_ids)
    )
    v_norm = uv[:, 1].clamp(0.0, 1.0)
    u_norm = uv[:, 0].clamp(0.0, 1.0)
    h_pos = (v_norm * float(h) - 0.5) if h > 1 else torch.zeros_like(v_norm)
    w_pos = (u_norm * float(w) - 0.5) if w > 1 else torch.zeros_like(u_norm)
    head_dim = int(video_freqs.shape[-1]) * 2
    dim_h = head_dim // 3
    dim_w = head_dim // 3
    dim_t = head_dim - dim_h - dim_w
    state.freqs = (
        torch.cat(
            [
                wan_rope_at_positions(t_pos, dim_t),
                wan_rope_at_positions(h_pos, dim_h),
                wan_rope_at_positions(w_pos, dim_w),
            ],
            dim=-1,
        )
        .reshape(uv.shape[0], 1, -1)
        .to(dtype=video_freqs.dtype)
    )
    return state.freqs


def _zero_t_mod(
    dit: nn.Module,
    state: _ActionState,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cached = state.zero_t_mod
    if (
        cached is not None
        and cached.shape[0] == batch
        and cached.dtype == dtype
        and cached.device == device
    ):
        return cached
    timestep = torch.zeros((batch,), dtype=dtype, device=device)
    embedded = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(dtype))
    state.zero_t_mod = dit.time_projection(embedded).unflatten(1, (6, dit.dim))
    return state.zero_t_mod


def _modulation_chunks(block: nn.Module, t_mod: torch.Tensor) -> tuple[torch.Tensor, ...]:
    has_sequence = t_mod.ndim == 4
    chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
        6, dim=2 if has_sequence else 1
    )
    return tuple(value.squeeze(2) for value in chunks) if has_sequence else chunks


def action_key_log_weight(
    *,
    video_token_count: int,
    action_token_count: int,
    action_log_reference: torch.Tensor,
) -> torch.Tensor:
    if video_token_count <= 0 or action_token_count <= 0:
        raise ValueError("video and action token counts must be positive")
    if action_log_reference.numel() != 1:
        raise ValueError("action_log_reference must be scalar")
    return action_log_reference.reshape(()) - math.log(float(action_token_count))


def _project(module: nn.Module, value: torch.Tensor, *, action_branch: bool) -> torch.Tensor:
    if not action_branch and hasattr(module, "base_layer"):
        return module.base_layer(value)
    return module(value)


def _joint_self_attention(
    self_attn: nn.Module,
    attention_runtime: AttentionBackendRuntime,
    video: torch.Tensor,
    action: torch.Tensor,
    video_freqs: torch.Tensor,
    action_freqs: torch.Tensor,
    log_reference: torch.Tensor,
    *,
    action_reads_video: bool,
) -> tuple[torch.Tensor, torch.Tensor]:

    q_video = self_attn.norm_q(_project(self_attn.q, video, action_branch=False))
    k_video = self_attn.norm_k(_project(self_attn.k, video, action_branch=False))
    v_video = _project(self_attn.v, video, action_branch=False)
    rope_backend = getattr(self_attn, "rope_backend", "baseline")
    q_video = rope_apply(q_video, video_freqs, self_attn.num_heads, backend=rope_backend)
    k_video = rope_apply(k_video, video_freqs, self_attn.num_heads, backend=rope_backend)

    q_action = self_attn.norm_q(_project(self_attn.q, action, action_branch=True))
    k_action = self_attn.norm_k(_project(self_attn.k, action, action_branch=True))
    v_action = _project(self_attn.v, action, action_branch=True)
    q_action = rope_apply(q_action, action_freqs, self_attn.num_heads, backend=rope_backend)
    k_action = rope_apply(k_action, action_freqs, self_attn.num_heads, backend=rope_backend)

    action_bias = action_key_log_weight(
        video_token_count=k_video.shape[1],
        action_token_count=k_action.shape[1],
        action_log_reference=log_reference,
    )
    full_k = torch.cat([k_video, k_action], dim=1)
    full_v = torch.cat([v_video, v_action], dim=1)
    key_bias = torch.cat(
        [
            torch.zeros(k_video.shape[1], dtype=k_video.dtype, device=k_video.device),
            torch.ones(k_action.shape[1], dtype=k_video.dtype, device=k_video.device)
            * action_bias.to(dtype=k_video.dtype, device=k_video.device),
        ]
    )
    video_output = attention_runtime.attention(
        self_attn.attn,
        q_video,
        full_k,
        full_v,
        action_bias=action_bias,
        key_log_bias=key_bias,
        video_token_count=k_video.shape[1],
        num_heads=self_attn.num_heads,
    )
    if action_reads_video:
        action_output = self_attn.attn(q_action, full_k, full_v)
    else:
        action_output = self_attn.attn(q_action, k_action, v_action)
    return (
        _project(self_attn.o, video_output, action_branch=False),
        _project(self_attn.o, action_output, action_branch=True),
    )


def _routed_ffn(ffn: nn.Module, value: torch.Tensor, *, action_branch: bool) -> torch.Tensor:
    for layer in ffn:
        value = _project(layer, value, action_branch=action_branch)
    return value


def _action_block_forward(
    dit: nn.Module,
    adapter: ActionAdapter,
    block: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    video_freqs: torch.Tensor,
    state: _ActionState,
) -> torch.Tensor:
    if state.tokens is None:
        raise RuntimeError("action block called without a staged condition")
    if getattr(dit, "_action_lora_routing", None) != "action_branch":
        raise RuntimeError("Stage 2 requires action_branch LoRA routing")
    video_seq_len = int(video_freqs.shape[0])
    action_tokens = state.tokens.to(dtype=x.dtype, device=x.device)
    if action_tokens.shape[0] == 1 and x.shape[0] > 1:
        action_tokens = action_tokens.expand(x.shape[0], -1, -1)
    elif action_tokens.shape[0] != x.shape[0]:
        raise RuntimeError("action and video batch sizes differ")
    action_count = action_tokens.shape[1]
    if x.shape[1] == video_seq_len:
        x = torch.cat([x, action_tokens], dim=1)
    elif x.shape[1] != video_seq_len + action_count:
        raise RuntimeError("unexpected video/action sequence length")
    state.cond_size = action_count
    state.video_seq_len = video_seq_len
    video = x[:, :video_seq_len]
    action = x[:, video_seq_len:]
    action_freqs = _action_freqs(adapter, state, video_freqs)
    if action_freqs.shape[0] != action_count:
        raise RuntimeError("action RoPE and token lengths differ")

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _modulation_chunks(
        block, t_mod
    )
    action_t_mod = _zero_t_mod(
        dit,
        state,
        batch=action.shape[0],
        dtype=t_mod.dtype,
        device=t_mod.device,
    )
    (
        action_shift_msa,
        action_scale_msa,
        action_gate_msa,
        action_shift_mlp,
        action_scale_mlp,
        action_gate_mlp,
    ) = _modulation_chunks(block, action_t_mod)
    video_input = modulate(block.norm1(video), shift_msa, scale_msa)
    action_input = modulate(block.norm1(action), action_shift_msa, action_scale_msa)
    video_attn, action_attn = _joint_self_attention(
        block.self_attn,
        adapter.attention_runtime,
        video_input,
        action_input,
        video_freqs,
        action_freqs,
        adapter.selfattn_action_log_reference_count,
        action_reads_video=adapter.action_reads_video,
    )
    video = block.gate(video, gate_msa, video_attn)
    video = video + block.cross_attn(block.norm3(video), context)
    video_input = modulate(block.norm2(video), shift_mlp, scale_mlp)
    video = block.gate(
        video,
        gate_mlp,
        _routed_ffn(block.ffn, video_input, action_branch=False),
    )
    action = block.gate(action, action_gate_msa, action_attn)
    action_input = modulate(block.norm2(action), action_shift_mlp, action_scale_mlp)
    action = block.gate(
        action,
        action_gate_mlp,
        _routed_ffn(block.ffn, action_input, action_branch=True),
    )
    return torch.cat([video, action], dim=1)


def attach_action_adapter(
    dit: nn.Module,
    adapter: ActionAdapter,
    *,
    register_adapter: bool = True,
) -> None:
    if getattr(dit, "_action_adapter_attached", False):
        if getattr(dit, "_action_adapter_ref", None) is not adapter:
            raise RuntimeError("a different action adapter is already attached")
        return
    dit._action_state = _ActionState()

    if register_adapter:
        dit._action_adapter_ref = adapter
    else:
        object.__setattr__(dit, "_action_adapter_ref", adapter)
    dit._action_lora_routing = "action_branch"
    for block_id, block in enumerate(dit.blocks):
        block._action_block_id = block_id
        original_forward = block.forward

        if block_id == 0:

            def patched_forward(
                self,
                x,
                context,
                t_mod,
                freqs,
                _original_forward=original_forward,
            ):
                state: _ActionState = dit._action_state
                if state.tokens is None:
                    return _original_forward(x, context, t_mod, freqs)

                state.freqs = None
                state.zero_t_mod = None
                return _action_block_forward(dit, adapter, self, x, context, t_mod, freqs, state)
        else:

            def patched_forward(
                self,
                x,
                context,
                t_mod,
                freqs,
                _original_forward=original_forward,
            ):
                state: _ActionState = dit._action_state
                if state.tokens is None:
                    return _original_forward(x, context, t_mod, freqs)
                return _action_block_forward(dit, adapter, self, x, context, t_mod, freqs, state)

        block.forward = types.MethodType(patched_forward, block)

    original_head_forward = dit.head.forward

    def patched_head(self, x, t_mod, _original_forward=original_head_forward):
        state: _ActionState = dit._action_state
        if state.video_seq_len is not None and x.shape[1] > state.video_seq_len:
            x = x[:, : state.video_seq_len]
        return _original_forward(x, t_mod)

    dit.head.forward = types.MethodType(patched_head, dit.head)
    dit._action_adapter_attached = True


def _store_condition(
    dit: nn.Module,
    tokens: torch.Tensor,
    uv: torch.Tensor,
    valid: torch.Tensor,
    time_ids: torch.Tensor,
) -> None:
    if not getattr(dit, "_action_adapter_attached", False):
        raise RuntimeError("attach_action_adapter must be called first")
    state: _ActionState = dit._action_state
    state.tokens = tokens
    state.uv = uv
    state.valid = valid
    state.time_ids = time_ids
    state.freqs = None
    state.zero_t_mod = None
    state.cond_size = int(tokens.shape[1])
    state.video_seq_len = None
    state.video_grid = None


def stage_dense_condition(
    dit: nn.Module,
    adapter: ActionAdapter,
    action: torch.Tensor,
    time_mask: torch.Tensor,
    dim_mask: torch.Tensor,
    point_time_mask: torch.Tensor | None = None,
) -> None:
    _store_condition(
        dit,
        *adapter.encoder.forward_dense(action, time_mask, dim_mask, point_time_mask),
    )


def stage_sparse_condition(
    dit: nn.Module,
    adapter: ActionAdapter,
    uv: torch.Tensor,
    time_ids: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    family: str,
) -> None:
    _store_condition(
        dit,
        *adapter.encoder.forward_sparse(uv, time_ids, token_ids, family=family),
    )


def stage_sample_condition(
    dit: nn.Module,
    adapter: ActionAdapter,
    sample: dict[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    _store_condition(dit, *adapter.encode_sample(sample, device=device, dtype=dtype))


def _encode_sample_condition(
    adapter: ActionAdapter,
    sample: dict[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
):
    if "action" in sample:
        action = sample["action"].to(device=device, dtype=dtype)
        time_mask = sample["action_time_mask"].to(device=device)
        dim_mask = sample["action_dim_mask"].to(device=device)
        point_time_mask = sample.get("action_point_time_mask")
        if point_time_mask is not None:
            point_time_mask = point_time_mask.to(device=device)
        if action.ndim == 2:
            action = action.unsqueeze(0)
            time_mask = time_mask.unsqueeze(0)
            dim_mask = dim_mask.unsqueeze(0)
            if point_time_mask is not None:
                point_time_mask = point_time_mask.unsqueeze(0)
        if action.shape[-1] > adapter.l_max:
            raise ValueError(f"dense action width {action.shape[-1]} exceeds {adapter.l_max}")
        if action.shape[-1] < adapter.l_max:
            padding = adapter.l_max - action.shape[-1]
            action = F.pad(action, (0, padding))
            dim_mask = F.pad(dim_mask, (0, padding), value=False)
            if point_time_mask is not None:
                if padding % 2:
                    raise ValueError("dense action padding must preserve UV pairs")
                point_time_mask = F.pad(point_time_mask, (0, padding // 2), value=False)
        return adapter.encoder.forward_dense(
            action,
            time_mask,
            dim_mask,
            point_time_mask,
        )

    family = str(sample.get("task", ""))
    if family not in {"force", "camera", "robot"}:
        raise ValueError(f"cannot route sparse action family {family!r}")
    return adapter.encoder.forward_sparse(
        sample["action_token_uv"].to(device=device, dtype=torch.float32),
        sample["action_token_time"].to(device=device, dtype=torch.float32),
        sample["action_token_id"].to(device=device, dtype=torch.long),
        family=family,
    )


def clear_action_condition(dit: nn.Module) -> None:
    state: _ActionState | None = getattr(dit, "_action_state", None)
    if state is None:
        return
    state.tokens = None
    state.uv = None
    state.valid = None
    state.time_ids = None
    state.freqs = None
    state.zero_t_mod = None
    state.cond_size = 0
    state.video_seq_len = None
    state.video_grid = None


@contextmanager
def suspend_action_condition(dit: nn.Module):

    state: _ActionState | None = getattr(dit, "_action_state", None)
    if state is None:
        yield
        return
    dit._action_state = _ActionState()
    try:
        yield
    finally:
        dit._action_state = state
