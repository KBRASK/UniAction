# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.
# Modified for UniAction; upstream copyright and Apache-2.0 attribution are in NOTICE.
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class FlowUniPCScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: Sequence[int] | None = None,
    ) -> None:
        if solver_order < 1:
            raise ValueError("solver_order must be positive")
        if solver_type not in ("bh1", "bh2"):
            raise ValueError(f"Unsupported UniPC solver_type: {solver_type}")

        self.num_train_timesteps = int(num_train_timesteps)
        self.solver_order = int(solver_order)
        self.solver_type = solver_type
        self.lower_order_final = bool(lower_order_final)
        self.disable_corrector = set(disable_corrector or ())

        alphas = np.linspace(
            1.0,
            1.0 / self.num_train_timesteps,
            self.num_train_timesteps,
        )[::-1].copy()
        base_sigmas = torch.from_numpy(1.0 - alphas).to(torch.float32)
        self.sigma_max = float(base_sigmas[0])
        self.sigma_min = float(base_sigmas[-1])

        self.num_inference_steps: int | None = None
        self.sigmas = base_sigmas.cpu()
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self.timestep_list: list[torch.Tensor | None] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 1
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        denoising_strength: float = 1.0,
        shift: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not 0.0 < denoising_strength <= 1.0:
            raise ValueError("denoising_strength must be inside (0, 1]")
        if shift <= 0:
            raise ValueError("shift must be positive")

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(denoising_strength)
        sigmas = np.linspace(
            sigma_start,
            self.sigma_min,
            int(num_inference_steps) + 1,
        ).copy()[:-1]
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).cpu()
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device,
            dtype=torch.int64,
        )
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 1
        self._step_index = None
        self._begin_index = None

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1.0 - sigma, sigma

    def _convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self.step_index is None:
            raise RuntimeError("UniPC step index has not been initialized")
        sigma = self.sigmas[self.step_index]
        return sample - sigma * model_output

    def _multistep_uni_p_bh_update(
        self,
        sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        if self.step_index is None:
            raise RuntimeError("UniPC step index has not been initialized")
        m0 = self.model_outputs[-1]
        if m0 is None:
            raise RuntimeError("UniPC predictor is missing the current model output")

        sigma_t = self.sigmas[self.step_index + 1]
        sigma_s0 = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = self.model_outputs[-(i + 1)]
            if mi is None:
                raise RuntimeError("UniPC predictor history is incomplete")
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.tensor(1.0))
        rks_tensor = torch.stack(rks).to(sample.device)
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0
        b_h = hh if self.solver_type == "bh1" else torch.expm1(hh)

        r_matrix = []
        b_vector = []
        factorial_i = 1
        for i in range(1, order + 1):
            r_matrix.append(torch.pow(rks_tensor, i - 1))
            b_vector.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        r_matrix = torch.stack(r_matrix)
        b_tensor = torch.stack(b_vector).to(sample.device)

        if d1s:
            d1_tensor = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
            else:
                rhos_p = torch.linalg.solve(r_matrix[:-1, :-1], b_tensor[:-1]).to(sample.dtype)
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1_tensor)
        else:
            pred_res = 0

        sample_next = (
            sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0 - alpha_t * b_h * pred_res
        )
        return sample_next.to(sample.dtype)

    def _multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        if self.step_index is None:
            raise RuntimeError("UniPC step index has not been initialized")
        m0 = self.model_outputs[-1]
        if m0 is None:
            raise RuntimeError("UniPC corrector is missing predictor history")

        sigma_t = self.sigmas[self.step_index]
        sigma_s0 = self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = self.model_outputs[-(i + 1)]
            if mi is None:
                raise RuntimeError("UniPC corrector history is incomplete")
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.tensor(1.0))
        rks_tensor = torch.stack(rks).to(this_sample.device)
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0
        b_h = hh if self.solver_type == "bh1" else torch.expm1(hh)

        r_matrix = []
        b_vector = []
        factorial_i = 1
        for i in range(1, order + 1):
            r_matrix.append(torch.pow(rks_tensor, i - 1))
            b_vector.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        r_matrix = torch.stack(r_matrix)
        b_tensor = torch.stack(b_vector).to(this_sample.device)

        d1_tensor = torch.stack(d1s, dim=1) if d1s else None
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=last_sample.dtype, device=last_sample.device)
        else:
            rhos_c = torch.linalg.solve(r_matrix, b_tensor).to(last_sample.dtype)

        if d1_tensor is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1_tensor)
        else:
            corr_res = 0
        d1_t = this_model_output - m0
        corrected = (
            sigma_t / sigma_s0 * last_sample
            - alpha_t * h_phi_1 * m0
            - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        )
        return corrected.to(last_sample.dtype)

    def index_for_timestep(
        self,
        timestep: int | torch.Tensor,
        schedule_timesteps: torch.Tensor | None = None,
    ) -> int:
        schedule = self.timesteps if schedule_timesteps is None else schedule_timesteps
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(schedule.device)
        indices = (schedule == timestep).nonzero(as_tuple=False)
        if len(indices) == 0:
            raise ValueError(f"Timestep {timestep} is not in the UniPC schedule")
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos].item())

    def _init_step_index(self, timestep: int | torch.Tensor) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self.begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before FlowUniPCScheduler.step")
        if self.step_index is None:
            self._init_step_index(timestep)
        if self.step_index is None:
            raise RuntimeError("UniPC step index initialization failed")

        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )
        converted = self._convert_model_output(model_output, sample)
        if use_corrector:
            sample = self._multistep_uni_c_bh_update(
                converted,
                self.last_sample,
                sample,
                self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = converted
        self.timestep_list[-1] = (
            timestep.detach().clone() if isinstance(timestep, torch.Tensor) else timestep
        )

        if self.lower_order_final:
            order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            order = self.solver_order
        self.this_order = min(order, self.lower_order_nums + 1)

        self.last_sample = sample
        previous = self._multistep_uni_p_bh_update(sample, self.this_order)
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1
        return previous
