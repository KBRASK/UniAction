from pathlib import Path
import random

import numpy as np
import torch

from .api import frames, trajectory


def queries(command_uv, command_valid, task):
    uv, valid = np.asarray(command_uv, dtype=float), np.asarray(command_valid, dtype=bool)
    if task not in ("hand", "robot") or uv.shape != (81, 12, 2) or valid.shape != (81, 12):
        raise ValueError("Expected a hand or robot condition with 81 frames and 12 points")
    valid = valid & np.isfinite(uv).all(-1)
    times = np.full(12, -1, dtype=int)
    seeds = np.full((12, 2), np.nan)
    for point in range(12):
        observed = np.flatnonzero(valid[:, point])
        if not len(observed) or (task == "robot" and observed[0] != 0):
            continue
        times[point] = observed[0]
        seeds[point] = uv[observed[0], point] * [832.0, 480.0]
    inside = (seeds >= 0).all(-1) & (seeds < [832.0, 480.0]).all(-1)
    times[~inside] = -1
    if task == "robot" and not (times == 0).any():
        raise ValueError("Robot tracking requires an observed frame-zero query")
    return seeds, times


class PoseEvaluator:
    def __init__(self, weights=None, device="cuda"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and str(torch.__version__) != "2.10.0+cu128":
            raise ValueError("Calibrated pose evaluation requires PyTorch 2.10.0 with CUDA 12.8")
        from cotracker.predictor import CoTrackerPredictor

        if weights is None:
            raise ValueError("Provide the local CoTracker3 offline checkpoint with --weights")
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        torch.backends.cudnn.benchmark = False
        self.model = (
            CoTrackerPredictor(checkpoint=str(weights), offline=True).to(self.device).eval()
        )

    @torch.inference_mode()
    def __call__(self, generated, command_uv, command_valid, *, task, static=False):
        video = frames(generated)
        seeds, times = queries(command_uv, command_valid, task)
        ids = np.flatnonzero(times >= 0)
        measured = np.full((81, 12, 2), np.nan)
        valid = np.zeros((81, 12), dtype=bool)
        if len(ids):
            query = np.column_stack([times[ids], seeds[ids]]).astype(np.float32)
            tensor = (
                torch.from_numpy(video.copy()).permute(0, 3, 1, 2)[None].float().to(self.device)
            )
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(0)
            tracks, visible = self.model(
                tensor,
                queries=torch.from_numpy(query)[None].to(self.device),
                backward_tracking=False,
            )
            tracks, visible = tracks[0].cpu().numpy(), visible[0].cpu().numpy()
            for column, point in enumerate(ids):
                measured[:, point] = tracks[:, column]
                valid[:, point] = visible[:, column] & np.isfinite(tracks[:, column]).all(-1)
                valid[: times[point], point] = False
                valid[:, point] &= (tracks[:, column] >= 0).all(-1)
                valid[:, point] &= (tracks[:, column] < [832.0, 480.0]).all(-1)
        result = trajectory(measured, valid, command_uv, command_valid, task=task, static=static)
        return result, dict(
            measured_px=measured,
            measured_valid=valid,
            command_uv=command_uv,
            command_valid=command_valid,
            seed_points_px=seeds,
            seed_frames=times,
        )
