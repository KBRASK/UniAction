from pathlib import Path
import json

import numpy as np
from torch.utils.data import Dataset

from uniaction.data.camera_keys import key_tokens
from .common import load_video, source_path, resize_center_crop_frames


class CameraDataset(Dataset):
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.records = [json.loads(line) for line in Path(metadata).read_text().splitlines()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        frames, fps = load_video(source_path(self.root, row["video"]))
        if len(frames) != 81 or fps is None or abs(fps - 16) > 1e-6 or len(row["key_states"]) != 80:
            raise ValueError("Camera samples require 81 frames at 16 fps and 80 key intervals")
        times = (np.arange(80) + 0.5).tolist()
        return dict(
            video=resize_center_crop_frames(frames, width=832, height=480),
            prompt=row["prompt"],
            semantic_task="camera",
            camera_layout="screen_shared",
            camera_event_times=times,
            action_normalization="token_list",
            action_token_vocab_size=16,
            **key_tokens(row["key_states"], times),
        )
