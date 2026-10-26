import numpy as np

CAMERA_KEYS = ("W", "A", "S", "D", "I", "K", "J", "L")
KEY_TO_ID = {key: index for index, key in enumerate(CAMERA_KEYS)}
OPPOSITE_KEY = dict(zip(CAMERA_KEYS, ("S", "D", "W", "A", "K", "I", "L", "J")))
SCREEN_SHARED_UV = np.asarray(
    [
        (0.50, 0.24),
        (0.24, 0.50),
        (0.50, 0.76),
        (0.76, 0.50),
        (0.50, 0.24),
        (0.50, 0.76),
        (0.24, 0.50),
        (0.76, 0.50),
    ],
    dtype=np.float32,
)


def key_tokens(states, midpoints):
    import torch

    ids, times = [], []
    for state, time in zip(states, midpoints, strict=True):
        if any(key not in KEY_TO_ID for key in state) or len(set(state)) != len(state):
            raise ValueError("invalid camera key state")
        for key in state:
            ids.append(KEY_TO_ID[key])
            times.append(float(time))
    token_ids = np.asarray(ids, dtype=np.int64)
    return {
        "action_token_id": torch.from_numpy(token_ids),
        "action_token_time": torch.tensor(times, dtype=torch.float32),
        "action_token_uv": torch.from_numpy(SCREEN_SHARED_UV[token_ids].copy()),
    }
