# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for UniAction; upstream copyright and Apache-2.0 attribution are in NOTICE.
from __future__ import annotations

import torch


def _custom_forward(module):
    def forward(*args, **kwargs):
        return module(*args, **kwargs)

    return forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *args,
    preserve_rng_state: bool = True,
    **kwargs,
):

    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                _custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
                preserve_rng_state=preserve_rng_state,
            )
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            _custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
        )
    return model(*args, **kwargs)
