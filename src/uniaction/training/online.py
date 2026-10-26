from collections import OrderedDict

import torch

from .encoding import encode_prompt_context, encode_training_video
from .prepare import load_preprocessors, split_action


class OnlineEncoder:
    def __init__(self, base_model, device):
        self.pipe, self.model = load_preprocessors(base_model, device)
        self.contexts = OrderedDict()
        self.empty_context = encode_prompt_context(self.pipe, "", output_device=device)

    @torch.no_grad()
    def __call__(self, sample):
        prompt = sample["prompt"]
        if prompt not in self.contexts:
            self.contexts[prompt] = encode_prompt_context(self.pipe, prompt)
            if len(self.contexts) > 64:
                self.contexts.popitem(last=False)
        self.contexts.move_to_end(prompt)
        visual = encode_training_video(
            self.pipe,
            self.model,
            sample["video"],
            prompt,
            precomputed_context=self.contexts[prompt],
            output_device=self.pipe.device,
        )
        tensors, fields = split_action(sample)
        return dict(
            fields,
            **tensors,
            **visual,
            task=sample["task"],
            precomputed_height=self.model.height,
            precomputed_width=self.model.width,
            precomputed_num_frames=self.model.num_frames,
        )
