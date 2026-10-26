# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for UniAction; upstream copyright and Apache-2.0 attribution are in NOTICE.
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from PIL import Image
from tqdm import tqdm

from .dit import WanModel, sinusoidal_embedding_1d
from .ops import gradient_checkpoint_forward
from .scheduler import FlowMatchScheduler


class PipelineUnit:
    def __init__(
        self,
        *,
        separate_cfg: bool = False,
        input_params: tuple[str, ...] = (),
        input_params_posi: dict[str, str] | None = None,
        input_params_nega: dict[str, str] | None = None,
    ) -> None:
        self.separate_cfg = separate_cfg
        self.input_params = input_params
        self.input_params_posi = input_params_posi or {}
        self.input_params_nega = input_params_nega or {}

    def process(self, pipe: "WanPipeline", **kwargs) -> dict:
        raise NotImplementedError


class PipelineUnitRunner:
    def __call__(
        self,
        unit: PipelineUnit,
        pipe: "WanPipeline",
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
    ) -> tuple[dict, dict, dict]:
        if unit.separate_cfg:
            positive = {
                name: inputs_posi.get(source) for name, source in unit.input_params_posi.items()
            }
            positive.update({name: inputs_shared.get(name) for name in unit.input_params})
            output = unit.process(pipe, **positive)
            inputs_posi.update(output)
            if inputs_shared["cfg_scale"] != 1:
                negative = {
                    name: inputs_nega.get(source) for name, source in unit.input_params_nega.items()
                }
                negative.update({name: inputs_shared.get(name) for name in unit.input_params})
                inputs_nega.update(unit.process(pipe, **negative))
            else:
                inputs_nega.update(output)
        else:
            values = {name: inputs_shared.get(name) for name in unit.input_params}
            inputs_shared.update(unit.process(pipe, **values))
        return inputs_shared, inputs_posi, inputs_nega


class WanPipeline(torch.nn.Module):
    def __init__(self, *, device="cpu", torch_dtype=torch.bfloat16) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.time_division_factor = 4
        self.time_division_remainder = 1
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer = None
        self.text_encoder = None
        self.image_encoder = None
        self.dit = None
        self.dit2 = None
        self.vae = None
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
        ]
        self.model_fn = model_fn_wan_video

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def load_models_to_device(self, _model_names: Sequence[str]) -> None:

        return None

    def check_resize_height_width(self, height, width, num_frames=None):
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1)
                // self.height_division_factor
                * self.height_division_factor
            )
        if width % self.width_division_factor != 0:
            width = (
                (width + self.width_division_factor - 1)
                // self.width_division_factor
                * self.width_division_factor
            )
        if num_frames is None:
            return height, width
        if num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames = (
                (num_frames + self.time_division_factor - 1)
                // self.time_division_factor
                * self.time_division_factor
                + self.time_division_remainder
            )
        return height, width, num_frames

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        image = torch.tensor(np.asarray(image, dtype=np.float32))
        image = image.to(
            dtype=torch_dtype or self.torch_dtype,
            device=device or self.device,
        )
        image = image * ((max_value - min_value) / 255) + min_value
        return repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):

        images = torch.from_numpy(
            np.stack(
                [np.asarray(image, dtype=np.float32) for image in video],
                axis=0,
            )
        )
        images = images.to(
            dtype=torch_dtype or self.torch_dtype,
            device=device or self.device,
        )
        images = images * ((max_value - min_value) / 255) + min_value
        frames = [repeat(image, "H W C -> B C H W", B=1) for image in images]
        return torch.stack(frames, dim=pattern.index("T") // 2)

    def vae_output_to_image(
        self,
        vae_output,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        return Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy())

    def vae_output_to_video(
        self,
        vae_output,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        return [
            self.vae_output_to_image(
                image,
                pattern="H W C",
                min_value=min_value,
                max_value=max_value,
            )
            for image in vae_output
        ]

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_torch_dtype=torch.float32,
        device=None,
        torch_dtype=None,
    ):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(
            shape,
            generator=generator,
            device=rand_device,
            dtype=rand_torch_dtype,
        )
        return noise.to(
            dtype=torch_dtype or self.torch_dtype,
            device=device or self.device,
        )

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        input_image: Image.Image | None = None,
        input_video: list[Image.Image] | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        switch_DiT_boundary: float = 0.875,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        sample_solver: str = "euler",
        tiled: bool = False,
        tile_size=None,
        tile_stride=None,
        progress_bar_cmd=tqdm,
        output_type: str = "quantized",
    ):
        if (
            self.dit is None
            or self.vae is None
            or self.text_encoder is None
            or self.tokenizer is None
        ):
            raise RuntimeError("inference requires DiT, VAE, text encoder, and tokenizer")
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            shift=sigma_shift,
            solver=sample_solver,
        )
        inputs_shared = {
            "input_image": input_image,
            "input_video": input_video,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": seed,
            "rand_device": rand_device,
            "cfg_scale": cfg_scale,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        active_dit = self.dit
        for step, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None:
                active_dit = self.dit2
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            positive = self.model_fn(
                dit=active_dit,
                **inputs_shared,
                **inputs_posi,
                timestep=timestep,
            )
            if cfg_scale != 1.0:
                negative = self.model_fn(
                    dit=active_dit,
                    **inputs_shared,
                    **inputs_nega,
                    timestep=timestep,
                )
                prediction = negative + cfg_scale * (positive - negative)
            else:
                prediction = positive
            inputs_shared["latents"] = self.scheduler.step(
                prediction,
                self.scheduler.timesteps[step],
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            return self.vae_output_to_video(video)
        if output_type == "floatpoint":
            return video
        raise ValueError(f"unsupported output_type {output_type!r}")


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "seed",
                "rand_device",
            )
        )

    def process(
        self,
        pipe,
        height,
        width,
        num_frames,
        seed,
        rand_device,
    ):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        return {"noise": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
            )
        )

    def process(
        self,
        pipe,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_video is None:
            return {"latents": noise}
        video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        return {
            "latents": pipe.scheduler.add_noise(
                input_latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
        }


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            separate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
        )

    def process(self, pipe, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        lengths = mask.gt(0).sum(dim=1).long()
        embedding = pipe.text_encoder(ids, mask)
        for index, length in enumerate(lengths):
            embedding[index, length:] = 0
        return {"context": embedding}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_image", "height", "width"))

    def process(self, pipe, input_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        context = pipe.image_encoder.encode_image([image])
        return {"clip_feature": context.to(dtype=pipe.torch_dtype, device=pipe.device)}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            )
        )

    def process(
        self,
        pipe,
        input_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        mask = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        mask[:, 1:] = 0
        vae_input = torch.cat(
            [
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width, device=image.device),
            ],
            dim=1,
        )
        mask = torch.cat([torch.repeat_interleave(mask[:, 0:1], 4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, height // 8, width // 8)
        mask = mask.transpose(1, 2)[0]
        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": torch.cat([mask, y]).unsqueeze(0).to(pipe.torch_dtype)}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            )
        )

    def process(
        self,
        pipe,
        input_image,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        first = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, 0:1] = first
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": first,
        }


def model_fn_wan_video(
    *,
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    checkpoint_preserve_rng_state: bool = True,
    checkpoint_exempt_blocks: int = 0,
    **_,
) -> torch.Tensor:

    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.cat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (
                        latents.shape[2] - 1,
                        latents.shape[3] * latents.shape[4] // 4,
                    ),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        time_embedding = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
        time_modulation = dit.time_projection(time_embedding).unflatten(2, (6, dit.dim))
    else:
        time_embedding = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        time_modulation = dit.time_projection(time_embedding).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.cat([x] * context.shape[0], dim=0)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        context = torch.cat([dit.img_emb(clip_feature), context], dim=1)

    x = dit.patchify(x)
    frames, height, width = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    frequencies = torch.cat(
        [
            dit.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            dit.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            dit.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1)
    if getattr(dit, "_uniaction_rope_backend", "baseline") == "quack":
        frequencies = frequencies.to(device=x.device, dtype=torch.complex64)
    else:
        frequencies = frequencies.to(x.device)

    if not 0 <= checkpoint_exempt_blocks <= len(dit.blocks):
        raise ValueError(
            f"checkpoint_exempt_blocks must be in [0,{len(dit.blocks)}], "
            f"got {checkpoint_exempt_blocks}"
        )
    checkpointed_block_count = len(dit.blocks) - checkpoint_exempt_blocks
    for block_index, block in enumerate(dit.blocks):
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing and block_index < checkpointed_block_count,
            use_gradient_checkpointing_offload and block_index < checkpointed_block_count,
            x,
            context,
            time_modulation,
            frequencies,
            preserve_rng_state=checkpoint_preserve_rng_state,
        )
    x = dit.head(x, time_embedding)
    return dit.unpatchify(x, (frames, height, width))
