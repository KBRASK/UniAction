from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate an action-conditioned video with the released UniAction model."
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--trained", action="store_true", help="use a package exported from your own training run"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--condition",
        type=Path,
        required=True,
        help="NPZ action input: force, camera, hand, or robot",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overlay", action="store_true", help="also save a condition-overlay video"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--text-guidance", type=float, default=1.0)
    parser.add_argument("--action-guidance", type=float, default=2.0)
    parser.add_argument(
        "--guidance", choices=("separate", "joint", "counterfactual"), default="separate"
    )
    parser.add_argument("--attention-backend", choices=("fa4", "sdpa"), default="fa4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--negative-prompt")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new output path")
    if args.overlay:
        import importlib

        importlib.import_module("uniaction.overlay")
        if args.output.with_name(args.output.stem + "-overlay.mp4").exists():
            parser.error("Overlay output already exists; choose a new output path")
    from uniaction.conditions import load_condition
    from uniaction.pipeline import DEFAULT_NEGATIVE_PROMPT, UniActionPipeline

    condition = load_condition(args.condition)
    pipe = UniActionPipeline.from_pretrained(
        args.base_model,
        args.checkpoint_dir,
        device=args.device,
        attention_backend=args.attention_backend,
        cache_dir=args.cache_dir,
        trained=args.trained,
    )
    pipe.generate(
        args.image,
        args.prompt,
        condition,
        output=args.output,
        seed=args.seed,
        steps=args.steps,
        text_guidance=args.text_guidance,
        action_guidance=args.action_guidance,
        guidance=args.guidance,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT
        if args.negative_prompt is None
        else args.negative_prompt,
        overlay=args.overlay,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
