import argparse


def main():
    parser = argparse.ArgumentParser(description="Train UniAction from videos or prepared caches.")
    parser.add_argument("--base-model", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sources", help="training-sources.json included with the training set")
    inputs.add_argument("--cache-manifest", help="use precomputed VAE and text-encoder caches")
    parser.add_argument("--stage", type=int, choices=(1, 2), help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage1-checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--stop-file", help=argparse.SUPPRESS)
    parser.add_argument(
        "--gpus", type=int, choices=(1, 2, 4, 8, 16), help="GPU count; defaults to all visible GPUs"
    )
    parser.add_argument("--bf16-manifest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--save-every", type=int, default=250, help="optimizer updates between recovery checkpoints"
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        help="optional shorter run; default 2500 updates = 40,000 samples",
    )
    args = parser.parse_args()
    if args.stage is None:
        if args.stage1_checkpoint or args.stop_file:
            parser.error("Checkpoint handoff is managed automatically")
        from uniaction.training.workflow import train_all

        train_all(args)
    else:
        from uniaction.training.runner import train

        train(args)


if __name__ == "__main__":
    main()
