import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Write a camera or force condition for an 832x480 first frame."
    )
    parser.add_argument("--task", choices=("camera", "force"), required=True)
    parser.add_argument("--key", choices=tuple("WASDIJKL"), default="W")
    parser.add_argument("--start", nargs=2, type=float, default=(0.45, 0.6))
    parser.add_argument("--end", nargs=2, type=float, default=(0.60, 0.6))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from uniaction import conditions

    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    if args.output.suffix != ".npz":
        parser.error("Output must end with .npz")
    if args.task == "camera":
        conditions.camera(args.key)
        values = dict(task="camera", key=args.key)
    else:
        conditions.force(args.start, args.end)
        values = dict(
            task="force",
            start=np.array(args.start, dtype=np.float32),
            end=np.array(args.end, dtype=np.float32),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    print(args.output)


if __name__ == "__main__":
    main()
