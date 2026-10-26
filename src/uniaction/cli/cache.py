import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Encode 81-frame training sources or index completed cache streams."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode")
    encode.add_argument("--base-model", required=True)
    encode.add_argument("--sources", help="training-sources.json included with the training set")
    encode.add_argument(
        "--stream",
        choices=(
            "hand",
            "force_balls",
            "force_dominos",
            "force_flowers",
            "robot",
            "camera",
        ),
    )
    encode.add_argument("--source-root")
    encode.add_argument("--metadata")
    encode.add_argument("--output", required=True)
    encode.add_argument("--device", default="cuda")
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--cache-root", required=True)
    manifest.add_argument("--output", required=True)
    args = parser.parse_args()
    from uniaction.training.prepare import encode_sources, encode_stream, cache_manifest

    if args.command == "encode":
        single = (args.stream, args.source_root, args.metadata)
        if args.sources:
            if any(single):
                parser.error("--sources cannot be combined with per-stream arguments")
            encode_sources(args)
        else:
            if not all(single):
                parser.error("provide --sources, or --stream, --source-root, --metadata")
            encode_stream(args)
    else:
        print(cache_manifest(args.cache_root, args.output))


if __name__ == "__main__":
    main()
