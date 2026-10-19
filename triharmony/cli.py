"""Command-line entry point; numerical modules are imported only after setup."""
import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser(description="Tri-Harmony main-table reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-source", "prepare-target"):
        command = commands.add_parser(name, help="Build frozen features from a label-separated manifest")
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--checkpoint", required=True, help="Local DINOv3 safetensors weights")
        command.add_argument("--device", default="cuda")
        command.add_argument("--cache-dir")
        command.add_argument("--batch-size", type=int, default=16)
        if name == "prepare-target":
            command.add_argument("--source-memory", required=True)
    for name in ("train", "infer"):
        command = commands.add_parser(name, help="Run the fixed Full method without evaluation labels")
        command.add_argument("--config", help="Defaults to the main seed-42 configuration")
        command.add_argument("--seed", type=int, help="Sole seed; 42 uses the original runtime, 7/21/3407 use archived one-thread runtime")
        command.add_argument("--cpu-threads", type=int, help="Explicit non-reference runtime override")
        command.add_argument("--target", required=True)
        command.add_argument("--output", required=True, help="New output directory; must not exist")
        command.add_argument("--device", default="cuda")
        if name == "train":
            command.add_argument("--source", required=True)
        else:
            command.add_argument("--checkpoint", required=True)
    command = commands.add_parser("lock-predictions", help="Freeze prediction hashes before evaluation")
    command.add_argument("--runs", nargs="+", required=True, help="Run directories or prediction JSON paths")
    command.add_argument("--output", required=True, help="New JSON manifest path")
    command = commands.add_parser("evaluate", help="Evaluate locked final predictions")
    command.add_argument("--predictions-manifest", required=True)
    command.add_argument("--labels", required=True)
    command.add_argument("--output", required=True)
    command = commands.add_parser("tables", help="Aggregate all 32 cohorts into main Tables 1--4")
    command.add_argument("--metrics", required=True)
    command.add_argument("--output", required=True)
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command.startswith("prepare-"):
        from .preparation import prepare_source_inputs, prepare_target_inputs
        action = prepare_source_inputs if args.command == "prepare-source" else prepare_target_inputs
    elif args.command in ("train", "infer"):
        from .runner import train, infer
        action = train if args.command == "train" else infer
    elif args.command == "lock-predictions":
        from .evaluation import lock_predictions
        action = lock_predictions
    elif args.command == "evaluate":
        from .evaluation import evaluate
        action = evaluate
    else:
        from .tables import aggregate
        action = aggregate
    print(json.dumps(action(args), indent=2, ensure_ascii=False))
