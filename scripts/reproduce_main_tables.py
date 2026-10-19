#!/usr/bin/env python3
"""Run 32 main-table cohorts per seed from external datasets and fixed maps."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(REPO), check=True)


def verified_features(path, manifest, checkpoint, source_memory=None):
    if not path.exists():
        return False
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("status") != "complete":
        raise ValueError("Incomplete preparation; use a fresh output directory")
    if meta.get("manifest_sha256") != sha(manifest) or meta.get("backbone_sha256") != sha(checkpoint):
        raise ValueError("Preparation inputs changed")
    if source_memory is not None and meta.get("source_memory_sha256") != sha(source_memory):
        raise ValueError("Source memory changed; rebuild the dependent target features")
    for item in meta["outputs"]:
        if sha(path.parent / item["name"]) != item["sha256"]:
            raise ValueError("Prepared feature artifact changed")
    return True


def verified_run(path, source, target, seed):
    if not path.exists():
        return False
    manifest = json.loads((path / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("seed") != seed:
        raise ValueError("Incomplete or incompatible run; use a fresh output directory")
    for key, artifact in (
        ("source_sha256", source), ("target_sha256", target),
        ("prediction_sha256", path / "learned_predictions.json"),
        ("checkpoint_sha256", path / "checkpoint_epoch40.pt"),
        ("training_sha256", path / "training.json"),
    ):
        if manifest.get(key) != sha(artifact):
            raise ValueError("Changed run artifact: " + key)
    return True


def stage(output, name, inputs, destination, command):
    """Resume only a stage committed after its child command completed."""
    marker = output / ("stage_" + name + ".json")
    provenance = {str(path.relative_to(output)): sha(path) for path in inputs}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("inputs") != provenance or saved.get("command") != list(map(str, command)):
            raise ValueError("Changed stage inputs: " + name)
        actual = [destination] if destination.is_file() else sorted(p for p in destination.rglob("*") if p.is_file())
        recorded = saved.get("outputs", [])
        if not recorded or {p.relative_to(output).as_posix() for p in actual} != {r["path"] for r in recorded}:
            raise ValueError("Missing or extra stage output: " + name)
        for row in recorded:
            if sha(output / row["path"]) != row["sha256"]:
                raise ValueError("Changed stage output: " + name)
        return
    if destination.exists():
        raise ValueError("Uncommitted partial stage " + name + "; preserve it and use a fresh output directory")
    run(*command)
    artifacts = [destination] if destination.is_file() else sorted(p for p in destination.rglob("*") if p.is_file())
    if not artifacts:
        raise ValueError("Stage produced no outputs: " + name)
    marker.write_text(json.dumps({
        "inputs": provenance, "command": list(map(str, command)),
        "outputs": [{"path": p.relative_to(output).as_posix(), "sha256": sha(p)} for p in artifacts],
    }, indent=2) + "\n")


def validate_raw_inputs(output):
    from triharmony.preparation import _load_manifest
    for path in sorted((output / "manifests").glob("*.json")):
        if path.name.endswith("_labels.json"):
            continue
        _load_manifest(path, "source" if path.name == "source.json" else "target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="Local DINOv3 safetensors file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 21, 3407])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List paths and stages without reading data or running models")
    args = parser.parse_args()
    data, output, weights = (Path(x).expanduser().resolve() for x in
                             (args.data_root, args.output, args.checkpoint))
    seeds = list(dict.fromkeys(args.seeds))
    if set(seeds) - {42, 7, 21, 3407}:
        parser.error("The reference main-table seeds are 42, 7, 21 and 3407")
    registry = json.loads((REPO / "configs/main_cohorts.json").read_text())
    cohorts = [(protocol, dataset, category)
               for protocol in ("table1", "table2")
               for dataset in ("mvtec", "mtd")
               for category in sorted(registry[dataset])]
    if args.dry_run:
        print(json.dumps({
            "source_images": str(data / "AeBAD_crop"),
            "checkpoint": str(weights), "output": str(output), "seeds": seeds,
            "source_preparations": 1, "target_preparations": len(cohorts),
            "full_fits": len(cohorts) * len(seeds),
            "cohorts": [list(c) for c in cohorts],
            "final_outputs": ["table1_ours.csv", "table2_ours.csv",
                              "table3_categories.csv", "table4_categories.csv"] +
                              (["table1_mean_std.csv", "table2_mean_std.csv",
                                "table3_categories_mean_std.csv",
                                "table4_categories_mean_std.csv"] if len(seeds) > 1 else []),
        }, indent=2))
        return
    if not args.device.startswith("cuda"):
        parser.error("Main-table feature preparation uses CUDA. Use individual commands for CPU smoke tests.")
    if output.exists() and not args.resume:
        raise FileExistsError("Output exists; choose a new directory or explicitly use --resume")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256()
    for file in sorted(list((REPO / "triharmony").rglob("*.py")) +
                       list((REPO / "scripts").glob("*.py")) +
                       list((REPO / "configs").glob("*.json"))):
        code_hash.update(file.relative_to(REPO).as_posix().encode())
        code_hash.update(file.read_bytes())
    plan = {"schema_version": 1, "data_root": str(data), "backbone_sha256": sha(weights),
            "seeds": seeds, "device": args.device, "code_sha256": code_hash.hexdigest(),
            "config_sha256": sha(REPO / "configs/main_seed42.json")}
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Resume inputs or source code differ from the saved plan")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    sys.path.insert(0, str(REPO))
    completion = output / "complete.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved.get("plan") != plan:
            raise ValueError("Completed campaign plan differs")
        for entry in saved["outputs"]:
            if sha(output / entry["path"]) != entry["sha256"]:
                raise ValueError("A completed campaign artifact changed")
        validate_raw_inputs(output)
        print("Verified complete main-table campaign:", output / "tables")
        return
    manifests = output / "manifests"
    manifests.mkdir(exist_ok=True)
    source_manifest, source = manifests / "source.json", output / "features/source.pt"
    if not source_manifest.exists():
        run("-m", "triharmony.manifests", "source", "--root", data / "AeBAD_crop",
            "--output", source_manifest)
    # Validate raw-image/mask hashes on every invocation, including resume.
    sys.path.insert(0, str(REPO))
    from triharmony.preparation import _load_manifest
    _load_manifest(source_manifest, "source")
    if not verified_features(source, source_manifest, weights):
        run("-m", "triharmony", "prepare-source", "--manifest", source_manifest,
            "--checkpoint", weights, "--device", args.device, "--output", source)
    runs, label_files = [], []
    for protocol, dataset, category in cohorts:
        name = f"{protocol}_{dataset}_{category}"
        manifest, labels = manifests / (name + ".json"), manifests / (name + "_labels.json")
        root = data / ("mvtec" if dataset == "mvtec" else "mtd_anomaly_detection")
        maps = data / "maps" / ("musc" if protocol == "table1" else "cpr") / dataset
        if not manifest.exists():
            run("-m", "triharmony.manifests", "target", "--root", root, "--maps", maps,
                "--protocol", protocol, "--dataset", dataset, "--category", category,
                "--output", manifest, "--labels", labels)
        _load_manifest(manifest, "target")
        if not labels.is_file():
            raise FileNotFoundError(labels)
        target = output / ("features/" + name + ".pt")
        if not verified_features(target, manifest, weights, source.with_name("source_memory.pt")):
            run("-m", "triharmony", "prepare-target", "--manifest", manifest,
                "--source-memory", source.with_name("source_memory.pt"),
                "--checkpoint", weights, "--device", args.device, "--output", target)
        for seed in seeds:
            run_dir = output / "runs" / (name + "_s" + str(seed))
            if not verified_run(run_dir, source, target, seed):
                run("-m", "triharmony", "train", "--config", REPO / "configs/main_seed42.json",
                    "--seed", seed, "--source", source, "--target", target,
                    "--device", args.device, "--output", run_dir)
            runs.append(run_dir)
        label_files.append(labels)
    # Target labels are first consumed after every requested prediction artifact exists.
    prediction_manifest, labels = output / "predictions.json", output / "evaluation_labels.json"
    stage(output, "predictions", [p / "learned_predictions.json" for p in runs],
          prediction_manifest, ("-m", "triharmony", "lock-predictions", "--runs", *runs,
                                "--output", prediction_manifest))
    stage(output, "labels", label_files, labels,
          ("-m", "triharmony.manifests", "merge-labels", "--inputs", *label_files, "--output", labels))
    evaluation, tables = output / "evaluation", output / "tables"
    stage(output, "evaluation", [prediction_manifest, labels], evaluation,
          ("-m", "triharmony", "evaluate", "--predictions-manifest", prediction_manifest,
           "--labels", labels, "--output", evaluation))
    stage(output, "tables", [evaluation / "cohort_metrics.csv"], tables,
          ("-m", "triharmony", "tables", "--metrics", evaluation / "cohort_metrics.csv",
           "--output", tables))
    artifacts = [p for p in output.rglob("*") if p.is_file()
                 and "feature_cache" not in p.parts and p.name != "complete.json"]
    completion.write_text(json.dumps({
        "status": "complete", "plan": plan,
        "outputs": [{"path": p.relative_to(output).as_posix(), "sha256": sha(p)}
                    for p in sorted(artifacts)],
    }, indent=2) + "\n")
    print("Main-table outputs:", tables)


if __name__ == "__main__":
    main()
