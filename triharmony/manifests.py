"""Create checksum-pinned input manifests without bundling any dataset.

Only this setup command reads dataset directory names. The target input manifest
contains opaque IDs; class names are written to a separate evaluator-only file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAP_EXTENSIONS = (".npy", ".png", ".jpg", ".jpeg", ".bmp")
EXCLUDED_TARGET_FOLDERS = {"good", "normal", "combined", "mt_free"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_key(path, root):
    # Sample identity follows the dataset layout, even for symlinked files.
    relative = path.absolute().relative_to(root.absolute()).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]


def _images(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def _write(path, data):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Choose a new manifest path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _relative(file, destination):
    return os.path.relpath(file.resolve(), destination.resolve().parent)


def source(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    classes = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    if len(classes) != 4:
        raise ValueError(f"Expected the four processed AeBAD-S anomaly classes, found {classes}")
    rows = []
    for index, name in enumerate(classes):
        for image in _images(root / "images" / name):
            mask = root / "masks" / name / image.name
            if not mask.is_file():
                raise FileNotFoundError(mask)
            rows.append({
                "sample_id": stable_key(image, root / "images"),
                "image": _relative(image, output), "image_sha256": sha256(image),
                "mask": _relative(mask, output), "mask_sha256": sha256(mask),
                "class_index": index,
            })
    if len(rows) != 1278:
        raise ValueError(f"Main-table source cohort requires 1278 processed crops, found {len(rows)}")
    _write(output, {"schema_version": 1, "kind": "source", "sample_count": len(rows),
                    "class_names": classes, "rows": rows})
    print(json.dumps({"manifest": str(output), "samples": len(rows)}))


def _find_map(root, dataset, category, image):
    folder = root / category / image.parent.name
    if dataset == "mtd" and not folder.exists():
        folder = root / image.parent.name
    matches = [folder / (image.stem + suffix) for suffix in MAP_EXTENSIONS]
    found = [p for p in matches if p.is_file()]
    if not found:
        raise FileNotFoundError(f"No map for {image.name} under {folder}")
    # Raw NPY has precedence, matching the original CPR/MTD protocol.
    return found[0]


def target(args):
    root, maps = Path(args.root).resolve(), Path(args.maps).resolve()
    output, labels_file = Path(args.output).resolve(), Path(args.labels).resolve()
    if output == labels_file:
        raise ValueError("Input and evaluation-label files must be separate")
    registry = json.loads((Path(__file__).resolve().parent.parent / "configs/main_cohorts.json").read_text())
    cohort = registry[args.dataset][args.category]
    test_root = root / args.category / "test" if args.dataset == "mvtec" else root / "test"
    rows, labels = [], {}
    for folder in sorted(p for p in test_root.iterdir() if p.is_dir()):
        if folder.name.lower() in EXCLUDED_TARGET_FOLDERS:
            continue
        for image in _images(folder):
            map_path = _find_map(maps, args.dataset, args.category, image)
            if args.protocol == "table2" and args.dataset == "mtd" and map_path.suffix != ".npy":
                raise ValueError("Paper Table 2 MTD requires original float CPR NPY maps; PNG changes the input")
            key = stable_key(image, test_root)
            rows.append({
                "sample_id": key, "image": _relative(image, output),
                "image_sha256": sha256(image), "map": _relative(map_path, output),
                "map_sha256": sha256(map_path),
            })
            labels[hashlib.sha256(key.encode("utf-8")).hexdigest()] = folder.name
    if len(rows) != cohort["sample_count"]:
        raise ValueError(f"Expected {cohort['sample_count']} images, found {len(rows)}")
    if len(set(labels.values())) != cohort["known_k"]:
        raise ValueError("Folder layout disagrees with the fixed protocol; do not infer or retune K")
    if output.exists() or labels_file.exists():
        raise FileExistsError("Choose new output and label paths")
    _write(output, {
        "schema_version": 1, "kind": "target", "protocol": args.protocol,
        "dataset": args.dataset, "category": args.category,
        "sample_count": len(rows), "known_k": cohort["known_k"],
        "contains_target_gt": False, "contains_target_normal": False,
        "rows": rows,
    })
    _write(labels_file, {"schema_version": 1, "labels": labels})
    print(json.dumps({"manifest": str(output), "evaluation_labels": str(labels_file),
                      "samples": len(rows), "known_k": cohort["known_k"]}))


def merge_labels(args):
    merged = {}
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in data["labels"].items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Conflicting evaluation labels for {key}")
            merged[key] = value
    _write(args.output, {"schema_version": 1, "labels": merged})
    print(json.dumps({"labels": len(merged), "output": args.output}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("source")
    p.add_argument("--root", required=True, help="Processed AeBAD_crop directory containing images/ and masks/")
    p.add_argument("--output", required=True)
    p.set_defaults(func=source)
    p = sub.add_parser("target")
    p.add_argument("--root", required=True, help="MVTec or preprocessed MTD root")
    p.add_argument("--maps", required=True)
    p.add_argument("--protocol", choices=["table1", "table2"], required=True)
    p.add_argument("--dataset", choices=["mvtec", "mtd"], required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--labels", required=True, help="Separate evaluator-only label file")
    p.set_defaults(func=target)
    p = sub.add_parser("merge-labels")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=merge_labels)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
