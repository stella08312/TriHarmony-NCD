"""Build main-table inputs from images, maps, source masks and local DINOv3 weights.

No target labels are read here. All generated arrays and caches belong to the
user-selected output directory. Downloaded data and backbone weights remain
external to the code release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .features import common, data_views, descriptors, mvem_views
from .features.eventlet_teacher import build_eventlet_teacher

EVIDENCE_NAMES = (
    "top", "source_normal", "shape", "morphology", "micro", "eventlet",
    "ret_subspace", "crop_summary", "crop_regions", "source_memory_contrast",
    "source_memory_profile", "map_stats_224", "map_stats_448",
)
FORBIDDEN_TARGET_FIELDS = {
    "label", "labels", "class_id", "class_index", "defect_type", "is_normal",
    "ground_truth", "target_labels", "source_label",
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(base, path):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _load_manifest(path, kind):
    path = Path(path).resolve()
    manifest = _json(path)
    if manifest.get("kind") != kind:
        raise ValueError(f"Expected a {kind} manifest: {path}")
    rows = manifest.get("rows", [])
    if not rows or manifest.get("sample_count") != len(rows):
        raise ValueError("Manifest row count is empty or inconsistent")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    if kind == "target":
        if manifest.get("contains_target_gt") is not False:
            raise ValueError("Target manifest must explicitly exclude target ground truth")
        if manifest.get("contains_target_normal") is not False:
            raise ValueError("The main-table profile uses anomalous images only")
        if manifest.get("protocol") not in ("table1", "table2"):
            raise ValueError("Unknown map protocol")
        if not 1 <= int(manifest.get("known_k", 0)) <= len(rows):
            raise ValueError("Known K must be supplied as a positive protocol input")
        for record in [manifest, *rows]:
            if FORBIDDEN_TARGET_FIELDS.intersection(record):
                raise ValueError("Target class information is not an input to preparation")
    for row in rows:
        for field in ("image", "mask" if kind == "source" else "map"):
            file = _resolve(path.parent, row[field])
            if not file.is_file():
                raise FileNotFoundError(file)
            expected = row.get(field + "_sha256")
            if expected is None or sha256(file) != expected:
                raise ValueError(f"Missing or mismatched {field} checksum for {row['sample_id']}")
            row[field] = str(file)
    return manifest, path


def _new_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Preserve the existing artifact; choose a new output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _setup(args, manifest, manifest_path, output):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Download the documented DINOv3 weights and pass --checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a smoke test")
    common.seed_everything(42)
    common.DINOv3PatchExtractor.checkpoint_path = checkpoint
    config = SimpleNamespace(
        protocol=manifest.get("protocol", "table1"),
        dataset=manifest.get("dataset", "source"),
        category=manifest.get("category", "AeBAD"),
        image_size=224, hr_image_size=448, grid224=16, grid448=32,
        token_dim=768, feature_backend="dinov3_vitb16",
        feature_batch_size=int(args.batch_size), seed=42,
        n_types=int(manifest.get("known_k", 4)),
        _target_n_types=int(manifest.get("known_k", 4)),
    )
    # Isolate cache generations by both inputs and weights. A changed checkpoint
    # or changed image/map can never silently reuse the preceding cache.
    identity = hashlib.sha256(
        (sha256(manifest_path) + sha256(checkpoint) + "triharmony-features-v1").encode()
    ).hexdigest()[:24]
    root = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else output.parent / "feature_cache"
    cache = common.FeatureCache(
        root / identity, device, 224, 448, 768, "dinov3_vitb16",
        int(args.batch_size), 42,
    )
    # Retain the historical one-image extraction order; it is also memory-bounded.
    mvem_views._tokens_for_images_batch = lambda c, paths, size, side: {
        p: c.tokens_for_image(p, size, side) for p in dict.fromkeys(Path(x) for x in paths)
    }
    metadata = {
        "manifest_sha256": sha256(manifest_path),
        "backbone_sha256": sha256(checkpoint),
        "backbone": "vit_base_patch16_dinov3.lvd1689m",
        "device": str(device), "torch_version": str(torch.__version__),
        "feature_seed": 42, "target_labels_used": False,
    }
    return device, config, cache, metadata


def _validate_tensors(values, n):
    for name, value in values.items():
        if value.ndim != 2 or value.shape[0] != n or not torch.isfinite(value).all():
            raise ValueError(f"Invalid feature tensor {name}: {tuple(value.shape)}")


def _write_pt(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_metadata(path, metadata, outputs):
    metadata = dict(metadata, status="complete", outputs=[
        {"name": p.name, "sha256": sha256(p)} for p in outputs
    ])
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def prepare_source_inputs(args):
    manifest, manifest_path = _load_manifest(args.manifest, "source")
    output = _new_output(args.output)
    memory_path = output.with_name(output.stem + "_memory.pt")
    if memory_path.exists():
        raise FileExistsError(memory_path)
    device, config, cache, metadata = _setup(args, manifest, manifest_path, output)
    names = manifest["class_names"]
    if len(names) != 4 or names != sorted(set(names)):
        raise ValueError("Expected four sorted AeBAD source classes")
    rows = manifest["rows"]
    samples = []
    for row in rows:
        index = int(row["class_index"])
        if not 0 <= index < len(names):
            raise ValueError("Invalid source class index")
        samples.append(common.Sample(
            Path(row["image"]), Path(row["mask"]), names[index],
            "AeBAD", "aebad", "source", names[index],
        ))
    batch = cache.build_batch(samples, config, normal_images=None)
    memories = {}
    for resolution in ("224", "448"):
        for polarity in ("normal", "anomaly"):
            value = mvem_views._source_normal_reference_patch_bank(
                batch, config, resolution=resolution,
                set_summary=(resolution == "224" and polarity == "normal"),
                polarity=polarity,
            )
            memories[f"_source_{polarity}_bank{resolution}"] = value
            setattr(config, f"_source_{polarity}_bank{resolution}", value)
    normal224 = memories["_source_normal_bank224"]
    normal448 = memories["_source_normal_bank448"]
    # Historical source descriptors use the source masks directly. The source
    # memories are used for target evidence, not fed back into these descriptors.
    base = data_views.build_view_bundle(batch, None)
    teacher = build_eventlet_teacher(
        batch.tokens224, batch.tokens448, batch.map224, batch.map448,
        batch.coords224, batch.coords448, None, "table1",
        memory_bank448=None,
    )
    crop = mvem_views._actual_crop_image_region_view(batch, cache, config, is_target=False)
    dense = descriptors._balanced_teacher_feature(
        base.views[7].detach().cpu().numpy(), base.views[3].detach().cpu().numpy(),
    )
    sparse = descriptors._balanced_teacher_feature(
        base.views[0].detach().cpu().numpy(), base.views[2].detach().cpu().numpy(),
        teacher["eventlet_raw"].detach().cpu().numpy(),
        base.views[3].detach().cpu().numpy(), descriptors._crop_summary_numpy(crop),
    )
    values = {"dense": torch.from_numpy(dense), "sparse": torch.from_numpy(sparse)}
    _validate_tensors(values, len(rows))
    if dense.shape[1] != 17038 or sparse.shape[1] != 11627:
        raise ValueError("The source descriptor dimensions differ from the paper")
    payload = dict(
        values, keys=[row["sample_id"] for row in rows],
        class_id=torch.tensor([int(row["class_index"]) for row in rows], dtype=torch.long),
        source_class_names=names, contains_target_gt=False,
        contains_normal_input=False, source_mask_used=True, provenance=metadata,
    )
    memory = {
        key: value.detach().cpu().float() for key, value in memories.items()
    }
    memory.update(source_keys=payload["keys"], source_class_names=names,
                  contains_target_gt=False, provenance=metadata)
    _write_pt(output, payload)
    _write_pt(memory_path, memory)
    _write_metadata(output.with_suffix(".json"), metadata, [output, memory_path])
    result = {"source": str(output), "source_memory": str(memory_path), "samples": len(rows)}
    print(json.dumps(result, indent=2))
    return result


def _target_batch(rows, cache):
    t224, t448, m224, m448, metadata = [], [], [], [], []
    for row in rows:
        path = Path(row["image"])
        amap = common.read_map(Path(row["map"]), 224)
        crop, crop_map, box = cache.roi_crop(path, amap, row["sample_id"])
        t224.append(cache.tokens_for_image(path, 224, 16))
        t448.append(cache.tokens_for_image(crop, 448, 32))
        m224.append(F.interpolate(amap[None], size=(16, 16), mode="area")[0, 0].flatten())
        m448.append(F.interpolate(crop_map[None], size=(32, 32), mode="area")[0, 0].flatten())
        metadata.append({"image": str(path), "map": row["map"],
                         "roi_image": str(crop), "roi_bbox_xyxy_norm": box})
    return common.Batch(
        F.normalize(torch.stack(t224), dim=-1),
        F.normalize(torch.stack(t448), dim=-1),
        torch.stack(m224).to(cache.device), torch.stack(m448).to(cache.device),
        cache.coords(16), cache.coords(32), [""] * len(rows), None, None, metadata,
    )


def prepare_target_inputs(args):
    manifest, manifest_path = _load_manifest(args.manifest, "target")
    output = _new_output(args.output)
    device, config, cache, metadata = _setup(args, manifest, manifest_path, output)
    try:
        memory = torch.load(args.source_memory, map_location="cpu", weights_only=True)
    except TypeError:
        memory = torch.load(args.source_memory, map_location="cpu")
    if memory.get("contains_target_gt") is not False:
        raise ValueError("Unsafe source memory")
    if memory.get("provenance", {}).get("backbone_sha256") != metadata["backbone_sha256"]:
        raise ValueError("Source memory and target preparation must use the same backbone weights")
    for resolution in ("224", "448"):
        for polarity in ("normal", "anomaly"):
            name = f"_source_{polarity}_bank{resolution}"
            value = memory[name]
            if value.ndim != 2 or value.shape[1] != 768 or not torch.isfinite(value).all():
                raise ValueError("Invalid source memory tensor")
            setattr(config, name, value.to(device))
    rows = manifest["rows"]
    batch = _target_batch(rows, cache)
    bundle = mvem_views._build_mvem_bundle(
        batch, config._source_normal_bank224, config, is_target=True, cache=cache,
    )
    def clean(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().float()
        else:
            value = torch.from_numpy(np.asarray(value, dtype=np.float32))
        return torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
    crop_summary = descriptors._crop_summary_numpy(bundle.views[12])
    evidence = {
        "top": clean(bundle.views[0]), "source_normal": clean(bundle.views[2]),
        "shape": clean(bundle.views[3]), "morphology": clean(bundle.views[4]),
        "micro": clean(bundle.views[7]), "eventlet": clean(bundle.views[8]),
        "ret_subspace": clean(bundle.views[10]), "crop_summary": clean(crop_summary),
        "crop_regions": clean(bundle.views[12]),
        "source_memory_contrast": clean(bundle.care_contrast_view),
        "source_memory_profile": clean(bundle.source_memory_profile_view),
        "map_stats_224": clean(descriptors._map_stats_numpy(bundle.map224)),
        "map_stats_448": clean(descriptors._map_stats_numpy(bundle.map448)),
    }
    assert tuple(evidence) == EVIDENCE_NAMES
    _validate_tensors(evidence, len(rows))
    dense = clean(descriptors._balanced_teacher_feature(
        evidence["micro"].numpy(), evidence["shape"].numpy()))
    sparse = clean(descriptors._balanced_teacher_feature(
        evidence["top"].numpy(), evidence["source_normal"].numpy(),
        evidence["eventlet"].numpy(), evidence["shape"].numpy(), crop_summary))
    metadata["source_memory_sha256"] = sha256(args.source_memory)
    payload = {
        "keys": [row["sample_id"] for row in rows],
        "known_k": int(manifest["known_k"]),
        "protocol": manifest["protocol"], "dataset": manifest["dataset"],
        "category": manifest["category"], "p0_dense": dense, "p0_sparse": sparse,
        "evidence": evidence,
        "map_density": bundle.map224.detach().cpu().float().mean(dim=1),
        "map_density_224": bundle.map224.detach().cpu().float().mean(dim=1),
        "map_density_448": bundle.map448.detach().cpu().float().mean(dim=1),
        "contains_target_gt": False, "contains_target_normal": False,
        "target_gt_fields": [], "target_labels_used": False,
        "source_class_identity_used_for_target_clustering": False,
        "source_role": "source-mask normal/anomaly evidence memory only",
        "frozen_feature_cache": True, "provenance": metadata,
    }
    _write_pt(output, payload)
    _write_metadata(output.with_suffix(".json"), metadata, [output])
    result = {"target": str(output), "samples": len(rows), "known_k": payload["known_k"]}
    print(json.dumps(result, indent=2))
    return result
