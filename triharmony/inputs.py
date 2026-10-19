"""GT-free source and target tensor contracts."""
import re
from .utils import torch_load, opaque_sample_key

TARGET_FLAGS = (
    "contains_target_gt", "contains_target_normal", "target_labels_used",
    "source_class_identity_used_for_target_clustering",
)
FORBIDDEN_TARGET_FIELDS = {
    "label", "labels", "target_label", "target_labels", "class_id", "class_ids",
    "ground_truth", "gt", "gt_mask", "gt_masks", "mask", "masks", "y",
    "prediction", "predictions", "pseudo_labels", "target_class",
}
REQUIRED_EVIDENCE = {
    "micro", "shape", "top", "source_normal", "eventlet", "crop_summary",
    "source_memory_contrast", "crop_regions",
}

def _reject_gt_fields(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_TARGET_FIELDS:
                raise ValueError("Forbidden target label/assignment field: %s" % key)
            _reject_gt_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                _reject_gt_fields(item)

def _tensor(value, name, shape=None):
    import torch
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise ValueError("%s must be a finite tensor" % name)
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError("Unexpected %s shape: %r" % (name, tuple(value.shape)))
    return value.detach().cpu()

def _keys(payload, minimum):
    keys = payload.get("keys")
    if not isinstance(keys, list) or len(keys) < minimum:
        raise ValueError("Missing or insufficient sample keys")
    if any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{24}", key)
           for key in keys):
        raise ValueError("keys must be opaque 24-character hexadecimal identities")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate sample keys")
    return keys

def load_source(path):
    import torch
    value = torch_load(path)
    if value.get("contains_target_gt") is not False:
        raise ValueError("Source cache must declare contains_target_gt=false")
    keys = _keys(value, 4)
    dense = _tensor(value.get("dense"), "dense", (len(keys), 17038)).float()
    sparse = _tensor(value.get("sparse"), "sparse", (len(keys), 11627)).float()
    labels = _tensor(value.get("class_id"), "class_id", (len(keys),))
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Source class_id must use an integer dtype")
    if set(labels.tolist()) != {0, 1, 2, 3}:
        raise ValueError("Source labels must contain exactly the four AeBAD classes")
    return {"dense": dense, "sparse": sparse, "class_id": labels.long(),
            "keys": keys, "contains_target_gt": False}

def load_target(path):
    import torch
    value = torch_load(path)
    _reject_gt_fields(value)
    for flag in TARGET_FLAGS:
        if value.get(flag) is not False:
            raise ValueError("Target cache must declare %s=false" % flag)
    if value.get("frozen_feature_cache") is not True:
        raise ValueError("Target evidence must come from the frozen feature extractor")
    if value.get("target_gt_fields") != []:
        raise ValueError("Target cache must declare empty target_gt_fields")
    keys = _keys(value, 25)
    known_k = value.get("known_k")
    if type(known_k) is not int or not 1 <= known_k <= len(keys):
        raise ValueError("Supply an a-priori known_k in [1, N]")
    if value.get("protocol") not in ("table1", "table2"):
        raise ValueError("protocol must be table1 or table2")
    if value.get("dataset") not in ("mvtec", "mtd"):
        raise ValueError("dataset must be mvtec or mtd")
    if not isinstance(value.get("category"), str) or not value["category"]:
        raise ValueError("Missing category")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict) or not REQUIRED_EVIDENCE <= set(evidence):
        raise ValueError("Missing one or more required evidence blocks")
    clean = {}
    for name, block in evidence.items():
        block = _tensor(block, name)
        if block.ndim != 2 or block.shape[0] != len(keys) or block.shape[1] < 1:
            raise ValueError("Evidence must be a sample-aligned nonempty matrix: %s" % name)
        clean[name] = block.float()
    if clean["crop_regions"].shape[1] % 8 or clean["crop_regions"].shape[1] < 8 * 192:
        raise ValueError("crop_regions must contain eight components of at least 192 dimensions")
    support = _tensor(value.get("map_density"), "map_density", (len(keys),)).float()
    if not torch.all((support >= 0) & (support <= 1)):
        raise ValueError("Map density must lie in [0, 1]")
    return {"protocol": value["protocol"], "dataset": value["dataset"],
            "category": value["category"], "keys": [opaque_sample_key(k) for k in keys],
            "known_k": known_k, "evidence": clean, "anomaly_support": support,
            "contains_target_gt": False}
