"""Fixed method settings and configuration loading."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Tuple

METHOD_NAME = "CAPE-P0 TRIADIC-HARMONY NCD"
IMPLEMENTATION_REVISION = "triadic_harmony_v14_learned_only_r2"

SEED = 42
ADDITIONAL_NUMERIC_SEEDS: Tuple[int, ...] = ()
EPOCHS = 40
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
SOURCE_BATCH_SIZE = 32
TARGET_BATCH_SIZE = 32
BATCH_SIZE = 64
SOURCE_CLASSES = 4

EVIDENCE_FAMILIES: Tuple[Tuple[str, ...], ...] = (
    ("micro", "shape"),
    ("top", "source_normal", "eventlet", "shape", "crop_summary"),
    ("source_memory_contrast", "top", "eventlet"),
)
EXPERT_SPECS = (
    ("micro_corresponding_band", 0, "corresponding", 0.50, 10, 27),
    ("semantic_hungarian_band", 1, "hungarian", 0.25, 10, 27),
    ("micro_chamfer_band_dense", 0, "chamfer", 0.85, 0, 27),
    ("source_chamfer_single", 2, "chamfer", 0.50, 20, 9),
)
CHANNELS = len(EXPERT_SPECS)
SPECTRAL_BANDS = 3
COMPONENT_PROJECTION_DIM = 64
LOCAL_SCALE_NEIGHBORS = 24
SOURCE_SPLIT_PCA_DIM = 8
SOURCE_SPLIT_MINIMUM_CROSS_EVIDENCE_ARI = 0.95
SOURCE_SPLIT_MINIMUM_CHILD = BATCH_SIZE // 4
VIEW_GROUP_SIZE = 9
SUPPORT_COMPONENT_COUNT = 8
SUPPORT_ACTIVE_COMPONENT_REFERENCE = 2
SUPPORT_CENTER = SUPPORT_ACTIVE_COMPONENT_REFERENCE / SUPPORT_COMPONENT_COUNT
SUPPORT_LOGIT_SCALE = float(SUPPORT_COMPONENT_COUNT)
SUPPORT_EPSILON = 1.0e-6

LOSS_WEIGHTS = {
    "source_supervised_relation": 1.0,
    "target_scale_normalized_heldout_prediction": 1.0,
    "target_deletion_persistence_transport": 1.0,
    "target_known_k_spectral_capacity": 1.0,
}
SOURCE_TARGET_PARALLEL_FRACTION = 0.75
SOURCE_ORTHOGONAL_RETENTION = 0.25

FIXED = {
    "schema_version": 1, "method": "TriHarmony-Full-v14", "epochs": 40,
    "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.001,
    "source_batch_size": 32, "target_batch_size": 32,
    "loss_weights": {
        "source_supervised_relation": 1.0,
        "target_scale_normalized_heldout_prediction": 1.0,
        "target_deletion_persistence_transport": 1.0,
        "target_known_k_spectral_capacity": 1.0,
    },
    "source_target_parallel_fraction": 0.75,
    "source_orthogonal_retention": 0.25,
    "source_split_minimum_cross_evidence_ari": 0.95,
    "cohort_policy": "anomalous_only", "known_k_policy": "supplied_a_priori",
    "evaluation_labels_allowed_in_training": False,
}
DEFAULT = dict(FIXED, seed=42, runtime={
    "profile": "historical_v14", "torch_cpu_threads": 96,
    "clear_blas_thread_overrides": True, "deterministic_cudnn": True,
    "allow_tf32": False,
})

def load_config(path=None, seed=None, cpu_threads=None):
    config = json.loads(Path(path).read_text()) if path else copy.deepcopy(DEFAULT)
    if set(config) != set(DEFAULT):
        raise ValueError("Configuration keys must match the supplied main config")
    for key, expected in FIXED.items():
        if config[key] != expected:
            raise ValueError("The main-table configuration fixes %s=%r" % (key, expected))
    if seed is not None:
        config["seed"] = seed
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    runtime = config["runtime"]
    if set(runtime) != set(DEFAULT["runtime"]):
        raise ValueError("Unknown or missing runtime configuration keys")
    if runtime["profile"] in ("historical_v14", "historical_seed3"):
        # Seed 42 was run in the original 96-thread environment. The later
        # seed-7/21/3407 runs fixed both torch and BLAS/OpenMP to one thread.
        # This reproduces archived execution conditions; equations are identical.
        seed_study = config["seed"] in (7, 21, 3407)
        runtime["profile"] = "historical_seed3" if seed_study else "historical_v14"
        runtime["torch_cpu_threads"] = 1 if seed_study else 96
        runtime["clear_blas_thread_overrides"] = not seed_study
    if cpu_threads is not None:
        runtime["torch_cpu_threads"] = cpu_threads
        runtime["profile"] = "custom_cpu_threads"
        runtime["clear_blas_thread_overrides"] = True
    if type(runtime["torch_cpu_threads"]) is not int or runtime["torch_cpu_threads"] < 1:
        raise ValueError("torch_cpu_threads must be positive")
    if runtime["profile"] not in ("historical_v14", "historical_seed3", "custom_cpu_threads"):
        raise ValueError("Unknown runtime profile")
    if runtime["deterministic_cudnn"] is not True:
        raise ValueError("deterministic_cudnn must remain true")
    expected_clear = runtime["profile"] != "historical_seed3"
    if runtime["clear_blas_thread_overrides"] is not expected_clear:
        raise ValueError("BLAS thread policy must match the selected runtime profile")
    if runtime["allow_tf32"] is not False:
        raise ValueError("TF32 is disabled by the reference runtime")
    return config
