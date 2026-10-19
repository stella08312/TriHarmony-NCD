"""File helpers and numerical runtime setup."""
import hashlib
import json
import os
import platform
import random
from pathlib import Path

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

def new_output_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path

def torch_load(path):
    import torch
    # Inputs/checkpoints must be generated locally or obtained from a trusted source.
    # PyTorch 2.0 predates the later weights_only default; keep the call explicit.
    return torch.load(Path(path), map_location="cpu", weights_only=False)

def opaque_sample_key(raw_key):
    return hashlib.sha256(str(raw_key).encode()).hexdigest()

def resolve_relative(base, path):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def prepare_environment(config):
    # historical_seed3 (seeds 7, 21, 3407) pins OpenMP/MKL/OpenBLAS to one thread.
    # The variables are read when numpy/torch load, so this must run before they
    # are imported; the CLI imports numerical modules only after this call.
    previous = {}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        previous[name] = os.environ.get(name)
        if config["runtime"]["profile"] == "historical_seed3":
            os.environ[name] = "1"
        elif config["runtime"]["clear_blas_thread_overrides"]:
            os.environ.pop(name, None)
    return previous

def initialize(config, device_name):
    import numpy as np
    import torch
    from . import config as constants, model as core
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["runtime"]["torch_cpu_threads"])
    constants.SEED = core.SEED = seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
        torch.cuda.manual_seed_all(seed)
    elif device.type != "cpu":
        raise ValueError("Only cpu and cuda devices are supported")
    return device

def runtime_record(config, device):
    import numpy, scipy, sklearn, torch
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "torch_cpu_threads": torch.get_num_threads(),
        "runtime_configuration": config["runtime"],
        "blas_environment": {key: os.environ.get(key) for key in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
