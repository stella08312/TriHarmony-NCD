from __future__ import annotations
import math
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def _crop_summary_numpy(crop_view: torch.Tensor, n_crops: int = 8) -> np.ndarray:
    x = crop_view.detach().cpu().numpy().astype(np.float32)
    if x.shape[1] % int(n_crops) != 0:
        return x
    b = x.shape[0]
    crop_dim = x.shape[1] // int(n_crops)
    crops = x.reshape(b, int(n_crops), crop_dim)
    return np.concatenate([crops.mean(axis=1), crops.max(axis=1), crops.std(axis=1)], axis=1).astype(np.float32)


def _map_stats_numpy(map_flat: torch.Tensor) -> np.ndarray:
    x = map_flat.detach().cpu().numpy().astype(np.float32)
    q = np.quantile(x, [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99], axis=1).T.astype(np.float32)
    return np.concatenate(
        [
            x.mean(axis=1, keepdims=True),
            x.std(axis=1, keepdims=True),
            x.max(axis=1, keepdims=True),
            q,
        ],
        axis=1,
    ).astype(np.float32)


def _balanced_teacher_feature(*blocks: np.ndarray) -> np.ndarray:
    rows = []
    n_blocks = max(1, len(blocks))
    for block in blocks:
        x = np.nan_to_num(block.astype(np.float32))
        x = StandardScaler().fit_transform(x)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        rows.append((x / np.clip(norm, 1e-6, None) / math.sqrt(float(n_blocks))).astype(np.float32))
    return np.concatenate(rows, axis=1).astype(np.float32)
