from __future__ import annotations
import hashlib, math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from . import data_views as base
from .eventlet_teacher import build_eventlet_teacher

def _source_normal_reference_patch_bank(
    batch: base.Batch,
    args,
    resolution: str = "224",
    set_summary: bool = False,
    summary_attr: str = "_source_normal_reference_summary",
    polarity: str = "normal",
) -> Optional[torch.Tensor]:
    """Build a class-balanced source memory from labeled source masks only."""

    if batch.source_label is None:
        return None
    if str(resolution) == "448":
        tokens = batch.tokens448
        amap = batch.map448
    else:
        tokens = batch.tokens224
        amap = batch.map224
    if tokens.numel() == 0 or amap.numel() == 0:
        return None

    n_patches = int(amap.shape[1])
    low_fraction = 0.35
    min_tokens = 32
    max_tokens = 64
    per_class_max = 1024
    max_total = 8192

    k = int(round(float(n_patches) * max(0.0, min(1.0, low_fraction))))
    k = max(1, min(n_patches, k))
    if min_tokens > 0:
        k = max(k, min(min_tokens, n_patches))
    if max_tokens > 0:
        k = min(k, max_tokens)

    clean_map = torch.nan_to_num(amap.float(), nan=1.0, posinf=1.0, neginf=0.0)
    if str(polarity) == "anomaly":
        _, select_idx = torch.topk(clean_map, k=k, dim=1)
    else:
        _, select_idx = torch.topk(-clean_map, k=k, dim=1)
    selected = tokens.gather(1, select_idx[:, :, None].expand(-1, -1, tokens.shape[-1])).reshape(-1, tokens.shape[-1])
    selected_scores = clean_map.gather(1, select_idx).reshape(-1)
    sample_labels = batch.source_label.to(device=tokens.device, dtype=torch.long).reshape(-1)
    token_labels = sample_labels[:, None].expand(-1, k).reshape(-1)

    keep_parts: List[torch.Tensor] = []
    for cls in sorted(int(x) for x in token_labels.detach().cpu().unique().tolist()):
        idx = (token_labels == int(cls)).nonzero(as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        # Prefer the cleanest source-normal pixels first, then subsample evenly
        # if one source class contributes too many tokens.
        order = selected_scores[idx].argsort(descending=str(polarity) == "anomaly")
        idx = idx[order]
        if per_class_max > 0 and idx.numel() > per_class_max:
            step = max(1, int(math.ceil(float(idx.numel()) / float(per_class_max))))
            idx = idx[::step][:per_class_max]
        keep_parts.append(idx)

    if keep_parts:
        keep_idx = torch.cat(keep_parts, dim=0)
        bank = selected.index_select(0, keep_idx)
        kept_scores = selected_scores.index_select(0, keep_idx)
        kept_labels = token_labels.index_select(0, keep_idx)
    else:
        bank = selected
        kept_scores = selected_scores
        kept_labels = token_labels

    bank = F.normalize(torch.nan_to_num(bank, nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    if max_total > 0 and bank.shape[0] > max_total:
        step = max(1, int(math.ceil(float(bank.shape[0]) / float(max_total))))
        bank = bank[::step][:max_total]
        kept_scores = kept_scores[::step][:max_total]
        kept_labels = kept_labels[::step][:max_total]

    if set_summary:
        usage = torch.bincount(kept_labels.detach().cpu(), minlength=int(sample_labels.max().detach().cpu()) + 1).tolist()
        setattr(
            args,
            summary_attr,
            {
                "source_normal_ref_resolution": str(resolution),
                "source_normal_ref_polarity": str(polarity),
                "source_normal_ref_low_fraction": float(k) / float(n_patches),
                "source_normal_ref_tokens": int(bank.shape[0]),
                "source_normal_ref_classes": int(len([x for x in usage if int(x) > 0])),
                "source_normal_ref_per_class_usage": {str(i): int(v) for i, v in enumerate(usage)},
                "source_normal_ref_mask_score_mean": float(kept_scores.detach().mean().cpu()) if kept_scores.numel() else 0.0,
                "source_normal_ref_mask_score_max": float(kept_scores.detach().max().cpu()) if kept_scores.numel() else 0.0,
                "source_normal_ref_per_class_max_tokens": int(per_class_max),
                "source_normal_ref_max_total_tokens": int(max_total),
            },
        )
    return bank.contiguous()


def _ret448_view(batch, normal_bank: Optional[torch.Tensor]) -> torch.Tensor:
    bank = normal_bank if normal_bank is not None and normal_bank.shape[-1] == batch.tokens448.shape[-1] else None
    return torch.nan_to_num(
        base._ret_eventlet_signature(batch.tokens448, batch.map448, batch.coords448, bank, 24, 64, compact=True),
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    )


def _patch_contrast_view(tokens: torch.Tensor, amap: torch.Tensor, low_fraction: float, high_fraction: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    clean_map = torch.nan_to_num(amap.float(), nan=0.0, posinf=0.0, neginf=0.0)
    n_patches = int(clean_map.shape[1])
    low_k = max(1, min(n_patches, int(round(float(n_patches) * max(0.0, min(1.0, float(low_fraction)))))))
    high_k = max(1, min(n_patches, int(round(float(n_patches) * max(0.0, min(1.0, float(high_fraction)))))))

    map_min = clean_map.min(dim=1, keepdim=True).values
    map_max = clean_map.max(dim=1, keepdim=True).values
    map_span = (map_max - map_min).clamp_min(1e-6)
    norm_map = ((clean_map - map_min) / map_span).clamp(0.0, 1.0)

    _, low_idx = torch.topk(-norm_map, k=low_k, dim=1)
    _, high_idx = torch.topk(norm_map, k=high_k, dim=1)
    dim = int(tokens.shape[-1])
    low_tokens = tokens.gather(1, low_idx[:, :, None].expand(-1, -1, dim)).mean(dim=1)
    high_tokens = tokens.gather(1, high_idx[:, :, None].expand(-1, -1, dim)).mean(dim=1)
    low_tokens = F.normalize(low_tokens, dim=1)
    high_tokens = F.normalize(high_tokens, dim=1)
    delta = F.normalize(high_tokens - low_tokens, dim=1)
    abs_delta = F.normalize((high_tokens - low_tokens).abs(), dim=1)

    high_score = norm_map.gather(1, high_idx).mean(dim=1, keepdim=True)
    low_score = norm_map.gather(1, low_idx).mean(dim=1, keepdim=True)
    gap = (high_score - low_score).clamp(0.0, 1.0)
    shifted = clean_map - clean_map.min(dim=1, keepdim=True).values
    mass = shifted.sum(dim=1, keepdim=True)
    uniform = torch.full_like(shifted, 1.0 / float(max(1, n_patches)))
    prob = torch.where(mass > 1e-8, shifted / mass.clamp_min(1e-8), uniform)
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / math.log(float(max(2, n_patches)))
    sharpness = (1.0 - entropy).clamp(0.0, 1.0)
    std = norm_map.std(dim=1, keepdim=True).clamp(0.0, 1.0)
    token_gap = (1.0 - (high_tokens * low_tokens).sum(dim=1, keepdim=True)).clamp(0.0, 2.0) * 0.5
    stats = torch.cat(
        [
            high_score,
            low_score,
            gap,
            entropy.clamp(0.0, 1.0),
            sharpness,
            std,
            token_gap,
            torch.full_like(gap, float(low_k) / float(n_patches)),
            torch.full_like(gap, float(high_k) / float(n_patches)),
        ],
        dim=1,
    )
    view = torch.cat([low_tokens, high_tokens, delta, abs_delta, stats], dim=1)
    view = torch.nan_to_num(view, nan=0.0, posinf=10.0, neginf=-10.0)
    summary = {
        "low_fraction_selected": float(low_k) / float(n_patches),
        "high_fraction_selected": float(high_k) / float(n_patches),
        "map_entropy_mean": float(entropy.detach().mean().cpu()),
        "map_sharpness_mean": float(sharpness.detach().mean().cpu()),
        "map_gap_mean": float(gap.detach().mean().cpu()),
        "token_gap_mean": float(token_gap.detach().mean().cpu()),
    }
    return F.normalize(view, dim=1), summary


def _select_low_high_pooled(tokens: torch.Tensor, amap: torch.Tensor, low_fraction: float, high_fraction: float) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    clean_map = torch.nan_to_num(amap.float(), nan=0.0, posinf=0.0, neginf=0.0)
    n_patches = int(clean_map.shape[1])
    low_k = max(1, min(n_patches, int(round(float(n_patches) * max(0.0, min(1.0, float(low_fraction)))))))
    high_k = max(1, min(n_patches, int(round(float(n_patches) * max(0.0, min(1.0, float(high_fraction)))))))
    map_min = clean_map.min(dim=1, keepdim=True).values
    map_max = clean_map.max(dim=1, keepdim=True).values
    norm_map = ((clean_map - map_min) / (map_max - map_min).clamp_min(1e-6)).clamp(0.0, 1.0)
    _, low_idx = torch.topk(-norm_map, k=low_k, dim=1)
    _, high_idx = torch.topk(norm_map, k=high_k, dim=1)
    dim = int(tokens.shape[-1])
    low = tokens.gather(1, low_idx[:, :, None].expand(-1, -1, dim)).mean(dim=1)
    high = tokens.gather(1, high_idx[:, :, None].expand(-1, -1, dim)).mean(dim=1)
    return F.normalize(low, dim=1), F.normalize(high, dim=1)


def _bank_nearest(query: torch.Tensor, bank: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bank is None or bank.numel() == 0:
        zeros = torch.zeros_like(query)
        stat = torch.zeros((query.shape[0], 2), device=query.device, dtype=query.dtype)
        return zeros, stat[:, :1], stat
    bank = F.normalize(torch.nan_to_num(bank.to(device=query.device, dtype=query.dtype), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    vals, idxs, topk_vals = [], [], []
    k = min(16, int(bank.shape[0]))
    for start in range(0, query.shape[0], 64):
        sim = query[start : start + 64] @ bank.T
        val, idx = sim.max(dim=1)
        vals.append(val)
        idxs.append(idx)
        topk_vals.append(sim.topk(k=k, dim=1).values.mean(dim=1))
    value = torch.cat(vals, dim=0)
    index = torch.cat(idxs, dim=0)
    nearest = bank.index_select(0, index)
    # mean top-k similarity is more stable than max alone under source-domain shift.
    topk_mean = torch.cat(topk_vals, dim=0)
    stats = torch.stack([value, topk_mean], dim=1)
    return nearest, value.reshape(-1, 1), stats


def _source_memory_profile_view(batch, args, is_target: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    low_fraction = 0.12
    high_fraction = 0.12
    bank_n224 = getattr(args, "_source_normal_bank224", None)
    bank_a224 = getattr(args, "_source_anomaly_bank224", None)
    bank_n448 = getattr(args, "_source_normal_bank448", None)
    bank_a448 = getattr(args, "_source_anomaly_bank448", None)

    low224, high224 = _select_low_high_pooled(batch.tokens224, batch.map224, low_fraction, high_fraction)
    low448, high448 = _select_low_high_pooled(batch.tokens448, batch.map448, low_fraction, high_fraction)

    per_res: List[torch.Tensor] = []
    stats_summary: Dict[str, float] = {"source_memory_profile_active": bool(is_target)}
    for prefix, low, high, normal_bank, anomaly_bank in [
        ("lr", low224, high224, bank_n224, bank_a224),
        ("hr", low448, high448, bank_n448, bank_a448),
    ]:
        if not is_target:
            per_res.append(torch.zeros((low.shape[0], 20), device=low.device, dtype=low.dtype))
            continue
        _, low_n_max, low_n_stats = _bank_nearest(low, normal_bank)
        _, low_a_max, low_a_stats = _bank_nearest(low, anomaly_bank)
        _, high_n_max, high_n_stats = _bank_nearest(high, normal_bank)
        _, high_a_max, high_a_stats = _bank_nearest(high, anomaly_bank)
        low_normality = low_n_max - low_a_max
        high_anomaly = high_a_max - high_n_max
        low_topk_normality = low_n_stats[:, 1:2] - low_a_stats[:, 1:2]
        high_topk_anomaly = high_a_stats[:, 1:2] - high_n_stats[:, 1:2]
        profile = torch.cat(
            [
                low_n_stats,
                low_a_stats,
                high_n_stats,
                high_a_stats,
                low_normality,
                high_anomaly,
                low_topk_normality,
                high_topk_anomaly,
                high_n_max - low_n_max,
                high_a_max - low_a_max,
                high_anomaly - low_normality,
                high_topk_anomaly - low_topk_normality,
                (low_n_max + high_a_max) - (low_a_max + high_n_max),
                torch.sigmoid(8.0 * low_normality),
                torch.sigmoid(8.0 * high_anomaly),
                torch.sigmoid(6.0 * (high_anomaly - low_normality)),
            ],
            dim=1,
        )
        per_res.append(profile)
        stats_summary[f"{prefix}_source_memory_profile_low_normality"] = float(low_normality.detach().mean().cpu())
        stats_summary[f"{prefix}_source_memory_profile_high_anomaly"] = float(high_anomaly.detach().mean().cpu())
    view = torch.cat([per_res[0], per_res[1], per_res[1] - per_res[0]], dim=1)
    view = torch.nan_to_num(view, nan=0.0, posinf=10.0, neginf=-10.0)
    view = (view - view.mean(dim=0, keepdim=True)) / view.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    stats_summary["source_memory_profile_dim"] = int(view.shape[1])
    return F.normalize(view, dim=1), stats_summary


def _source_memory_contrast_view(batch, args, is_target: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    low_fraction = 0.12
    high_fraction = 0.12
    bank_n224 = getattr(args, "_source_normal_bank224", None)
    bank_a224 = getattr(args, "_source_anomaly_bank224", None)
    bank_n448 = getattr(args, "_source_normal_bank448", None)
    bank_a448 = getattr(args, "_source_anomaly_bank448", None)

    low224, high224 = _select_low_high_pooled(batch.tokens224, batch.map224, low_fraction, high_fraction)
    low448, high448 = _select_low_high_pooled(batch.tokens448, batch.map448, low_fraction, high_fraction)

    chunks: List[torch.Tensor] = []
    stats_summary: Dict[str, float] = {"source_memory_contrast_active": bool(is_target)}
    for prefix, low, high, normal_bank, anomaly_bank in [
        ("lr", low224, high224, bank_n224, bank_a224),
        ("hr", low448, high448, bank_n448, bank_a448),
    ]:
        if not is_target:
            dim = int(low.shape[1])
            chunks.extend(
                [
                    torch.zeros_like(low),
                    torch.zeros_like(high),
                    torch.zeros((low.shape[0], 10), device=low.device, dtype=low.dtype),
                ]
            )
            stats_summary[f"{prefix}_source_memory_dim"] = float(dim * 2 + 10)
            continue
        low_n, low_n_max, low_n_stats = _bank_nearest(low, normal_bank)
        low_a, low_a_max, low_a_stats = _bank_nearest(low, anomaly_bank)
        high_n, high_n_max, high_n_stats = _bank_nearest(high, normal_bank)
        high_a, high_a_max, high_a_stats = _bank_nearest(high, anomaly_bank)
        normal_pull = F.normalize(low - low_n, dim=1)
        anomaly_push = F.normalize(high_a - high_n, dim=1)
        polarity_stats = torch.cat(
            [
                low_n_stats,
                low_a_stats,
                high_n_stats,
                high_a_stats,
                (low_n_max - low_a_max),
                (high_a_max - high_n_max),
            ],
            dim=1,
        )
        chunks.extend([normal_pull, anomaly_push, polarity_stats])
        stats_summary[f"{prefix}_low_normal_max"] = float(low_n_max.detach().mean().cpu())
        stats_summary[f"{prefix}_low_anomaly_max"] = float(low_a_max.detach().mean().cpu())
        stats_summary[f"{prefix}_high_normal_max"] = float(high_n_max.detach().mean().cpu())
        stats_summary[f"{prefix}_high_anomaly_max"] = float(high_a_max.detach().mean().cpu())
    view = torch.nan_to_num(torch.cat(chunks, dim=1), nan=0.0, posinf=10.0, neginf=-10.0)
    stats_summary["source_memory_contrast_dim"] = int(view.shape[1])
    return F.normalize(view, dim=1), stats_summary


def _care_contrast_view(batch, args, is_target: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build the fixed 12% low/high multi-resolution CARE evidence view."""
    lowres_view, lowres_summary = _patch_contrast_view(
        batch.tokens224, batch.map224, 0.12, 0.12
    )
    highres_view, highres_summary = _patch_contrast_view(
        batch.tokens448, batch.map448, 0.12, 0.12
    )
    ret448 = _ret448_view(batch, None)
    desc448 = base._morph_descriptor(batch.map448, batch.coords448)
    source_view, source_summary = _source_memory_contrast_view(batch, args, is_target=is_target)
    view = torch.cat([highres_view, lowres_view, ret448, desc448, source_view], dim=1)
    view = F.normalize(torch.nan_to_num(view, nan=0.0, posinf=10.0, neginf=-10.0), dim=1)
    summary = {
        "mode": "multi_resolution_low_high_source_memory",
        "view_dim": int(view.shape[1]),
        "lowres_map_entropy_mean": lowres_summary["map_entropy_mean"],
        "lowres_map_gap_mean": lowres_summary["map_gap_mean"],
        "lowres_token_gap_mean": lowres_summary["token_gap_mean"],
        "highres_map_entropy_mean": highres_summary["map_entropy_mean"],
        "highres_map_sharpness_mean": highres_summary["map_sharpness_mean"],
        "highres_map_gap_mean": highres_summary["map_gap_mean"],
        "highres_token_gap_mean": highres_summary["token_gap_mean"],
        "low_fraction_selected": 0.12,
        "high_fraction_selected": 0.12,
        **source_summary,
    }
    return view, summary


def _attach_care_evidence(
    bundle: base.ViewBundle,
    batch,
    args,
    is_target: bool,
) -> base.ViewBundle:
    contrast_view, summary = _care_contrast_view(batch, args, is_target=is_target)
    profile_view, profile_summary = _source_memory_profile_view(batch, args, is_target=is_target)
    setattr(bundle, "care_contrast_view", contrast_view)
    setattr(bundle, "source_memory_profile_view", profile_view)
    summary.update(profile_summary)
    summary["view_storage"] = "side_channel"
    if is_target:
        setattr(args, "_care_contrast_summary", summary)
    return bundle


def _ret_pca_pseudotype_views(
    ret_raw: Optional[torch.Tensor],
    confidence: torch.Tensor,
    is_target: bool,
    n_types: int,
    seed: int,
    dim: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(confidence.shape[0])
    k = max(2, int(n_types))
    device = confidence.device
    dtype = confidence.dtype
    if not is_target or ret_raw is None or n <= k:
        return (
            torch.zeros(n, int(dim), device=device, dtype=dtype),
            torch.zeros(n, k, device=device, dtype=dtype),
        )

    x = ret_raw.detach().cpu().numpy().astype(np.float32)
    x = StandardScaler().fit_transform(np.nan_to_num(x))
    eff_dim = min(int(dim), x.shape[0] - 1, x.shape[1])
    if eff_dim > 0:
        z = PCA(n_components=eff_dim, random_state=int(seed)).fit_transform(x)
    else:
        z = np.zeros((x.shape[0], 0), dtype=np.float32)
    if eff_dim < int(dim):
        z = np.concatenate([z.astype(np.float32), np.zeros((x.shape[0], int(dim) - eff_dim), dtype=np.float32)], axis=1)
    z = np.nan_to_num(z.astype(np.float32))
    pred = KMeans(n_clusters=k, n_init=24, max_iter=300, random_state=int(seed)).fit_predict(z).astype(np.int64)
    onehot = np.zeros((n, k), dtype=np.float32)
    onehot[np.arange(n), pred] = 1.0
    conf = confidence.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    onehot = onehot * (0.55 + 0.45 * np.clip(conf, 0.0, 1.0))
    return (
        torch.from_numpy(z).to(device=device, dtype=dtype),
        torch.from_numpy(onehot.astype(np.float32)).to(device=device, dtype=dtype),
    )


def _connected_components_numpy(binary: np.ndarray) -> List[np.ndarray]:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components: List[np.ndarray] = []
    ys, xs = np.where(binary)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        coords = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        mask = np.zeros((h, w), dtype=np.float32)
        yy, xx = zip(*coords)
        mask[np.asarray(yy), np.asarray(xx)] = 1.0
        components.append(mask)
    return components


def _component_window(mask: np.ndarray, pad: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask.astype(np.float32)
    h, w = mask.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    out = np.zeros_like(mask, dtype=np.float32)
    out[y0:y1, x0:x1] = 1.0
    return out


def _component_crop_masks_from_map(amap: torch.Tensor, side: int, n_crops: int = 8) -> torch.Tensor:
    b, n = amap.shape
    if side * side != n:
        return torch.zeros(b, n_crops, n, device=amap.device, dtype=amap.dtype)
    maps = torch.nan_to_num(amap.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    maps_np = maps.cpu().numpy().astype(np.float32)
    out = np.zeros((b, n_crops, n), dtype=np.float32)
    pad = max(1, int(round(float(side) * 0.08)))
    peak_radius = max(1, int(round(float(side) * 0.14)))
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")

    for i in range(b):
        img = maps_np[i].reshape(side, side)
        maxv = float(img.max())
        if maxv <= 1e-8:
            out[i, :, 0] = 1.0
            continue
        img = img / max(maxv, 1e-8)
        candidates: List[Tuple[float, np.ndarray]] = []

        for thr in (0.72, 0.55, 0.38):
            binary = img >= thr
            for comp in _connected_components_numpy(binary):
                area = float(comp.mean())
                if area <= 0.0:
                    continue
                component_mass = float((img * comp).sum() / float(max(1, side * side)))
                component_max = float((img * comp).max())
                component_mean = float((img * comp).sum() / max(1.0, float(comp.sum())))
                window = _component_window(comp, pad=pad)
                window_area = float(window.mean())
                score = (
                    1.10 * component_max
                    + 0.65 * component_mean
                    + 0.45 * np.sqrt(component_mass + 1e-8)
                    + 0.15 * np.sqrt(area + 1e-8)
                    - 0.05 * window_area
                )
                candidates.append((score, window.astype(np.float32)))
                if comp.sum() >= 2:
                    candidates.append((score * 0.92, comp.astype(np.float32)))

        flat_order = np.argsort(img.reshape(-1))[::-1]
        for rank in range(min(n_crops * 4, flat_order.shape[0])):
            idx = int(flat_order[rank])
            cy, cx = divmod(idx, side)
            window = ((np.abs(yy - cy) <= peak_radius) & (np.abs(xx - cx) <= peak_radius)).astype(np.float32)
            peak_score = float(img[cy, cx]) + 0.25 * float((img * window).mean()) - 0.04 * float(window.mean())
            candidates.append((peak_score, window))

        full_mask = np.ones((side, side), dtype=np.float32)
        candidates.append((0.20 + 0.20 * float(img.mean()), full_mask))
        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: List[np.ndarray] = []
        for _score, mask in candidates:
            if mask.sum() <= 0:
                continue
            accept = True
            for prev in selected:
                inter = float((mask * prev).sum())
                union = float(np.maximum(mask, prev).sum())
                if union > 0 and inter / union > 0.82:
                    accept = False
                    break
            if accept:
                selected.append(mask)
            if len(selected) >= n_crops:
                break
        while len(selected) < n_crops:
            selected.append(full_mask)
        out[i] = np.stack(selected[:n_crops], axis=0).reshape(n_crops, n)

    return torch.from_numpy(out).to(device=amap.device, dtype=amap.dtype)


def _boxes_from_component_masks(amap: torch.Tensor, side: int, n_crops: int = 8, unique_crops: int = 4) -> torch.Tensor:
    masks = _component_crop_masks_from_map(amap, side, n_crops=max(1, min(int(unique_crops), int(n_crops))))
    b = int(masks.shape[0])
    boxes = torch.zeros(b, masks.shape[1], 4, device=amap.device, dtype=amap.dtype)
    for i in range(b):
        for r in range(masks.shape[1]):
            grid = masks[i, r].view(side, side)
            nz = torch.nonzero(grid > 0.0, as_tuple=False)
            if nz.numel() == 0:
                idx = int(torch.argmax(amap[i]).detach().cpu())
                y0 = y1 = idx // side
                x0 = x1 = idx % side
            else:
                y0 = int(nz[:, 0].min().detach().cpu())
                y1 = int(nz[:, 0].max().detach().cpu())
                x0 = int(nz[:, 1].min().detach().cpu())
                x1 = int(nz[:, 1].max().detach().cpu())
            x0 = max(0, min(side - 1, x0))
            y0 = max(0, min(side - 1, y0))
            x1 = max(x0 + 1, min(side, x1 + 1))
            y1 = max(y0 + 1, min(side, y1 + 1))
            boxes[i, r] = torch.tensor([x0 / side, y0 / side, x1 / side, y1 / side], device=amap.device, dtype=amap.dtype)
    if boxes.shape[1] < n_crops:
        repeats = math.ceil(float(n_crops) / float(max(1, boxes.shape[1])))
        boxes = boxes.repeat(1, repeats, 1)
    return boxes[:, :n_crops]


def _save_actual_crop_image(cache_dir: Path, image_path: Path, box: Sequence[float], tag: str, image_size: int) -> Path:
    rounded = tuple(round(float(v), 5) for v in box)
    digest = hashlib.sha1(f"{image_path}:{rounded}:{tag}:{image_size}:tcvrd-v1".encode()).hexdigest()[:24]
    crop_dir = cache_dir / "tcvrd_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{digest}.png"
    if crop_path.exists():
        return crop_path
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = rounded
    left = max(0, min(w - 1, int(round(x0 * w))))
    top = max(0, min(h - 1, int(round(y0 * h))))
    right = max(left + 1, min(w, int(round(x1 * w))))
    bottom = max(top + 1, min(h, int(round(y1 * h))))
    crop = img.crop((left, top, right, bottom)).resize((image_size, image_size), Image.BICUBIC)
    crop.save(crop_path)
    return crop_path


def _tokens_for_images_batch(cache, image_paths: Sequence[Path], image_size: int, target_side: int) -> Dict[Path, torch.Tensor]:
    unique_paths = list(dict.fromkeys(Path(p) for p in image_paths))
    out: Dict[Path, torch.Tensor] = {}
    missing: List[Tuple[Path, Path]] = []
    for path in unique_paths:
        cpath = cache._cache_path(f"{path}:{image_size}:{target_side}:{cache.token_dim}:{cache.backend}:v2")
        if cpath.exists():
            try:
                out[path] = torch.from_numpy(np.load(cpath)["tokens"].astype(np.float32)).to(cache.device)
                continue
            except Exception:
                cpath.unlink(missing_ok=True)
        missing.append((path, cpath))
    if missing:
        extractor = cache._extractor(image_size)
        if extractor is None:
            for path, cpath in missing:
                token = cache._fallback_tokens(path, image_size, target_side)
                token = F.normalize(token, dim=-1)
                tmp = cpath.with_name(f"{cpath.stem}.{id(token)}.tmp.npz")
                np.savez_compressed(tmp, tokens=token.detach().cpu().numpy().astype(np.float32))
                tmp.replace(cpath)
                out[path] = token.to(cache.device)
        else:
            images = [Image.open(path).convert("RGB") for path, _ in missing]
            toks, side = extractor.pil_batch_tokens(images)
            for (path, cpath), arr in zip(missing, toks):
                token = torch.from_numpy(arr.astype(np.float32)).float()
                if side != target_side:
                    token = token.reshape(side, side, -1).permute(2, 0, 1)[None]
                    token = F.interpolate(token, size=(target_side, target_side), mode="bilinear", align_corners=False)
                    token = token[0].permute(1, 2, 0).reshape(target_side * target_side, -1)
                token = F.normalize(token, dim=-1)
                tmp = cpath.with_name(f"{cpath.stem}.{id(token)}.tmp.npz")
                np.savez_compressed(tmp, tokens=token.detach().cpu().numpy().astype(np.float32))
                tmp.replace(cpath)
                out[path] = token.to(cache.device)
    return out


def _crop_map_from_box(amap: torch.Tensor, side: int, box: torch.Tensor, target_side: int) -> torch.Tensor:
    x0, y0, x1, y1 = [float(v) for v in box.detach().cpu().tolist()]
    ix0 = max(0, min(side - 1, int(math.floor(x0 * side))))
    iy0 = max(0, min(side - 1, int(math.floor(y0 * side))))
    ix1 = max(ix0 + 1, min(side, int(math.ceil(x1 * side))))
    iy1 = max(iy0 + 1, min(side, int(math.ceil(y1 * side))))
    grid = amap.view(1, 1, side, side)[:, :, iy0:iy1, ix0:ix1]
    crop = F.interpolate(grid, size=(target_side, target_side), mode="bilinear", align_corners=False)[0, 0]
    return crop.flatten().clamp(0.0, 1.0)


def _flat_map_stats(flat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    m = torch.nan_to_num(flat_map, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    mass = m.sum().clamp_min(1e-6)
    prob = m / mass
    x = coords[:, 0].to(device=m.device, dtype=m.dtype)
    y = coords[:, 1].to(device=m.device, dtype=m.dtype)
    cx = (prob * x).sum()
    cy = (prob * y).sum()
    var = (prob * ((x - cx).pow(2) + (y - cy).pow(2))).sum()
    area = (m > 0.35).to(m.dtype).mean()
    return torch.stack(
        [
            area,
            mass / float(max(1, m.numel())),
            m.max(),
            m.mean(),
            cx,
            cy,
            var,
        ]
    )


def _actual_crop_image_region_view(
    batch,
    cache,
    args,
    is_target: bool,
    n_crops: int = 8,
    unique_crops: int = 8,
    allow_source: bool = False,
) -> torch.Tensor:
    crop_dim = int(args.token_dim) * 2 + 14
    b = len(batch.eval_label)
    if not is_target and not allow_source:
        return torch.zeros(b, n_crops * crop_dim, device=batch.device, dtype=batch.map224.dtype)
    side = int(round(math.sqrt(float(batch.map224.shape[1]))))
    if side * side != batch.map224.shape[1]:
        return torch.zeros(b, n_crops * crop_dim, device=batch.device, dtype=batch.map224.dtype)
    boxes = _boxes_from_component_masks(batch.map224, side, n_crops=n_crops, unique_crops=unique_crops)
    crop_paths: List[Path] = []
    sample_paths: List[List[Path]] = []
    for i, meta in enumerate(batch.meta):
        image_path = Path(str(meta["image"]))
        per_sample: List[Path] = []
        for r in range(n_crops):
            box = boxes[i, r].detach().cpu().tolist()
            crop_path = _save_actual_crop_image(cache.cache_dir, image_path, box, f"tcvrd_{i}_{r}", int(args.image_size))
            per_sample.append(crop_path)
            crop_paths.append(crop_path)
        sample_paths.append(per_sample)
    token_lookup = _tokens_for_images_batch(cache, crop_paths, int(args.image_size), int(args.grid224))
    rows: List[torch.Tensor] = []
    coords = batch.coords224
    for i in range(b):
        crops: List[torch.Tensor] = []
        for r, crop_path in enumerate(sample_paths[i]):
            token = token_lookup[crop_path].to(batch.device)
            crop_map = _crop_map_from_box(batch.map224[i], side, boxes[i, r], int(args.grid224)).to(batch.device)
            weights = crop_map.clamp_min(1e-6)
            weights = weights / weights.sum().clamp_min(1e-6)
            full_token = F.normalize(token.mean(dim=0), dim=0)
            weighted_token = F.normalize((token * weights[:, None]).sum(dim=0), dim=0)
            stats = _flat_map_stats(crop_map, coords).to(batch.device)
            x0, y0, x1, y1 = boxes[i, r].to(batch.device)
            bw = (x1 - x0).clamp_min(1e-6)
            bh = (y1 - y0).clamp_min(1e-6)
            geom = torch.stack(
                [
                    (bw * bh).clamp(0.0, 1.0),
                    stats[1],
                    stats[2],
                    stats[3],
                    ((x0 + x1) * 0.5).clamp(0.0, 1.0),
                    ((y0 + y1) * 0.5).clamp(0.0, 1.0),
                    (bw.pow(2) + bh.pow(2)).clamp(0.0, 2.0),
                ]
            )
            crops.append(torch.cat([full_token, stats, weighted_token, geom], dim=0))
        rows.append(torch.stack(crops, dim=0))
    out = torch.stack(rows, dim=0).to(device=batch.device, dtype=batch.map224.dtype)
    return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).reshape(b, -1)


def _build_mvem_bundle(
    batch,
    normal_bank: Optional[torch.Tensor],
    args,
    is_target: bool,
    cache,
) -> base.ViewBundle:
    """Build the single MVEM view set used by the paper."""
    bundle = base.build_view_bundle(batch, normal_bank)
    teacher = build_eventlet_teacher(
        batch.tokens224,
        batch.tokens448,
        batch.map224,
        batch.map448,
        batch.coords224,
        batch.coords448,
        normal_bank,
        args.protocol,
        memory_bank448=getattr(args, "_source_normal_bank448", None) if normal_bank is not None else None,
    )
    bundle.views.append(teacher["eventlet_raw"])
    confidence = torch.ones_like(teacher["eventlet_confidence"]) if is_target else torch.zeros_like(teacher["eventlet_confidence"])
    bundle.views.append(confidence)

    n_types = int(getattr(args, "_target_n_types", 0))
    if n_types <= 0:
        n_types = int(args.n_types) if int(args.n_types) > 0 else len(set(batch.eval_label))
    ret_raw = None
    if is_target:
        ret_raw = torch.nan_to_num(
            base._ret_eventlet_signature(
                batch.tokens224,
                batch.map224,
                batch.coords224,
                normal_bank,
                32,
                64,
                compact=False,
            ),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
    ret_subspace, ret_pseudotype = _ret_pca_pseudotype_views(
        ret_raw,
        confidence,
        is_target=is_target,
        n_types=n_types,
        seed=args.seed,
        dim=32,
    )
    if cache is None:
        raise RuntimeError("The Main Method requires FeatureCache for crop features.")
    crop_view = _actual_crop_image_region_view(batch, cache, args, is_target=is_target)
    bundle.views.extend([ret_subspace, ret_pseudotype, crop_view])
    return _attach_care_evidence(bundle, batch, args, is_target)
