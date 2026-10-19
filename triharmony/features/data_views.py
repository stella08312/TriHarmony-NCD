from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from .common import Batch, FeatureCache

@dataclass
class ViewBundle:
    views: List[torch.Tensor]
    eval_label: Sequence[str]
    source_label: Optional[torch.Tensor]
    meta: Sequence[Dict[str, object]]
    map224: torch.Tensor
    map448: torch.Tensor
    normal_patch_bank_size: int

    @property
    def device(self) -> torch.device:
        return self.views[0].device


def _weighted(tokens: torch.Tensor, amap: torch.Tensor) -> torch.Tensor:
    w = amap.clamp_min(1e-6)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (tokens * w[:, :, None]).sum(dim=1)


def _top_weighted(tokens: torch.Tensor, amap: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(amap, k=max(1, min(int(k), amap.shape[1])), dim=1)
    pooled = []
    selected = []
    for i in range(tokens.shape[0]):
        ww = vals[i].clamp_min(1e-6)
        ww = ww / ww.sum().clamp_min(1e-6)
        sel = tokens[i, idx[i]]
        pooled.append((sel * ww[:, None]).sum(dim=0))
        selected.append(sel)
    return torch.stack(pooled, dim=0), torch.stack(selected, dim=0)


def _top_patch_signature(tokens: torch.Tensor, amap: torch.Tensor, coords: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(int(k), amap.shape[1]))
    vals, idx = torch.topk(amap, k=k, dim=1)
    selected = tokens.gather(1, idx[:, :, None].expand(-1, -1, tokens.shape[-1]))
    selected_coords = coords[idx]
    weights = vals / vals.sum(dim=1, keepdim=True).clamp_min(1e-6)
    center = (selected_coords * weights[:, :, None]).sum(dim=1)
    spread = ((selected_coords - center[:, None, :]).pow(2) * weights[:, :, None]).sum(dim=1)
    score_stats = torch.stack(
        [
            vals.mean(dim=1),
            vals.std(dim=1),
            vals.max(dim=1).values,
            vals[:, 0],
            vals[:, -1],
        ],
        dim=1,
    )
    return torch.cat(
        [
            (selected * vals[:, :, None].clamp_min(1e-4)).reshape(tokens.shape[0], -1),
            selected_coords.reshape(tokens.shape[0], -1),
            vals,
            center,
            spread,
            score_stats,
        ],
        dim=1,
    )


def _quadrant_weighted(tokens: torch.Tensor, amap: torch.Tensor) -> torch.Tensor:
    b, n, dim = tokens.shape
    side = int(round(math.sqrt(float(n))))
    if side * side != n:
        return torch.zeros(b, dim * 4 + 12, device=tokens.device, dtype=tokens.dtype)
    tok = tokens.view(b, side, side, dim)
    grid = amap.view(b, side, side)
    masks = []
    half = side // 2
    base = torch.zeros(side, side, device=tokens.device, dtype=tokens.dtype)
    for y0, y1, x0, x1 in [(0, half, 0, half), (0, half, half, side), (half, side, 0, half), (half, side, half, side)]:
        m = base.clone()
        m[y0:y1, x0:x1] = 1.0
        masks.append(m)
    pooled = []
    stats = []
    for mask in masks:
        w = grid * mask[None]
        mass = w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        pooled.append((tok * (w / mass)[:, :, :, None]).sum(dim=(1, 2)))
        vals = w.reshape(b, -1)
        stats.append(torch.cat([mass.reshape(b, 1) / float(n), vals.mean(dim=1, keepdim=True), vals.max(dim=1, keepdim=True).values], dim=1))
    return torch.cat(pooled + stats, dim=1)


def _morph_descriptor(m: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    mass = m.sum(dim=1).clamp_min(1e-6)
    prob = m / mass[:, None]
    x = coords[:, 0][None]
    y = coords[:, 1][None]
    cx = (prob * x).sum(dim=1, keepdim=True)
    cy = (prob * y).sum(dim=1, keepdim=True)
    dx = x - cx
    dy = y - cy
    varx = (prob * dx.pow(2)).sum(dim=1, keepdim=True)
    vary = (prob * dy.pow(2)).sum(dim=1, keepdim=True)
    cov = (prob * dx * dy).sum(dim=1, keepdim=True)
    radial = (prob * torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
    entropy = -(prob * prob.clamp_min(1e-6).log()).sum(dim=1, keepdim=True) / math.log(float(m.shape[1]))
    mean = m.mean(dim=1, keepdim=True)
    std = m.std(dim=1, keepdim=True)
    mx = m.max(dim=1, keepdim=True).values
    top = torch.topk(m, k=max(1, min(16, m.shape[1])), dim=1).values.mean(dim=1, keepdim=True)
    area = (m > 0.5).float().mean(dim=1, keepdim=True)
    side = int(round(math.sqrt(float(m.shape[1]))))
    if side * side == m.shape[1]:
        grid = m.view(m.shape[0], 1, side, side)
        edge_x = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
        edge_y = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
    else:
        edge_x = torch.zeros_like(mean)
        edge_y = torch.zeros_like(mean)
    edge = edge_x + edge_y
    compact = edge / area.clamp_min(1e-4)
    peak_ratio = mx / mean.clamp_min(1e-4)
    return torch.cat([mean, std, mx, top, area, entropy, cx, cy, varx, vary, cov, radial, edge_x, edge_y, compact, peak_ratio], dim=1)


def _normal_residual(top224: torch.Tensor, top_tokens224: torch.Tensor, normal_bank: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    b, dim = top224.shape
    if normal_bank is None:
        return torch.zeros(b, dim, device=top224.device), torch.zeros(b, 5, device=top224.device)
    deltas = []
    sim_rows = []
    for start in range(0, b, 64):
        query = top224[start : start + 64]
        sim = query @ normal_bank.T
        val, idx = sim.max(dim=1)
        nearest = normal_bank[idx]
        deltas.append(query - nearest)
        selected = top_tokens224[start : start + 64].reshape(-1, dim)
        patch_sim = selected @ normal_bank.T
        patch_val = patch_sim.max(dim=1).values.reshape(-1, top_tokens224.shape[1])
        sim_rows.append(torch.stack([val, patch_val.mean(dim=1), patch_val.std(dim=1), patch_val.min(dim=1).values, patch_val.max(dim=1).values], dim=1))
    return torch.cat(deltas, dim=0), torch.cat(sim_rows, dim=0)


def _ret_eventlet_signature(
    tokens: torch.Tensor,
    amap: torch.Tensor,
    coords: torch.Tensor,
    normal_bank: Optional[torch.Tensor],
    high_k: int,
    low_k: int,
    compact: bool = False,
) -> torch.Tensor:
    k = max(1, min(int(high_k), amap.shape[1]))
    vals, idx = torch.topk(amap, k=k, dim=1)
    selected = F.normalize(tokens.gather(1, idx[:, :, None].expand(-1, -1, tokens.shape[-1])), dim=-1)
    selected_coords = coords[idx]
    if normal_bank is not None:
        flat = selected.reshape(-1, selected.shape[-1])
        sim = flat @ normal_bank.T
        best, best_idx = sim.max(dim=1)
        nearest = normal_bank[best_idx].reshape(selected.shape)
        sim_vals = best.reshape(selected.shape[0], selected.shape[1])
    else:
        _, low_idx = torch.topk(-amap, k=max(1, min(int(low_k), amap.shape[1])), dim=1)
        low_tokens = F.normalize(tokens.gather(1, low_idx[:, :, None].expand(-1, -1, tokens.shape[-1])), dim=-1)
        sim = torch.einsum("bkd,bld->bkl", selected, low_tokens)
        sim_vals, best_idx = sim.max(dim=2)
        nearest = low_tokens.gather(1, best_idx[:, :, None].expand(-1, -1, low_tokens.shape[-1]))
    residual = F.normalize(selected - nearest, dim=-1)
    weights = vals / vals.sum(dim=1, keepdim=True).clamp_min(1e-6)
    center = (selected_coords * weights[:, :, None]).sum(dim=1)
    spread = ((selected_coords - center[:, None, :]).pow(2) * weights[:, :, None]).sum(dim=1)
    if compact:
        residual_norm = (selected - nearest).norm(dim=-1)
        entropy = -(weights * weights.clamp_min(1e-6).log()).sum(dim=1, keepdim=True) / math.log(float(max(2, k)))
        off_center = (center - 0.5).norm(dim=1, keepdim=True)
        stats = torch.cat(
            [
                vals.mean(dim=1, keepdim=True),
                vals.std(dim=1, keepdim=True),
                vals.amax(dim=1, keepdim=True),
                vals.amin(dim=1, keepdim=True),
                sim_vals.mean(dim=1, keepdim=True),
                sim_vals.std(dim=1, keepdim=True),
                sim_vals.amin(dim=1, keepdim=True),
                sim_vals.amax(dim=1, keepdim=True),
                residual_norm.mean(dim=1, keepdim=True),
                residual_norm.std(dim=1, keepdim=True),
                residual_norm.amax(dim=1, keepdim=True),
                center,
                spread,
                entropy,
                off_center,
            ],
            dim=1,
        )
        return torch.cat([selected.reshape(selected.shape[0], -1), residual.reshape(residual.shape[0], -1), stats], dim=1)
    stats = torch.cat(
        [
            vals,
            sim_vals,
            selected_coords.reshape(selected.shape[0], -1),
            center,
            spread,
            vals.mean(dim=1, keepdim=True),
            vals.std(dim=1, keepdim=True),
            sim_vals.mean(dim=1, keepdim=True),
            sim_vals.std(dim=1, keepdim=True),
            sim_vals.amin(dim=1, keepdim=True),
            sim_vals.amax(dim=1, keepdim=True),
        ],
        dim=1,
    )
    return torch.cat([selected.reshape(selected.shape[0], -1), residual.reshape(residual.shape[0], -1), stats], dim=1)


def build_view_bundle(
    batch,
    normal_bank: Optional[torch.Tensor],
) -> ViewBundle:
    mean224 = batch.tokens224.mean(dim=1)
    mean448 = batch.tokens448.mean(dim=1)
    weighted224 = _weighted(batch.tokens224, batch.map224)
    weighted448 = _weighted(batch.tokens448, batch.map448)
    top224, top_tokens224 = _top_weighted(batch.tokens224, batch.map224, 16)
    top448, _ = _top_weighted(batch.tokens448, batch.map448, 32)
    normal_delta, normal_stats = _normal_residual(top224, top_tokens224, normal_bank)
    desc = torch.cat([_morph_descriptor(batch.map224, batch.coords224), _morph_descriptor(batch.map448, batch.coords448)], dim=1)
    desc = torch.nan_to_num(desc, nan=0.0, posinf=10.0, neginf=-10.0)
    raw_top = F.normalize(torch.cat([top224, top448], dim=1), dim=1)
    raw_all = F.normalize(torch.cat([mean224, mean448, weighted224, weighted448, top224, top448], dim=1), dim=1)
    normal_view = F.normalize(torch.cat([top224, top448, normal_delta, normal_stats], dim=1), dim=1)
    shape_view = F.normalize(torch.cat([top224, top448, desc], dim=1), dim=1)
    morph_view = F.normalize(desc, dim=1)
    local224 = _quadrant_weighted(batch.tokens224, batch.map224)
    local448 = _quadrant_weighted(batch.tokens448, batch.map448)
    local_view = F.normalize(torch.cat([local224, local448, desc], dim=1), dim=1)
    event_view = F.normalize(
        torch.cat([top224, top448, local224, local448, normal_delta, normal_stats, desc], dim=1),
        dim=1,
    )
    micro224 = _top_patch_signature(batch.tokens224, batch.map224, batch.coords224, 8)
    micro448 = _top_patch_signature(batch.tokens448, batch.map448, batch.coords448, 12)
    micro_view = F.normalize(torch.cat([micro224, micro448, desc], dim=1), dim=1)
    views = [raw_top, raw_all, normal_view, shape_view, morph_view, local_view, event_view, micro_view]
    return ViewBundle(
        views=views,
        eval_label=batch.eval_label,
        source_label=batch.source_label,
        meta=batch.meta,
        map224=batch.map224,
        map448=batch.map448,
        normal_patch_bank_size=int(normal_bank.shape[0]) if normal_bank is not None else 0,
    )


def take_view_bundle(bundle: ViewBundle, index: torch.Tensor) -> ViewBundle:
    idx = index.detach().cpu().tolist()
    out = ViewBundle(
        views=[v[index] for v in bundle.views],
        eval_label=[bundle.eval_label[i] for i in idx],
        source_label=bundle.source_label[index] if bundle.source_label is not None else None,
        meta=[bundle.meta[i] for i in idx],
        map224=bundle.map224[index],
        map448=bundle.map448[index],
        normal_patch_bank_size=bundle.normal_patch_bank_size,
    )
    if hasattr(bundle, "care_contrast_view"):
        setattr(out, "care_contrast_view", getattr(bundle, "care_contrast_view")[index])
    if hasattr(bundle, "source_memory_profile_view"):
        setattr(out, "source_memory_profile_view", getattr(bundle, "source_memory_profile_view")[index])
    return out
