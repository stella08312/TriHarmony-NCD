from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _top_residual_eventlet(
    tokens: torch.Tensor,
    amap: torch.Tensor,
    coords: torch.Tensor,
    memory_bank: Optional[torch.Tensor],
    high_k: int,
    low_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = max(1, min(int(high_k), amap.shape[1]))
    vals, high_idx = torch.topk(amap, k=k, dim=1)
    high = F.normalize(tokens.gather(1, high_idx[:, :, None].expand(-1, -1, tokens.shape[-1])), dim=-1)
    high_coords = coords[high_idx]

    if memory_bank is not None and memory_bank.numel() > 0 and memory_bank.shape[-1] == tokens.shape[-1]:
        flat = high.reshape(-1, high.shape[-1])
        sim = flat @ F.normalize(memory_bank, dim=-1).T
        best_sim, best_idx = sim.max(dim=1)
        nearest = memory_bank[best_idx].reshape_as(high)
        sim_vals = best_sim.reshape(high.shape[0], high.shape[1])
    else:
        lk = max(1, min(int(low_k), amap.shape[1]))
        _, low_idx = torch.topk(-amap, k=lk, dim=1)
        low = F.normalize(tokens.gather(1, low_idx[:, :, None].expand(-1, -1, tokens.shape[-1])), dim=-1)
        sim = torch.einsum("bkd,bld->bkl", high, low)
        sim_vals, best_idx = sim.max(dim=2)
        nearest = low.gather(1, best_idx[:, :, None].expand(-1, -1, low.shape[-1]))

    residual = high - nearest
    residual_norm = residual.norm(dim=-1)
    residual = F.normalize(residual, dim=-1)
    weights = vals / vals.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = (residual * weights[:, :, None]).sum(dim=1)
    center = (high_coords * weights[:, :, None]).sum(dim=1)
    spread = ((high_coords - center[:, None, :]).pow(2) * weights[:, :, None]).sum(dim=1)
    stats = torch.cat(
        [
            vals.mean(dim=1, keepdim=True),
            vals.std(dim=1, keepdim=True),
            vals.amax(dim=1, keepdim=True),
            sim_vals.mean(dim=1, keepdim=True),
            sim_vals.std(dim=1, keepdim=True),
            sim_vals.amin(dim=1, keepdim=True),
            residual_norm.mean(dim=1, keepdim=True),
            residual_norm.std(dim=1, keepdim=True),
            residual_norm.amax(dim=1, keepdim=True),
            center,
            spread,
        ],
        dim=1,
    )
    confidence = torch.sigmoid((residual_norm.mean(dim=1, keepdim=True) + vals.mean(dim=1, keepdim=True) - sim_vals.mean(dim=1, keepdim=True)) / 0.5)
    return torch.nan_to_num(torch.cat([pooled, stats], dim=1), nan=0.0, posinf=10.0, neginf=-10.0), confidence


def build_eventlet_teacher(
    tokens224: torch.Tensor,
    tokens448: torch.Tensor,
    map224: torch.Tensor,
    map448: torch.Tensor,
    coords224: torch.Tensor,
    coords448: torch.Tensor,
    memory_bank: Optional[torch.Tensor],
    protocol: str,
    high_k: int = 32,
    low_k: int = 64,
    memory_bank448: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Create residual-eventlet teacher features without target labels.

    CARE-NCD passes the source-mask memory bank for every protocol. If a
    caller provides no bank, the function uses low-score patches from the
    same sample.
    """

    use_memory = memory_bank if memory_bank is not None and memory_bank.numel() > 0 else None
    use_memory448 = memory_bank448 if memory_bank448 is not None and memory_bank448.numel() > 0 else None
    e224, c224 = _top_residual_eventlet(tokens224, map224, coords224, use_memory, high_k=high_k, low_k=low_k)
    e448, c448 = _top_residual_eventlet(tokens448, map448, coords448, use_memory448, high_k=max(8, high_k), low_k=max(16, low_k))
    eventlet_raw = F.normalize(torch.cat([e224, e448, c224, c448], dim=1), dim=1)
    affinity = eventlet_raw @ eventlet_raw.T
    confidence = 0.5 * (c224 + c448)
    return {
        "eventlet_raw": eventlet_raw,
        "eventlet_affinity": affinity,
        "eventlet_confidence": confidence,
    }
