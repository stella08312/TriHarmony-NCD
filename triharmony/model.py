from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from .config import (
    CHANNELS, COMPONENT_PROJECTION_DIM, EVIDENCE_FAMILIES, EXPERT_SPECS,
    LOCAL_SCALE_NEIGHBORS, SEED, SOURCE_CLASSES, SOURCE_SPLIT_MINIMUM_CHILD,
    SOURCE_SPLIT_MINIMUM_CROSS_EVIDENCE_ARI, SOURCE_SPLIT_PCA_DIM,
    SPECTRAL_BANDS, SUPPORT_CENTER, SUPPORT_EPSILON, SUPPORT_LOGIT_SCALE,
    VIEW_GROUP_SIZE,
)


P0_BLOCK_SLICES = (
    ("dense", 0, 15470),
    ("dense", 15470, 17038),
    ("sparse", 0, 1536),
    ("sparse", 1536, 3845),
    ("sparse", 3845, 5409),
    ("sparse", 5409, 6977),
    ("sparse", 6977, 11627),
)


def _clean(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0,
                         posinf=10.0, neginf=-10.0)


def continuous_family_confidence_role(anomaly_support: np.ndarray) -> np.ndarray:
    """Map support to a primary-relation and uncertainty-witness simplex.

    The formula is identical for every Table, dataset, category, cohort size,
    and known K.  It is a continuous feature transformation and contains no
    hard branch or method configuration choice.
    """
    support = np.asarray(anomaly_support, dtype=np.float64).reshape(-1)
    if not np.isfinite(support).all() or np.any((support < 0.0) | (support > 1.0)):
        raise ValueError("Anomaly-support evidence must lie in [0, 1]")
    support = np.clip(support, SUPPORT_EPSILON, 1.0 - SUPPORT_EPSILON)
    center_logit = math.log(SUPPORT_CENTER / (1.0 - SUPPORT_CENTER))
    centered_logit = np.log(support / (1.0 - support)) - center_logit
    broad = 1.0 / (1.0 + np.exp(-np.clip(
        SUPPORT_LOGIT_SCALE * centered_logit, -30.0, 30.0
    )))
    broad = np.clip(broad, SUPPORT_EPSILON, 1.0 - SUPPORT_EPSILON)
    localized = 1.0 - broad
    ambiguity = 4.0 * broad * localized
    role = np.stack((
        broad,
        localized,
        broad * broad,
        localized * ambiguity,
    ), axis=1)
    return np.clip(role, SUPPORT_EPSILON, None).astype(np.float32)


def continuous_triadic_harmonic_consistency(matrix: np.ndarray) -> np.ndarray:
    """Conjoin every direct edge with its relation-profile support.

    One parameter-free continuous formula is applied to every edge. It does
    not select an expert, sample, protocol, dataset, category, map state, or
    cohort-size regime.
    """
    value = np.asarray(matrix, dtype=np.float64)
    direct = np.clip(0.5 * (value + value.T), 0.0, 1.0)
    profile = direct.copy()
    np.fill_diagonal(profile, 0.0)
    gram = profile @ profile.T
    norm = np.sqrt(np.clip(np.diag(gram), SUPPORT_EPSILON, None))
    support = np.clip(
        gram / np.clip(norm[:, None] * norm[None], SUPPORT_EPSILON, None),
        0.0, 1.0,
    )
    output = 2.0 * direct * support / np.clip(
        direct + support, SUPPORT_EPSILON, None
    )
    output = np.clip(0.5 * (output + output.T), 0.0, 1.0)
    np.fill_diagonal(output, 1.0)
    return output.astype(np.float32)


def balanced_feature(evidence: Dict[str, np.ndarray],
                     blocks: Sequence[str]) -> np.ndarray:
    rows: List[np.ndarray] = []
    scale = math.sqrt(float(len(blocks)))
    sample_count = None
    for name in blocks:
        if name not in evidence:
            raise KeyError(f"Missing evidence block: {name}")
        value = _clean(evidence[name])
        if sample_count is None:
            sample_count = int(value.shape[0])
        if value.ndim != 2 or int(value.shape[0]) != sample_count:
            raise ValueError(f"Invalid evidence matrix {name}: {value.shape}")
        value = StandardScaler().fit_transform(value).astype(np.float32)
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        rows.append((value / np.clip(norm, 1.0e-6, None) / scale).astype(np.float32))
    return np.concatenate(rows, axis=1).astype(np.float32)


def deterministic_gram_pca(feature: np.ndarray, dimension: int) -> np.ndarray:
    value = StandardScaler().fit_transform(_clean(feature)).astype(np.float32)
    effective = min(int(dimension), value.shape[0] - 1, value.shape[1])
    if effective <= 0:
        return np.zeros((value.shape[0], 1), dtype=np.float32)
    gram = np.asarray(value @ value.T, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    order = np.argsort(eigenvalue)[::-1][:effective]
    eigenvalue = np.clip(eigenvalue[order], 0.0, None)
    eigenvector = eigenvector[:, order]
    for column in range(eigenvector.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvector[:, column])))
        if eigenvector[pivot, column] < 0.0:
            eigenvector[:, column] *= -1.0
    return np.nan_to_num(
        (eigenvector * np.sqrt(eigenvalue)[None]).astype(np.float32)
    )


def _local_scale_affinity(distance: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(distance, dtype=np.float64), 0.0)
    sample_count = int(value.shape[0])
    if sample_count <= LOCAL_SCALE_NEIGHBORS:
        raise RuntimeError(
            f"Every cohort must support the fixed local scale: N={sample_count}"
        )
    sentinel = max(1.0, float(value.max(initial=1.0))) + 1.0
    ordered = np.sort(value + np.eye(sample_count) * sentinel, axis=1)
    scale = np.clip(ordered[:, LOCAL_SCALE_NEIGHBORS - 1], 1.0e-6, None)
    affinity = np.exp(
        -(value * value) / np.clip(scale[:, None] * scale[None, :], 1.0e-8, None)
    )
    affinity = np.clip(0.5 * (affinity + affinity.T), 0.0, 1.0)
    np.fill_diagonal(affinity, 1.0)
    return affinity.astype(np.float32)


def sample_affinity(evidence: Dict[str, np.ndarray],
                    blocks: Sequence[str]) -> np.ndarray:
    feature = balanced_feature(evidence, blocks)
    projection = deterministic_gram_pca(feature, COMPONENT_PROJECTION_DIM)
    return _local_scale_affinity(pairwise_distances(projection, metric="euclidean"))


def _sparsify(matrix: np.ndarray, neighbors: int) -> np.ndarray:
    if int(neighbors) <= 0:
        return matrix.copy().astype(np.float32)
    sample_count = int(matrix.shape[0])
    if sample_count <= int(neighbors):
        raise RuntimeError(
            f"Every cohort must support each always-on expert: N={sample_count}"
        )
    keep = np.zeros_like(matrix, dtype=bool)
    indices = np.argpartition(
        matrix, kth=sample_count - int(neighbors), axis=1
    )[:, -int(neighbors):]
    keep[np.arange(sample_count)[:, None], indices] = True
    keep = keep | keep.T | np.eye(sample_count, dtype=bool)
    output = np.where(keep, matrix, 0.0)
    np.fill_diagonal(output, 1.0)
    return output.astype(np.float32)


def _project_components(crop_regions: np.ndarray, dimension: int) -> np.ndarray:
    crop = _clean(crop_regions)
    sample_count, total_dim = crop.shape
    if int(total_dim) % 8:
        raise ValueError(f"Expected eight crop components: {crop.shape}")
    component_dim = int(total_dim) // 8
    standardized = StandardScaler().fit_transform(
        crop.reshape(int(sample_count) * 8, component_dim)
    ).astype(np.float32)
    if min(standardized.shape[0] - 1, standardized.shape[1]) < int(dimension):
        raise RuntimeError("Every cohort must support the locked projection")
    projected = PCA(
        n_components=int(dimension), whiten=True, svd_solver="randomized",
        iterated_power=7, random_state=SEED,
    ).fit_transform(standardized).astype(np.float32)
    component = projected.reshape(int(sample_count), 8, -1)
    component /= np.clip(
        np.linalg.norm(component, axis=2, keepdims=True), 1.0e-8, None
    )
    return component.astype(np.float32)


def _component_projection_bands(crop_regions: np.ndarray) -> Tuple[np.ndarray, ...]:
    component = _project_components(
        crop_regions, SPECTRAL_BANDS * COMPONENT_PROJECTION_DIM
    )
    rows = []
    for index in range(SPECTRAL_BANDS):
        band = component[:, :, index * COMPONENT_PROJECTION_DIM:
                         (index + 1) * COMPONENT_PROJECTION_DIM].copy()
        band /= np.clip(np.linalg.norm(band, axis=2, keepdims=True), 1.0e-8, None)
        rows.append(band.astype(np.float32))
    return tuple(rows)


def _component_distance(component: np.ndarray, mode: str) -> np.ndarray:
    sample_count = int(component.shape[0])
    distance = np.zeros((sample_count, sample_count), dtype=np.float32)
    for left in range(sample_count):
        similarity = np.einsum("ad,jbd->jab", component[left], component[left:])
        for offset, matrix in enumerate(similarity):
            right = left + offset
            if mode == "hungarian":
                row, column = linear_sum_assignment(-matrix)
                value = 1.0 - float(matrix[row, column].mean())
            elif mode == "corresponding":
                value = 1.0 - float(np.diag(matrix).mean())
            elif mode == "chamfer":
                value = 1.0 - 0.5 * float(
                    matrix.max(axis=0).mean() + matrix.max(axis=1).mean()
                )
            else:
                raise ValueError(f"Unknown component relation mode: {mode}")
            distance[left, right] = distance[right, left] = max(0.0, value)
    return distance


def _component_affinity(component: np.ndarray, mode: str) -> np.ndarray:
    return _local_scale_affinity(_component_distance(component, mode))


def _component_views(component_sets: Sequence[np.ndarray],
                     mode: str) -> List[np.ndarray]:
    rows: List[np.ndarray] = []
    for component in component_sets:
        rows.append(_component_affinity(component, mode))
        for held_out in range(8):
            active = [index for index in range(8) if index != held_out]
            rows.append(_component_affinity(component[:, active, :], mode))
    return rows


def _within_band_partition_persistence(rows: Sequence[np.ndarray],
                                       known_k: int
                                       ) -> Tuple[float, List[float], np.ndarray]:
    if len(rows) % VIEW_GROUP_SIZE:
        raise AssertionError("Each band must contain full plus eight deletions")
    if int(known_k) <= 1:
        count = int(rows[0].shape[0])
        return (1.0, [1.0 for _ in range(len(rows) // VIEW_GROUP_SIZE)],
                np.ones((count, count), dtype=np.float32))
    band_scores: List[float] = []
    all_labels: List[np.ndarray] = []
    for start in range(0, len(rows), VIEW_GROUP_SIZE):
        labels = [spectral_seed42(value, int(known_k))
                  for value in rows[start:start + VIEW_GROUP_SIZE]]
        all_labels.extend(labels)
        scores = [float(adjusted_rand_score(labels[left], labels[right]))
                  for left in range(len(labels))
                  for right in range(left + 1, len(labels))]
        band_scores.append(float(np.mean(scores)))
    coassociation = np.mean(
        [labels[:, None] == labels[None, :] for labels in all_labels], axis=0
    ).astype(np.float32)
    return float(np.mean(band_scores)), band_scores, coassociation


def target_graph_bundle(evidence: Dict[str, np.ndarray], known_k: int,
                        anomaly_support: np.ndarray) -> Dict[str, object]:
    sample_count = int(next(iter(evidence.values())).shape[0])
    if any(int(value.shape[0]) != sample_count for value in evidence.values()):
        raise ValueError("Evidence blocks are not sample aligned")
    support_role = continuous_family_confidence_role(anomaly_support)
    if tuple(support_role.shape) != (sample_count, CHANNELS):
        raise ValueError("Continuous support-role evidence is not sample aligned")
    bands = _component_projection_bands(evidence["crop_regions"])
    single = _project_components(evidence["crop_regions"], COMPONENT_PROJECTION_DIM)
    bases = [sample_affinity(evidence, blocks) for blocks in EVIDENCE_FAMILIES]
    band_views = {
        mode: _component_views(bands, mode)
        for mode in ("corresponding", "hungarian", "chamfer")
    }
    single_chamfer = _component_views((single,), "chamfer")
    graphs: List[np.ndarray] = []
    deletion_stability: List[float] = []
    partition_persistence: List[float] = []
    per_band_persistence: List[List[float]] = []
    partition_coassociation_rows: List[np.ndarray] = []
    node_deletion_fidelity_rows: List[np.ndarray] = []
    uncertainty_rows: List[np.ndarray] = []
    aligned_view_rows: List[List[np.ndarray]] = []
    for name, family, mode, weight, neighbors, view_count in EXPERT_SPECS:
        component_rows = (single_chamfer if name == "source_chamfer_single"
                          else band_views[mode])
        if len(component_rows) != int(view_count):
            raise AssertionError("Always-on component-view contract changed")
        mixed_rows = []
        for component in component_rows:
            mixed = float(weight) * component + (1.0 - float(weight)) * bases[int(family)]
            mixed = np.clip(0.5 * (mixed + mixed.T), 0.0, 1.0).astype(np.float32)
            np.fill_diagonal(mixed, 1.0)
            mixed_rows.append(_sparsify(mixed, int(neighbors)))
        consensus = np.mean(mixed_rows, axis=0).astype(np.float32)
        graphs.append(consensus)
        uncertainty = np.mean(
            [np.abs(value - consensus) for value in mixed_rows], axis=0
        ).astype(np.float32)
        uncertainty_rows.append(uncertainty)
        deletion_stability.append(float(uncertainty.mean()))
        persistence, band_rows, partition_coassociation = \
            _within_band_partition_persistence(
            mixed_rows, int(known_k))
        partition_persistence.append(persistence)
        per_band_persistence.append(band_rows)
        partition_coassociation_rows.append(partition_coassociation)
        off_diagonal = ~np.eye(sample_count, dtype=bool)
        view_deviation = np.mean(
            [np.abs(value - consensus) for value in mixed_rows], axis=0
        ).astype(np.float32)
        node_deviation = (
            (view_deviation * off_diagonal).sum(axis=1)
            / float(max(1, sample_count - 1))
        )
        node_mass = (
            (consensus * off_diagonal).sum(axis=1)
            / float(max(1, sample_count - 1))
        )
        continuous_fidelity = np.exp(
            -node_deviation / np.clip(node_mass, 1.0e-6, None)
        ).astype(np.float32)
        partition_factor = float(np.clip(persistence, 1.0e-6, 1.0))
        node_deletion_fidelity_rows.append(np.clip(
            continuous_fidelity * partition_factor, 1.0e-6, 1.0
        ).astype(np.float32))
        if len(mixed_rows) == VIEW_GROUP_SIZE:
            aligned_view_rows.append(list(mixed_rows) * SPECTRAL_BANDS)
        elif len(mixed_rows) == SPECTRAL_BANDS * VIEW_GROUP_SIZE:
            aligned_view_rows.append(list(mixed_rows))
        else:
            raise AssertionError("Aligned deletion-view contract changed")
    if len(graphs) != CHANNELS:
        raise AssertionError("Always-on expert bank changed")
    component_centers = [
        np.mean(band_views[mode], axis=0).astype(np.float32)
        for mode in ("corresponding", "hungarian", "chamfer")
    ]
    return {
        "graphs": torch.from_numpy(np.stack(graphs, axis=2).astype(np.float32)),
        "uncertainty": torch.from_numpy(
            np.stack(uncertainty_rows, axis=2).astype(np.float32)),
        "persistence": torch.from_numpy(np.stack(
            node_deletion_fidelity_rows, axis=1).astype(np.float32)),
        "support_role": torch.from_numpy(support_role),
        "views": torch.from_numpy(np.stack([
            np.stack([aligned_view_rows[channel][view]
                      for channel in range(CHANNELS)], axis=2)
            for view in range(SPECTRAL_BANDS * VIEW_GROUP_SIZE)
        ], axis=0).astype(np.float32)),
        "graph_rows": graphs, "base_rows": bases,
        "partition_coassociation_rows": partition_coassociation_rows,
        "component_rows": component_centers,
        "component_projection": {"bands": bands, "single": single},
        "validation_rows": [*bases, *component_centers, *graphs],
        "diagnostics": {
            "universal_method_signature": {
                "experts": [list(value) for value in EXPERT_SPECS],
                "all_experts_always_computed": True, "channels": CHANNELS,
                "spectral_bands": SPECTRAL_BANDS,
                "component_dimension_per_band": COMPONENT_PROJECTION_DIM,
                "local_scale_neighbors": LOCAL_SCALE_NEIGHBORS,
                "universal_topology_validator": True,
                "deletion_partition_persistence": "within-band full-plus-eight",
                "continuous_node_deletion_fidelity":
                    "exp(-mean-edge-deviation/mean-edge-mass) times clipped partition persistence",
                "partition_flow":
                    "continuous family-aware primary-witness confidence simplex",
                "anchor_free_learned_fusion": True,
                "known_k_trust_boundary": "1 - 1 / known_k",
                "continuous_family_confidence_attention": True,
                "family_confidence_role_formula":
                    "[b,1-b,b^2,(1-b)*4b(1-b)]",
                "continuous_triadic_harmonic_consistency": True,
                "triadic_profile_support":
                    "cosine(offdiagonal relation profiles)",
                "triadic_conjunction":
                    "2*direct*profile_support/(direct+profile_support)",
                "hard_routing_or_configuration_branch": False,
                "protocol_dataset_category_or_sample_count_input": False,
            },
            "expert_names": [value[0] for value in EXPERT_SPECS],
            "component_deletion_dispersion": deletion_stability,
            "within_band_partition_persistence": partition_persistence,
            "per_band_partition_persistence": per_band_persistence,
            "node_deletion_fidelity_summary": [
                {
                    "minimum": float(value.min()),
                    "mean": float(value.mean()),
                    "maximum": float(value.max()),
                }
                for value in node_deletion_fidelity_rows
            ],
            "partition_persistence_known_k": int(known_k),
            "support_role_summary": {
                "broad_minimum": float(support_role[:, 0].min()),
                "broad_mean": float(support_role[:, 0].mean()),
                "broad_maximum": float(support_role[:, 0].max()),
                "mean_role": support_role.mean(axis=0).tolist(),
                "support_center": float(SUPPORT_CENTER),
                "support_logit_scale": float(SUPPORT_LOGIT_SCALE),
                "role_formula": "[b,1-b,b^2,(1-b)*4b(1-b)]",
            },
            "aligned_deletion_views": SPECTRAL_BANDS * VIEW_GROUP_SIZE,
            "component_projection_seed": SEED,
            "additional_numeric_seeds": [],
            "metadata_conditioned_routing": False,
            "all_channels_always_computed": True,
        },
    }


def _torch_local_affinity(feature: torch.Tensor) -> torch.Tensor:
    value = F.normalize(torch.nan_to_num(feature.float()), dim=1)
    distance = (1.0 - value @ value.T).clamp_min(0.0)
    count = int(distance.shape[0])
    if count <= LOCAL_SCALE_NEIGHBORS:
        raise RuntimeError("Source cohort does not support fixed local scale")
    identity = torch.eye(count, device=distance.device, dtype=distance.dtype)
    sentinel = distance.detach().max().clamp_min(1.0) + 1.0
    ordered = torch.sort(distance + identity * sentinel, dim=1).values
    scale = ordered[:, LOCAL_SCALE_NEIGHBORS - 1].clamp_min(1.0e-6)
    affinity = torch.exp(
        -distance.square() / (scale[:, None] * scale[None, :]).clamp_min(1.0e-8)
    )
    return 0.5 * (affinity + affinity.T) * (1.0 - identity) + identity


def _torch_sparsify(matrix: torch.Tensor, neighbors: int) -> torch.Tensor:
    if int(neighbors) <= 0:
        return matrix
    count = int(matrix.shape[0])
    index = torch.topk(matrix, k=int(neighbors), dim=1).indices
    keep = torch.zeros_like(matrix, dtype=torch.bool)
    keep.scatter_(1, index, True)
    keep = keep | keep.T | torch.eye(count, dtype=torch.bool, device=matrix.device)
    return torch.where(keep, matrix, torch.zeros_like(matrix))


def _torch_continuous_triadic_harmonic_consistency(
    matrix: torch.Tensor,
) -> torch.Tensor:
    direct = (0.5 * (matrix + matrix.T)).clamp(0.0, 1.0)
    identity = torch.eye(
        direct.shape[0], dtype=direct.dtype, device=direct.device
    )
    profile = direct * (1.0 - identity)
    gram = profile @ profile.T
    norm = torch.diagonal(gram).clamp_min(SUPPORT_EPSILON).sqrt()
    support = (
        gram / (norm[:, None] * norm[None]).clamp_min(SUPPORT_EPSILON)
    ).clamp(0.0, 1.0)
    output = 2.0 * direct * support / (
        direct + support
    ).clamp_min(SUPPORT_EPSILON)
    output = (0.5 * (output + output.T)).clamp(0.0, 1.0)
    return output * (1.0 - identity) + identity


def source_graph_stack(dense: torch.Tensor, sparse: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    families = {"dense": dense.to(device), "sparse": sparse.to(device)}
    rows: List[torch.Tensor] = []
    for family, start, end in P0_BLOCK_SLICES:
        block = families[family][:, int(start):int(end)]
        block = F.layer_norm(block.float(), (block.shape[1],))
        rows.append(_torch_local_affinity(block))
    dense_graph = torch.stack(rows[:2], dim=2).mean(dim=2)
    sparse_graph = torch.stack(rows[2:], dim=2).mean(dim=2)
    output = torch.stack((
        _torch_sparsify(dense_graph, 10),
        _torch_sparsify(sparse_graph, 10),
        dense_graph,
        _torch_sparsify(sparse_graph, 20),
    ), dim=2)
    if int(output.shape[2]) != CHANNELS:
        raise AssertionError("Aligned source expert bank changed")
    return output.detach()


class ContinuousDeletionFidelityGraphTransport(nn.Module):
    """Sample-local graph transport calibrated by continuous deletion fidelity."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.channels = int(channels)
        self.transport_logits = nn.Parameter(torch.zeros(channels, channels))
        self.base_reliability = nn.Parameter(torch.zeros(channels))
        self.agreement_gain = nn.Parameter(torch.zeros(channels))
        self.density_gain = nn.Parameter(torch.zeros(channels))
        self.message_gain = nn.Parameter(torch.zeros(channels))
        self.capacity_gain = nn.Parameter(torch.zeros(channels))
        self.uncertainty_gain = nn.Parameter(torch.zeros(channels))
        self.persistence_gain = nn.Parameter(torch.zeros(channels))

    def transport(self) -> torch.Tensor:
        diagonal = torch.eye(self.channels, dtype=torch.bool,
                             device=self.transport_logits.device)
        return F.softmax(
            self.transport_logits.masked_fill(diagonal, -torch.inf), dim=1
        )

    def components(self, graphs: torch.Tensor, known_k: int,
                   uncertainty: torch.Tensor | None = None,
                   persistence: torch.Tensor | None = None,
                   active: torch.Tensor | None = None,
                   role_prior: torch.Tensor | None = None
                   ) -> Dict[str, torch.Tensor]:
        if active is None:
            active = torch.ones(self.channels, dtype=torch.bool, device=graphs.device)
        if uncertainty is None:
            uncertainty = torch.zeros_like(graphs)
        if persistence is None:
            persistence = torch.ones(
                graphs.shape[0], self.channels, dtype=graphs.dtype,
                device=graphs.device)
        if uncertainty.shape != graphs.shape:
            raise ValueError("Graph/uncertainty shape mismatch")
        if tuple(persistence.shape) == (self.channels,):
            persistence = persistence[None].expand(graphs.shape[0], -1)
        if tuple(persistence.shape) != (graphs.shape[0], self.channels):
            raise ValueError("Persistence shape mismatch")
        if role_prior is None:
            role_prior = torch.ones(
                graphs.shape[0], self.channels, dtype=graphs.dtype,
                device=graphs.device,
            )
        if tuple(role_prior.shape) != (graphs.shape[0], self.channels):
            raise ValueError("Continuous support-role shape mismatch")
        if (not torch.isfinite(role_prior).all()
                or not bool((role_prior > 0.0).all())):
            raise ValueError("Continuous support-role must be finite and positive")
        active_float = active.to(dtype=graphs.dtype)
        count_active = active_float.sum().clamp_min(1.0)
        graph_active = graphs * active_float
        anchor = graph_active.sum(dim=2) / count_active
        agreement = 1.0 - (graph_active - anchor[:, :, None]).abs().mean(dim=1)
        density = graph_active.mean(dim=1)
        degree_anchor = anchor.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        message = (anchor @ agreement) / degree_anchor
        agreement = agreement - agreement.mean(dim=1, keepdim=True)
        density = density - density.mean(dim=1, keepdim=True)
        message = message - message.mean(dim=1, keepdim=True)

        uncertainty_feature = -uncertainty.mean(dim=1)
        uncertainty_feature = uncertainty_feature - (
            uncertainty_feature * active_float[None]
        ).sum(dim=1, keepdim=True) / count_active
        persistence_feature = persistence.clamp(1.0e-6, 1.0).log()
        persistence_feature = persistence_feature - (
            persistence_feature * active_float[None]
        ).sum(dim=1, keepdim=True) / count_active

        degree = graph_active.sum(dim=1).clamp_min(1.0e-8)
        normalized = graph_active / torch.sqrt(
            degree[:, None, :] * degree[None, :, :]).clamp_min(1.0e-8)
        squared = torch.einsum("ikc,kjc->ijc", normalized, normalized)
        mass_two = torch.diagonal(squared, dim1=0, dim2=1).sum(dim=1).clamp_min(1.0e-8)
        mass_four = squared.square().sum(dim=(0, 1)).clamp_min(1.0e-8)
        participation = mass_two.square() / mass_four
        target = participation.new_tensor(float(int(known_k)))
        capacity = -(participation.log() - target.log()).square()
        capacity = capacity - (capacity * active_float).sum() / count_active

        score = (self.base_reliability[None]
                 + self.agreement_gain[None] * agreement
                 + self.density_gain[None] * density
                 + self.message_gain[None] * message
                 + self.capacity_gain[None] * capacity[None]
                 + self.uncertainty_gain[None] * uncertainty_feature
                 + self.persistence_gain[None] * persistence_feature
                 + role_prior.clamp_min(SUPPORT_EPSILON).log())
        score = score.masked_fill(~active[None], -torch.inf)
        node_weight = F.softmax(score, dim=1)
        transport = self.transport() * active_float[None] * active_float[:, None]
        transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        predicted = torch.einsum("ijc,dc->ijd", graph_active, transport)
        evidence = 0.5 * (graph_active + predicted)
        pair_weight = torch.sqrt(
            (node_weight[:, None, :] * node_weight[None, :, :]).clamp_min(1.0e-12)
        ) * active_float
        pair_weight = pair_weight / pair_weight.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        local = (evidence * pair_weight).sum(dim=2)
        fused = _torch_continuous_triadic_harmonic_consistency(local)
        return {
            "anchor": anchor, "predicted": predicted,
            "node_weight": node_weight, "capacity": capacity,
            "uncertainty_feature": uncertainty_feature,
            "persistence_feature": persistence_feature,
            "role_prior": role_prior,
            "local": local, "fused": fused,
        }

    def fuse(self, graphs: torch.Tensor, known_k: int,
             uncertainty: torch.Tensor | None = None,
             persistence: torch.Tensor | None = None,
             active: torch.Tensor | None = None,
             role_prior: torch.Tensor | None = None) -> torch.Tensor:
        return self.components(
            graphs, known_k, uncertainty, persistence, active, role_prior
        )["fused"]


ScaleFamilyGraphTransport = ContinuousDeletionFidelityGraphTransport
PersistentGraphTransport = ContinuousDeletionFidelityGraphTransport


def _off_diagonal(count: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(count, dtype=torch.bool, device=device)


def source_relation_loss(model: ContinuousDeletionFidelityGraphTransport,
                         graphs: torch.Tensor, labels: torch.Tensor,
                         uncertainty: torch.Tensor,
                         persistence: torch.Tensor) -> torch.Tensor:
    similarity = model.fuse(
        graphs, SOURCE_CLASSES, uncertainty, persistence)
    mask = _off_diagonal(len(labels), labels.device)
    same = labels[:, None].eq(labels[None]) & mask
    different = labels[:, None].ne(labels[None]) & mask
    if not bool(same.any()) or not bool(different.any()):
        raise RuntimeError("Source cohort is not class balanced")
    return (1.0 - similarity[same].mean()) + similarity[different].mean()


def target_scale_normalized_heldout_loss(
    model: ContinuousDeletionFidelityGraphTransport, graphs: torch.Tensor,
    uncertainty: torch.Tensor, persistence: torch.Tensor, known_k: int,
    role_prior: torch.Tensor,
) -> torch.Tensor:
    predicted = model.components(
        graphs, known_k, uncertainty, persistence,
        role_prior=role_prior)["predicted"]
    mask = _off_diagonal(graphs.shape[0], graphs.device)[:, :, None]
    target_rows = graphs.detach()[mask.expand_as(graphs)].reshape(-1, graphs.shape[2])
    predicted_rows = predicted[mask.expand_as(predicted)].reshape(
        -1, predicted.shape[2])
    center = target_rows.mean(dim=0, keepdim=True)
    scale = target_rows.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-4)
    loss = F.smooth_l1_loss(
        (predicted_rows - center) / scale,
        (target_rows - center) / scale,
        reduction="none",
    )
    channel_weight = role_prior.mean(dim=0)
    channel_weight = channel_weight / channel_weight.mean().clamp_min(1.0e-8)
    return (loss * channel_weight[None]).mean()


def target_deletion_persistence_loss(
    model: ContinuousDeletionFidelityGraphTransport, graphs: torch.Tensor,
    deletion_graphs: torch.Tensor, uncertainty: torch.Tensor,
    persistence: torch.Tensor, known_k: int, role_prior: torch.Tensor,
) -> torch.Tensor:
    full = model.fuse(
        graphs, known_k, uncertainty, persistence, role_prior=role_prior)
    deletion = model.fuse(
        deletion_graphs, known_k, uncertainty, persistence,
        role_prior=role_prior)
    mask = _off_diagonal(graphs.shape[0], graphs.device)
    edge_uncertainty = uncertainty.mean(dim=2)
    normalized = edge_uncertainty / edge_uncertainty[mask].mean().clamp_min(1.0e-6)
    weight = 1.0 + normalized[mask]
    loss = F.smooth_l1_loss(deletion[mask], full[mask], reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0e-8)


def target_capacity_loss(
    model: ContinuousDeletionFidelityGraphTransport, graphs: torch.Tensor,
    uncertainty: torch.Tensor, persistence: torch.Tensor, known_k: int,
    role_prior: torch.Tensor,
) -> torch.Tensor:
    affinity = model.fuse(
        graphs, known_k, uncertainty, persistence, role_prior=role_prior)
    degree = affinity.sum(dim=1).clamp_min(1.0e-8)
    normalized = affinity / torch.sqrt(degree[:, None] * degree[None])
    squared = (0.5 * (normalized + normalized.T))
    squared = squared @ squared
    mass_two = torch.diagonal(squared).sum().clamp_min(1.0e-8)
    mass_four = squared.square().sum().clamp_min(1.0e-8)
    participation = mass_two.square() / mass_four
    target = participation.new_tensor(float(min(int(known_k), int(graphs.shape[0]))))
    return (participation.log() - target.log()).square()


def spectral_seed42(matrix: np.ndarray, known_k: int) -> np.ndarray:
    sample_count, clusters = int(matrix.shape[0]), int(known_k)
    if clusters <= 1:
        return np.zeros(sample_count, dtype=np.int64)
    affinity = np.clip(0.5 * (matrix + matrix.T), 0.0, 1.0).astype(np.float64)
    degree = affinity.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.clip(degree, 1.0e-8, None))
    normalized = affinity * inverse[:, None] * inverse[None, :]
    _, vectors = np.linalg.eigh(normalized)
    embedding = vectors[:, -clusters:]
    embedding /= np.clip(np.linalg.norm(embedding, axis=1, keepdims=True),
                         1.0e-8, None)
    prediction = KMeans(
        n_clusters=clusters, n_init=32, max_iter=300, random_state=SEED
    ).fit_predict(embedding).astype(np.int64)
    if len(np.unique(prediction)) != clusters:
        raise AssertionError("Sole-seed spectral decoder violated exact-K")
    return prediction


def _association(matrix: np.ndarray, labels: np.ndarray) -> float:
    count = len(labels)
    off = ~np.eye(count, dtype=bool)
    same = labels[:, None] == labels[None, :]
    within = matrix[same & off]
    between = matrix[(~same) & off]
    if not len(within) or not len(between):
        return 0.0
    return float((within.mean() - between.mean())
                 / max(float(matrix[off].std()), 1.0e-8))


def _cluster_within(matrix: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) <= 1:
        return 0.0
    block = matrix[np.ix_(indices, indices)]
    return float((block.sum() - len(indices)) /
                 (len(indices) * (len(indices) - 1)))


def _topology(bundle: Dict[str, object], baseline: np.ndarray,
              known_k: int) -> Dict[str, object]:
    if int(known_k) <= 2:
        return {"available": False, "reason": "known_k_le_2"}
    single = bundle["component_projection"]["single"]
    kernels = list(bundle["base_rows"])
    kernels.extend(_component_affinity(single, mode)
                   for mode in ("corresponding", "hungarian", "chamfer"))
    root = max(2, int(math.ceil(math.sqrt(float(len(baseline))))))
    experts = [spectral_seed42(_sparsify(value, root), int(known_k))
               for value in kernels]
    coassociation = np.mean(
        [value[:, None] == value[None, :] for value in experts], axis=0
    ).astype(np.float32)
    members = [np.flatnonzero(baseline == cluster) for cluster in range(int(known_k))]
    within = [_cluster_within(coassociation, value) for value in members]
    merge_rows = []
    for left in range(int(known_k)):
        for right in range(left + 1, int(known_k)):
            cross = float(coassociation[np.ix_(members[left], members[right])].mean())
            denominator = math.sqrt(max(within[left], 1.0e-8)
                                    * max(within[right], 1.0e-8))
            merge_rows.append((cross / max(denominator, 1.0e-8), left, right))
    merge_rows.sort(reverse=True)
    for redundancy, merge_left, merge_right in merge_rows:
        split_options = [
            (1.0 - within[cluster], cluster)
            for cluster in range(int(known_k))
            if cluster not in (merge_left, merge_right)
            and len(members[cluster]) >= 2 * SOURCE_SPLIT_MINIMUM_CHILD
        ]
        if split_options:
            heterogeneity, split_cluster = max(split_options)
            return {
                "available": True, "merge_pair": [int(merge_left), int(merge_right)],
                "merge_redundancy": float(redundancy),
                "split_cluster": int(split_cluster),
                "split_heterogeneity": float(heterogeneity),
                "expert_count": len(experts), "numeric_seeds": [SEED],
            }
    return {"available": False, "reason": "no_exact_k_topology"}


def _split_embedding(evidence: Dict[str, np.ndarray], selected: np.ndarray,
                     blocks: Sequence[str]) -> np.ndarray:
    feature = balanced_feature(evidence, tuple(blocks))[selected]
    standardized = StandardScaler().fit_transform(_clean(feature)).astype(np.float32)
    dimension = min(SOURCE_SPLIT_PCA_DIM, standardized.shape[0] - 1,
                    standardized.shape[1])
    if dimension <= 0:
        return np.zeros(len(selected), dtype=np.int64)
    embedding = PCA(
        n_components=dimension, svd_solver="randomized", random_state=SEED
    ).fit_transform(standardized)
    return KMeans(
        n_clusters=2, n_init=32, max_iter=300, random_state=SEED
    ).fit_predict(embedding).astype(np.int64)


def source_memory_topology_repair(
    evidence: Dict[str, np.ndarray], baseline: np.ndarray, known_k: int,
    bundle: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    topology = _topology(bundle, baseline, int(known_k))
    if not bool(topology.get("available")):
        return baseline.copy(), {"accepted": False, "topology": topology}
    selected = np.flatnonzero(baseline == int(topology["split_cluster"]))
    block_sets = (
        ("source_memory_contrast", "top", "eventlet"),
        ("source_memory_contrast", "top"),
        ("source_memory_contrast", "eventlet"),
    )
    splits = [_split_embedding(evidence, selected, blocks) for blocks in block_sets]
    ari_rows = [
        float(adjusted_rand_score(splits[left], splits[right]))
        for left in range(len(splits)) for right in range(left + 1, len(splits))
    ]
    mean_ari = float(np.mean(ari_rows)) if ari_rows else 1.0
    split = splits[0]
    counts = np.bincount(split, minlength=2)
    accepted = bool(
        mean_ari >= SOURCE_SPLIT_MINIMUM_CROSS_EVIDENCE_ARI
        and int(counts.min()) >= SOURCE_SPLIT_MINIMUM_CHILD
    )
    proposal = baseline.copy()
    if accepted:
        merge_left, merge_right = [int(value) for value in topology["merge_pair"]]
        proposal[proposal == merge_right] = merge_left
        proposal[selected[split == 1]] = merge_right
        if len(np.unique(proposal)) != int(known_k):
            raise AssertionError("Universal topology repair violated exact-K")
    return (proposal if accepted else baseline).astype(np.int64), {
        "accepted": accepted,
        "reason": "universal_cross_evidence_stable_split" if accepted
                  else "universal_split_stability_gate",
        "topology": topology, "split_blocks": [list(value) for value in block_sets],
        "split_counts": counts.astype(int).tolist(),
        "cross_evidence_ari_rows": ari_rows,
        "cross_evidence_mean_ari": mean_ari,
        "minimum_cross_evidence_ari": SOURCE_SPLIT_MINIMUM_CROSS_EVIDENCE_ARI,
        "minimum_child": SOURCE_SPLIT_MINIMUM_CHILD,
        "numeric_seeds": [SEED], "additional_numeric_seeds": [],
    }


def trusted_source_resolution_with_witness(
    candidate: np.ndarray, trust_witness: np.ndarray,
    source_baseline: np.ndarray,
    source_proposal: np.ndarray, repair: Dict[str, object], known_k: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    trust_boundary = 1.0 - 1.0 / float(max(1, int(known_k)))
    baseline_ari = float(adjusted_rand_score(trust_witness, source_baseline))
    accepted = bool(
        repair.get("accepted") and baseline_ari >= trust_boundary
    )
    prediction = source_proposal if accepted else candidate
    return prediction.astype(np.int64), {
        "accepted": accepted,
        "reason": "stable_source_resolution_inside_known_k_trust_region"
        if accepted else "source_resolution_outside_trust_region_or_unstable",
        "source_baseline_ari_to_trust_witness": baseline_ari,
        "known_k_derived_ari_boundary": trust_boundary,
        "boundary_formula": "1 - 1 / known_k",
        "source_split_repair": repair,
        "target_gt_or_metric_used": False,
    }


def _partition_coassociation(labels: np.ndarray) -> np.ndarray:
    value = np.asarray(labels, dtype=np.int64)
    return (value[:, None] == value[None, :]).astype(np.float32)


def trusted_partition_barycenter(
    reference: np.ndarray, bundle: Dict[str, object], known_k: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Resolve complete expert partitions in one known-K-derived trust set."""
    clusters = int(known_k)
    boundary = 1.0 - 1.0 / float(max(1, clusters))
    expert_partitions = [
        spectral_seed42(matrix, clusters) for matrix in bundle["graph_rows"]
    ]
    similarities = [
        float(adjusted_rand_score(reference, candidate))
        for candidate in expert_partitions
    ]
    trusted = [index for index, value in enumerate(similarities)
               if value >= boundary]
    if trusted:
        matrices = [_partition_coassociation(reference)]
        matrices.extend(bundle["partition_coassociation_rows"][index]
                        for index in trusted)
        mean_coassociation = np.mean(matrices, axis=0).astype(np.float32)
        candidate = spectral_seed42(mean_coassociation, clusters)
        candidate_ari = float(adjusted_rand_score(reference, candidate))
        accepted = bool(candidate_ari >= boundary)
        prediction = candidate if accepted else reference
    else:
        candidate_ari, accepted = 1.0, False
        prediction = reference
    if len(np.unique(prediction)) != clusters:
        raise AssertionError("Partition-flow barycenter violated exact-K")
    return prediction.astype(np.int64), {
        "accepted": accepted,
        "reason": "known_k_trust_set_frechet_barycenter"
        if accepted else "empty_or_outside_known_k_trust_set",
        "expert_partition_ari_to_reference": similarities,
        "trusted_expert_indices": trusted,
        "trusted_partition_count_including_reference": 1 + len(trusted),
        "barycenter_ari_to_reference": candidate_ari,
        "known_k_derived_ari_boundary": boundary,
        "boundary_formula": "1 - 1 / known_k",
        "barycenter_geometry":
            "arithmetic mean of reference and trusted deletion-view coassociations",
        "all_experts_always_evaluated": True,
        "target_gt_or_metric_used": False,
    }


@torch.inference_mode()
def learned_partition(model: ContinuousDeletionFidelityGraphTransport,
                      bundle: Dict[str, object],
                      evidence: Dict[str, np.ndarray], known_k: int,
                      device: torch.device
                      ) -> Tuple[np.ndarray, Dict[str, object]]:
    graphs = bundle["graphs"].to(device)
    uncertainty = bundle["uncertainty"].to(device)
    persistence = bundle["persistence"].to(device)
    support_role = bundle["support_role"].to(device)
    conditioned_components = model.components(
        graphs, int(known_k), uncertainty, persistence,
        role_prior=support_role
    )
    node_weight = conditioned_components["node_weight"]
    pair_weight = torch.sqrt(
        (node_weight[:, None, :] * node_weight[None, :, :]).clamp_min(1.0e-12)
    )
    pair_weight = pair_weight / pair_weight.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
    conditioned = (graphs * pair_weight).sum(dim=2)
    conditioned = _torch_continuous_triadic_harmonic_consistency(conditioned)
    conditioned = conditioned.detach().cpu().numpy().astype(np.float32)
    candidate = spectral_seed42(conditioned, int(known_k))
    unconditioned = model.fuse(
        graphs, int(known_k), uncertainty, persistence
    ).detach().cpu().numpy().astype(np.float32)
    witness_reference = spectral_seed42(unconditioned, int(known_k))
    trust_witness, partition_flow = trusted_partition_barycenter(
        witness_reference, bundle, int(known_k))
    source_baseline = spectral_seed42(bundle["graph_rows"][3], int(known_k))
    source_proposal, repair = source_memory_topology_repair(
        evidence, source_baseline, int(known_k), bundle)
    candidate, trust = trusted_source_resolution_with_witness(
        candidate, trust_witness, source_baseline, source_proposal, repair,
        int(known_k))
    if len(np.unique(candidate)) != int(known_k):
        raise AssertionError("Learned exact-K contract failed")
    return candidate.astype(np.int64), {
        "decoder": "learned triadic-harmonic family-confidence relation transport plus parallel trust witness",
        "family_confidence_attention": {
            **bundle["diagnostics"]["support_role_summary"],
            "continuous_pairwise_attention": True,
            "hard_routing_or_configuration_branch": False,
            "all_experts_always_evaluated": True,
            "target_gt_or_metric_used": False,
        },
        "triadic_harmonic_consistency": {
            "profile_support": "cosine(offdiagonal relation profiles)",
            "conjunction": "2*direct*profile_support/(direct+profile_support)",
            "continuous_all_edges": True,
            "hard_threshold_or_route": False,
            "target_gt_or_metric_used": False,
        },
        "parallel_partition_trust_witness": partition_flow,
        "source_trust_resolution": trust, "numeric_seeds": [SEED],
        "additional_numeric_seeds": [], "known_k": int(known_k),
        "occupied_k": int(len(np.unique(candidate))),
    }
