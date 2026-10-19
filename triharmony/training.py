from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import torch

from .config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, LOSS_WEIGHTS, OPTIMIZER,
    SOURCE_BATCH_SIZE, SOURCE_CLASSES, SOURCE_ORTHOGONAL_RETENTION,
    SOURCE_TARGET_PARALLEL_FRACTION, TARGET_BATCH_SIZE, WEIGHT_DECAY,
)
from .model import (
    PersistentGraphTransport, learned_partition,
    source_graph_stack, source_relation_loss, target_capacity_loss,
    target_deletion_persistence_loss, target_graph_bundle,
    target_scale_normalized_heldout_loss,
)
from datetime import datetime, timezone
from . import config


def _state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def _cyclic_indices(count: int, start: int, length: int) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) + int(start)) % int(count)


def target_indices(sample_count: int, epoch: int, step: int) -> torch.Tensor:
    steps = int(math.ceil(float(sample_count) / float(TARGET_BATCH_SIZE)))
    if step >= steps:
        raise ValueError("Target step exceeds fixed schedule")
    start = ((int(epoch) - 1) + int(step) * TARGET_BATCH_SIZE) % int(sample_count)
    return _cyclic_indices(sample_count, start, TARGET_BATCH_SIZE)


def source_indices(labels: torch.Tensor, cursor: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    per_class, labels_cpu = SOURCE_BATCH_SIZE // SOURCE_CLASSES, labels.cpu().long()
    for offset in range(per_class):
        for class_id in range(SOURCE_CLASSES):
            available = torch.nonzero(labels_cpu == class_id, as_tuple=False).flatten()
            rows.append(available[(int(cursor) + offset) % len(available)])
    result = torch.stack(rows).long()
    if len(result) != SOURCE_BATCH_SIZE:
        raise AssertionError("Source batch-size contract changed")
    return result


def prepare_source(source: Dict[str, object], device: torch.device
                   ) -> Dict[str, object]:
    graphs = source_graph_stack(source["dense"], source["sparse"], device)
    labels = source["class_id"].long()
    labels_device = labels.to(device)
    mask = ~torch.eye(len(labels), dtype=torch.bool, device=device)
    same = labels_device[:, None].eq(labels_device[None]) & mask
    different = labels_device[:, None].ne(labels_device[None]) & mask
    same_weight = same.to(dtype=graphs.dtype)[:, :, None]
    different_weight = different.to(dtype=graphs.dtype)[:, :, None]
    quality = (
        (graphs * same_weight).sum(dim=1)
        / same_weight.sum(dim=1).clamp_min(1.0)
        - (graphs * different_weight).sum(dim=1)
        / different_weight.sum(dim=1).clamp_min(1.0)
    )
    return {
        "graphs": graphs,
        "uncertainty": (graphs - graphs.mean(dim=2, keepdim=True)).abs().detach(),
        "persistence": torch.sigmoid(quality).detach(),
        "labels": labels,
    }


def _gradient_geometry(model: PersistentGraphTransport, target_loss: torch.Tensor,
                       source_loss: torch.Tensor) -> Dict[str, float]:
    parameters = tuple(model.parameters())
    target_gradient = torch.autograd.grad(
        target_loss, parameters, retain_graph=True, allow_unused=True
    )
    source_gradient = torch.autograd.grad(source_loss, parameters, allow_unused=True)
    zero = target_loss.new_zeros(())
    raw_dot, target_sq, source_sq = zero.clone(), zero.clone(), zero.clone()
    transported_dot, transported_sq = zero.clone(), zero.clone()
    for parameter, target_row, source_row in zip(
            parameters, target_gradient, source_gradient):
        target_value = torch.zeros_like(parameter) if target_row is None else target_row
        source_value = torch.zeros_like(parameter) if source_row is None else source_row
        if not torch.isfinite(target_value).all() or not torch.isfinite(source_value).all():
            raise FloatingPointError("Non-finite gradient before transport")
        target_norm, source_norm = target_value.norm(), source_value.norm()
        raw_dot = raw_dot + (target_value * source_value).sum()
        target_sq = target_sq + target_value.square().sum()
        source_sq = source_sq + source_value.square().sum()
        if float(target_norm.detach()) > 0.0 and float(source_norm.detach()) > 0.0:
            target_unit = target_value / target_norm
            parallel = (source_value * target_unit).sum()
            orthogonal = source_value - parallel * target_unit
            orthogonal_norm = orthogonal.norm()
            if float(orthogonal_norm.detach()) > 0.0:
                direction = (SOURCE_TARGET_PARALLEL_FRACTION * target_unit
                             + SOURCE_ORTHOGONAL_RETENTION
                             * orthogonal / orthogonal_norm)
            else:
                direction = target_unit
            transported = torch.minimum(source_norm, target_norm) * direction
        else:
            transported = torch.zeros_like(parameter)
        parameter.grad = (target_value + transported).detach().clone()
        transported_dot = transported_dot + (target_value * transported).sum()
        transported_sq = transported_sq + transported.square().sum()
    target_norm_total, source_norm_total = target_sq.sqrt(), source_sq.sqrt()
    transported_norm_total = transported_sq.sqrt()
    if not torch.isfinite(target_norm_total) or float(target_norm_total.detach()) <= 0.0:
        raise RuntimeError("Target gradient saturated or became non-finite")
    raw_cosine = raw_dot / (
        target_norm_total * source_norm_total).clamp_min(1.0e-12)
    transported_cosine = transported_dot / (
        target_norm_total * transported_norm_total).clamp_min(1.0e-12)
    if float(transported_dot.detach()) < -1.0e-10:
        raise RuntimeError("Transported source update reduced target descent")
    if float(transported_norm_total.detach()) > float(target_norm_total.detach()) + 1.0e-6:
        raise RuntimeError("Transported source gradient exceeded target cap")
    return {
        "target_gradient_norm": float(target_norm_total.detach()),
        "source_gradient_norm": float(source_norm_total.detach()),
        "raw_source_target_cosine": float(raw_cosine.detach()),
        "transported_source_gradient_norm": float(transported_norm_total.detach()),
        "transported_source_target_cosine": float(transported_cosine.detach()),
        "transported_source_target_dot": float(transported_dot.detach()),
    }


def train_one(prepared_source: Dict[str, object], target: Dict[str, object],
              device: torch.device
              ) -> Tuple[PersistentGraphTransport, Dict[str, object],
                         Dict[str, object]]:
    if OPTIMIZER != "AdamW" or BATCH_SIZE != 64:
        raise RuntimeError("Fixed optimizer/batch contract changed")
    evidence = {
        str(name): value.detach().cpu().numpy().astype("float32")
        for name, value in target["evidence"].items()
    }
    known_k = int(target["known_k"])
    anomaly_support = target["anomaly_support"].detach().cpu().numpy().astype(
        "float32"
    )
    bundle = target_graph_bundle(evidence, known_k, anomaly_support)
    target_graphs = bundle["graphs"].to(device)
    target_uncertainty = bundle["uncertainty"].to(device)
    target_persistence = bundle["persistence"].to(device)
    target_support_role = bundle["support_role"].to(device)
    target_views = bundle["views"].to(device)
    source_graphs = prepared_source["graphs"]
    source_uncertainty = prepared_source["uncertainty"]
    source_persistence = prepared_source["persistence"]
    source_labels = prepared_source["labels"]
    model = PersistentGraphTransport().to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial_state = _state(model)
    started_at, clock = utc_now(), time.time()
    trace: List[Dict[str, object]] = []
    total_steps, nonzero_gradient_steps, source_cursor = 0, 0, 0
    first_geometry, last_geometry = None, None
    minimum_target_gradient, minimum_transported_cosine = float("inf"), float("inf")
    for epoch in range(1, EPOCHS + 1):
        steps = int(math.ceil(len(target["keys"]) / TARGET_BATCH_SIZE))
        running = {name: 0.0 for name in LOSS_WEIGHTS}
        running_geometry: Dict[str, float] = {}
        for step in range(steps):
            target_index = target_indices(len(target["keys"]), epoch, step).to(device)
            source_index_cpu = source_indices(source_labels, source_cursor)
            source_index = source_index_cpu.to(device)
            source_cursor += SOURCE_BATCH_SIZE // SOURCE_CLASSES
            labels = source_labels.index_select(0, source_index_cpu).to(device)
            source_batch = source_graphs.index_select(0, source_index).index_select(
                1, source_index)
            source_uncertainty_batch = source_uncertainty.index_select(
                0, source_index).index_select(1, source_index)
            source_persistence_batch = source_persistence.index_select(
                0, source_index)
            target_batch = target_graphs.index_select(0, target_index).index_select(
                1, target_index)
            target_uncertainty_batch = target_uncertainty.index_select(
                0, target_index).index_select(1, target_index)
            target_persistence_batch = target_persistence.index_select(
                0, target_index)
            target_support_role_batch = target_support_role.index_select(
                0, target_index)
            view_index = ((int(epoch) - 1) * steps + int(step)) % int(
                target_views.shape[0])
            target_deletion_batch = target_views[view_index].index_select(
                0, target_index).index_select(1, target_index)
            losses = {
                "source_supervised_relation": source_relation_loss(
                    model, source_batch, labels, source_uncertainty_batch,
                    source_persistence_batch),
                "target_scale_normalized_heldout_prediction":
                    target_scale_normalized_heldout_loss(
                        model, target_batch, target_uncertainty_batch,
                        target_persistence_batch, known_k,
                        target_support_role_batch),
                "target_deletion_persistence_transport":
                    target_deletion_persistence_loss(
                        model, target_batch, target_deletion_batch,
                        target_uncertainty_batch, target_persistence_batch,
                        known_k, target_support_role_batch),
                "target_known_k_spectral_capacity": target_capacity_loss(
                    model, target_batch, target_uncertainty_batch,
                    target_persistence_batch, known_k,
                    target_support_role_batch),
            }
            if tuple(losses) != tuple(LOSS_WEIGHTS):
                raise AssertionError("Four-loss contract changed")
            source_loss = losses["source_supervised_relation"]
            target_loss = sum(
                LOSS_WEIGHTS[name] * losses[name] for name in LOSS_WEIGHTS
                if name != "source_supervised_relation"
            )
            if not torch.isfinite(source_loss + target_loss):
                raise FloatingPointError(
                    f"Non-finite loss epoch={epoch} step={step}: "
                    f"{ {name: float(value.detach()) for name, value in losses.items()} }"
                )
            optimizer.zero_grad(set_to_none=True)
            geometry = _gradient_geometry(model, target_loss, source_loss)
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("Non-finite parameter after AdamW step")
            total_steps += 1
            nonzero_gradient_steps += int(geometry["target_gradient_norm"] > 0.0)
            minimum_target_gradient = min(
                minimum_target_gradient, geometry["target_gradient_norm"])
            minimum_transported_cosine = min(
                minimum_transported_cosine,
                geometry["transported_source_target_cosine"])
            if first_geometry is None:
                first_geometry = dict(geometry)
            last_geometry, running_geometry = dict(geometry), geometry
            for name, value in losses.items():
                running[name] += float(value.detach())
        with torch.no_grad():
            responsibility = model.components(
                target_batch, known_k, target_uncertainty_batch,
                target_persistence_batch,
                role_prior=target_support_role_batch)["node_weight"].mean(dim=0)
        trace.append({
            "epoch": epoch,
            "loss": {name: value / steps for name, value in running.items()},
            "gradient_geometry_last_step": running_geometry,
            "base_reliability": torch.softmax(
                model.base_reliability.detach(), dim=0).cpu().tolist(),
            "mean_expert_responsibility": responsibility.cpu().tolist(),
            "uncertainty_gain": model.uncertainty_gain.detach().cpu().tolist(),
            "persistence_gain": model.persistence_gain.detach().cpu().tolist(),
        })
    final_state = _state(model)
    parameter_change = {
        name: float((final_state[name] - initial_state[name]).norm())
        for name in initial_state
    }
    if not all(value > 0.0 for value in parameter_change.values()):
        raise RuntimeError(f"A trainable parameter did not change: {parameter_change}")
    learned_prediction, learned_decoder = learned_partition(
        model, bundle, evidence, known_k, device
    )
    training = {
        "status": "complete", "started_at": started_at, "ended_at": utc_now(),
        "elapsed_seconds": time.time() - clock, "epochs": EPOCHS,
        "optimizer": OPTIMIZER, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "source_batch_size": SOURCE_BATCH_SIZE, "target_batch_size": TARGET_BATCH_SIZE,
        "optimizer_steps": total_steps,
        "relation_module_total_parameters": total_parameters,
        "relation_module_trainable_parameters": trainable_parameters,
        "frozen_descriptor_extractor_resident_during_training": False,
        "loss_weights": dict(LOSS_WEIGHTS), "trace": trace,
        "first_gradient_geometry": first_geometry,
        "last_gradient_geometry": last_geometry,
        "minimum_target_gradient_norm": minimum_target_gradient,
        "minimum_transported_source_target_cosine": minimum_transported_cosine,
        "target_gradient_nonzero_steps": nonzero_gradient_steps,
        "target_gradient_nonzero_every_step": nonzero_gradient_steps == total_steps,
        "parameter_change_norm": parameter_change,
        "all_parameter_groups_changed": all(
            value > 0.0 for value in parameter_change.values()),
        "universal_graph_diagnostics": bundle["diagnostics"],
        "continuous_family_confidence_attention_in_all_target_losses": True,
        "family_aware_auxiliary_uncertainty_witness": True,
        "continuous_triadic_harmonic_consistency_in_fused_losses": True,
        "hard_routing_or_configuration_branch": False,
        **seed_contract(),
    }
    common = {
        "known_k": known_k, "target_gt_used": False,
        "target_metric_used": False, "numeric_seeds_used_by_decoder": [42],
        "additional_numeric_seeds_used": [],
    }
    learned = {
        **common, "prediction": learned_prediction.tolist(),
        "occupied_k": int(len(set(learned_prediction.tolist()))),
        "decoder": learned_decoder,
    }
    return model, training, learned


def utc_now():
    return datetime.now(timezone.utc).isoformat()

def seed_contract():
    return {"seed":config.SEED,"sole_numeric_seed":config.SEED,"additional_numeric_seeds":[]}
