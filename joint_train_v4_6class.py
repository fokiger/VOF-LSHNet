#!/usr/bin/env python3
"""Train a six-class V4 model after dropping run_voltage and run_temperature."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hierarchical_predictor_v4 import (
    HierarchicalConsistencyLoss,
    HierarchicalCrossGatedPredictor,
    PredictorConfig,
)
from semantic_prototypes_v4 import (
    SemanticPrototypeBundle,
    load_semantic_prototypes,
)


DROP_ORIGINAL_CLASS_IDS = (4, 6)
KEEP_ORIGINAL_CLASS_IDS = (0, 1, 2, 3, 5, 7)
SIX_CLASS_NAMES = (
    "run_normal",
    "charge_normal",
    "run_insulation",
    "charge_insulation",
    "charge_voltage",
    "charge_temperature",
)
ORIGINAL_TO_SIX_CLASS = {
    original: remapped
    for remapped, original in enumerate(KEEP_ORIGINAL_CLASS_IDS)
}
SIX_CLASS_TO_STATE = (0, 1, 0, 1, 1, 1)
SIX_CLASS_TO_FAULT = (0, 0, 1, 1, 2, 3)
SIX_CLASS_TO_NORMAL_FAULT = (0, 0, 1, 1, 1, 1)
SIX_RUN_CLASS_IDS = (0, 2)
SIX_CHARGE_CLASS_IDS = (1, 3, 4, 5)
NORMAL_CLASS_IDS = np.asarray((0, 1), dtype=np.int64)
FAULT_CLASS_IDS = np.asarray((2, 3, 4, 5), dtype=np.int64)
INSULATION_CLASS_IDS = np.asarray((2, 3), dtype=np.int64)


@dataclass
class TrainConfig:
    cache_dir: str = "./v4_feature_cache"
    semantic_prototype_path: str = "./semantic_prototypes_v4.npz"
    output_dir: str = "./semantic_training_outputs_v4_6class"
    epochs: int = 40
    batch_size: int = 64
    num_workers: int = 0
    early_stopping_patience: int = 0
    lr_scheduler_patience: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    label_smoothing: float = 0.03
    state_loss_weight: float = 0.20
    fault_loss_weight: float = 0.35
    normal_fault_loss_weight: float = 0.35
    grouped_normal_fault_loss_weight: float = 0.20
    consistency_loss_weight: float = 0.10
    alignment_loss_weight: float = 0.25
    structured_margin_weight: float = 0.0
    composition_loss_weight: float = 0.05
    structured_margin: float = 0.10
    disable_semantic_prototypes: bool = False
    disable_class_semantic_prototypes: bool = False
    disable_compositional_prototypes: bool = False
    disable_raw_branch: bool = False
    disable_wave_branch: bool = False
    disable_normal_fault_head: bool = False
    semantic_hierarchy_scale: float = 0.5
    normal_fault_hierarchy_scale: float = 0.5
    disable_state_condition: bool = True
    disable_state_logit_mask: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remap_original_labels(original_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(original_labels, dtype=np.int64)
    dropped = np.isin(labels, DROP_ORIGINAL_CLASS_IDS)
    if np.any(dropped):
        raise ValueError(
            "run_voltage/run_temperature labels must be filtered before remap"
        )
    remapped = np.empty_like(labels)
    for original, target in ORIGINAL_TO_SIX_CLASS.items():
        remapped[labels == original] = target
    return remapped


def mapped_labels_for_split(cache_dir: Path, split: str) -> np.ndarray:
    labels = np.load(cache_dir / f"labels_{split}.npy", mmap_mode="r")
    keep = np.isin(np.asarray(labels), KEEP_ORIGINAL_CLASS_IDS)
    return remap_original_labels(np.asarray(labels)[keep])


class SixClassCachedDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str):
        self.raw = np.load(cache_dir / f"raw_{split}.npy", mmap_mode="r")
        self.wave = np.load(cache_dir / f"wave_{split}.npy", mmap_mode="r")
        self.original_labels = np.load(
            cache_dir / f"labels_{split}.npy", mmap_mode="r"
        )
        if not (len(self.raw) == len(self.wave) == len(self.original_labels)):
            raise ValueError(f"cache length mismatch for split={split}")
        self.indices = np.flatnonzero(
            np.isin(np.asarray(self.original_labels), KEEP_ORIGINAL_CLASS_IDS)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_index = int(self.indices[index])
        original_label = int(self.original_labels[source_index])
        label = ORIGINAL_TO_SIX_CLASS[original_label]
        return (
            torch.tensor(self.raw[source_index], dtype=torch.float32),
            torch.tensor(self.wave[source_index], dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def make_loaders(
    cache_dir: Path, cfg: TrainConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for split in ("train", "val", "test"):
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        loaders.append(
            DataLoader(
                SixClassCachedDataset(cache_dir, split),
                batch_size=cfg.batch_size,
                shuffle=split == "train",
                num_workers=cfg.num_workers,
                pin_memory=cfg.device.startswith("cuda"),
                persistent_workers=cfg.num_workers > 0,
                generator=generator,
            )
        )
    return tuple(loaders)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=6).astype(np.float64)
    weights = len(labels) / (6.0 * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def normal_fault_weights(
    labels: np.ndarray, device: torch.device
) -> torch.Tensor:
    binary_labels = np.isin(labels, FAULT_CLASS_IDS).astype(np.int64)
    counts = np.bincount(binary_labels, minlength=2).astype(np.float64)
    weights = len(binary_labels) / (2.0 * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_criterion(
    labels: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> HierarchicalConsistencyLoss:
    return HierarchicalConsistencyLoss(
        class_weights=class_weights(labels, device),
        normal_fault_weights=normal_fault_weights(labels, device),
        label_smoothing=cfg.label_smoothing,
        state_weight=cfg.state_loss_weight,
        fault_weight=cfg.fault_loss_weight,
        normal_fault_weight=(
            0.0
            if cfg.disable_normal_fault_head
            else cfg.normal_fault_loss_weight
        ),
        grouped_normal_fault_weight=cfg.grouped_normal_fault_loss_weight,
        consistency_weight=cfg.consistency_loss_weight,
        alignment_weight=cfg.alignment_loss_weight,
        structured_margin_weight=cfg.structured_margin_weight,
        composition_weight=(
            0.0
            if cfg.disable_compositional_prototypes
            else cfg.composition_loss_weight
        ),
        structured_margin=cfg.structured_margin,
        class_to_state=SIX_CLASS_TO_STATE,
        class_to_fault=SIX_CLASS_TO_FAULT,
        class_to_normal_fault=SIX_CLASS_TO_NORMAL_FAULT,
    ).to(device)


def binary_detection_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "normal_to_fault_rate": float(fp / max(fp + tn, 1)),
        "fault_to_normal_rate": float(fn / max(fn + tp, 1)),
    }


def calculate_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    class_ids = np.arange(len(SIX_CLASS_NAMES))
    aucs = []
    auc_per_class = {}
    for class_id, name in enumerate(SIX_CLASS_NAMES):
        binary_labels = (labels == class_id).astype(np.int64)
        if len(np.unique(binary_labels)) > 1:
            class_auc = float(roc_auc_score(binary_labels, probs[:, class_id]))
            aucs.append(class_auc)
            auc_per_class[name] = class_auc
        else:
            auc_per_class[name] = None

    precision = precision_score(
        labels, preds, labels=class_ids, average=None, zero_division=0
    )
    recall = recall_score(
        labels, preds, labels=class_ids, average=None, zero_division=0
    )
    f1 = f1_score(
        labels, preds, labels=class_ids, average=None, zero_division=0
    )
    matrix = confusion_matrix(labels, preds, labels=class_ids)

    normal_support = int(matrix[NORMAL_CLASS_IDS, :].sum())
    fault_support = int(matrix[FAULT_CLASS_IDS, :].sum())
    binary_tp = int(matrix[np.ix_(FAULT_CLASS_IDS, FAULT_CLASS_IDS)].sum())
    binary_fp = int(matrix[np.ix_(NORMAL_CLASS_IDS, FAULT_CLASS_IDS)].sum())
    binary_tn = int(matrix[np.ix_(NORMAL_CLASS_IDS, NORMAL_CLASS_IDS)].sum())
    binary_fn = int(matrix[np.ix_(FAULT_CLASS_IDS, NORMAL_CLASS_IDS)].sum())
    binary_precision = (
        binary_tp / (binary_tp + binary_fp) if binary_tp + binary_fp else 0.0
    )
    binary_recall = (
        binary_tp / (binary_tp + binary_fn) if binary_tp + binary_fn else 0.0
    )
    binary_specificity = (
        binary_tn / (binary_tn + binary_fp) if binary_tn + binary_fp else 0.0
    )
    binary_f1 = (
        2.0 * binary_precision * binary_recall
        / (binary_precision + binary_recall)
        if binary_precision + binary_recall
        else 0.0
    )
    normal_to_insulation = int(
        matrix[np.ix_(NORMAL_CLASS_IDS, INSULATION_CLASS_IDS)].sum()
    )

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "balanced_accuracy": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "auc_macro": float(np.mean(aucs)) if aucs else 0.5,
        "minimum_class_recall": float(recall.min()),
        "precision_per_class": dict(
            zip(SIX_CLASS_NAMES, precision.tolist())
        ),
        "recall_per_class": dict(zip(SIX_CLASS_NAMES, recall.tolist())),
        "f1_per_class": dict(zip(SIX_CLASS_NAMES, f1.tolist())),
        "auc_per_class": auc_per_class,
        "confusion_matrix": matrix.tolist(),
        "normal_to_insulation": normal_to_insulation,
        "normal_to_fault": binary_fp,
        "fault_to_normal": binary_fn,
        "normal_to_insulation_rate": (
            float(normal_to_insulation / normal_support)
            if normal_support
            else 0.0
        ),
        "normal_to_fault_rate": (
            float(binary_fp / normal_support) if normal_support else 0.0
        ),
        "fault_to_normal_rate": (
            float(binary_fn / fault_support) if fault_support else 0.0
        ),
        "binary_fault_detection": {
            "tp": binary_tp,
            "fp": binary_fp,
            "tn": binary_tn,
            "fn": binary_fn,
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "specificity": float(binary_specificity),
            "f1": float(binary_f1),
        },
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    binary_f1 = float(metrics["binary_fault_detection"]["f1"])
    return float(
        0.35 * metrics["f1_macro"]
        + 0.20 * metrics["auc_macro"]
        + 0.15 * metrics["balanced_accuracy"]
        + 0.15 * metrics["minimum_class_recall"]
        + 0.15 * binary_f1
        - 0.20 * metrics["normal_to_fault_rate"]
        - 0.15 * metrics["fault_to_normal_rate"]
    )


def metrics_to_json(metrics: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    result = {key: convert(value) for key, value in metrics.items()}
    result["selection_score"] = score_metrics(metrics)
    return result


def state_from_labels(labels: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(
        SIX_CLASS_TO_STATE, dtype=torch.long, device=labels.device
    )
    return mapping[labels]


def accumulate_weighted(
    totals: dict[str, float],
    metrics: dict[str, float],
    weight: int,
) -> None:
    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + float(value) * weight


def semantic_parameter_metrics(model: torch.nn.Module) -> dict[str, float]:
    if not model.config.use_semantic_prototypes:
        return {}
    return {
        "prototype_logit_scale": float(
            model.logit_scale.detach().exp().clamp(max=100.0).item()
        ),
        "state_logit_scale": float(
            model.state_logit_scale.detach().exp().clamp(max=100.0).item()
        ),
        "fault_logit_scale": float(
            model.fault_logit_scale.detach().exp().clamp(max=100.0).item()
        ),
        "semantic_hierarchy_scale": float(
            model.config.semantic_hierarchy_scale
        ),
    }


def semantic_batch_diagnostics(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, float]:
    if "prototype_logits" not in output:
        return {}
    prototype_logits = output["prototype_logits"].detach().float()
    hierarchy_logits = output["semantic_hierarchy_logits"].detach().float()
    prototype_abs = prototype_logits.abs().mean()
    hierarchy_abs = hierarchy_logits.abs().mean()
    semantic_abs = prototype_abs + hierarchy_abs + 1e-8
    diagnostics = {
        "prototype_logit_abs_mean": float(prototype_abs.item()),
        "semantic_hierarchy_logit_abs_mean": float(hierarchy_abs.item()),
        "prototype_logit_fraction": float(
            (prototype_abs / semantic_abs).item()
        ),
        "semantic_hierarchy_logit_fraction": float(
            (hierarchy_abs / semantic_abs).item()
        ),
        "prototype_only_accuracy": float(
            (prototype_logits.argmax(dim=1) == labels).float().mean().item()
        ),
    }
    if hierarchy_logits.abs().sum().item() > 0:
        diagnostics["semantic_hierarchy_only_accuracy"] = float(
            (hierarchy_logits.argmax(dim=1) == labels).float().mean().item()
        )
    if "prototype_similarity" in output:
        similarity = output["prototype_similarity"].detach().float()
        positive = similarity.gather(1, labels[:, None]).squeeze(1)
        negative_mask = torch.ones_like(similarity, dtype=torch.bool)
        negative_mask.scatter_(1, labels[:, None], False)
        negative = similarity.masked_fill(~negative_mask, float("-inf")).amax(
            dim=1
        )
        diagnostics["prototype_positive_similarity"] = float(
            positive.mean().item()
        )
        diagnostics["prototype_best_negative_similarity"] = float(
            negative.mean().item()
        )
        diagnostics["prototype_similarity_margin"] = float(
            (positive - negative).mean().item()
        )
    if "class_semantic_weight" in output:
        diagnostics["class_semantic_weight_mean"] = float(
            output["class_semantic_weight"].detach().float().mean().item()
        )
    if "composed_semantic_weight" in output:
        diagnostics["composed_semantic_weight_mean"] = float(
            output["composed_semantic_weight"].detach().float().mean().item()
        )
    return diagnostics


def diagnostic_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    prefixes = (
        "fusion_gate",
        "prototype_",
        "semantic_",
        "class_semantic_",
        "composed_semantic_",
        "state_logit_scale",
        "fault_logit_scale",
    )
    result = {}
    for key, value in metrics.items():
        if key.startswith(prefixes) and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            result[key] = float(value)
    return result


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: HierarchicalConsistencyLoss,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0}
    sample_count = 0
    progress = tqdm(
        loader,
        desc="train",
        ncols=100,
        leave=False,
        disable=os.environ.get("V4_DISABLE_TQDM", "0") == "1",
    )
    for raw, wave, labels in progress:
        raw = raw.to(device, non_blocking=True)
        wave = wave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        operating_state = state_from_labels(labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(raw, wave, operating_state)
            loss, components = criterion(output, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("predictor produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        batch_size = len(labels)
        sample_count += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        accumulate_weighted(
            totals,
            semantic_batch_diagnostics(output, labels),
            batch_size,
        )
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    metrics = {
        name: value / max(sample_count, 1)
        for name, value in totals.items()
    }
    metrics.update(semantic_parameter_metrics(model))
    return metrics


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: HierarchicalConsistencyLoss,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    probabilities = []
    labels_all = []
    gate_sum = 0.0
    prototype_correct = 0
    composition_sum = 0.0
    composition_count = 0
    diagnostic_totals: dict[str, float] = {}
    prototype_predictions = []
    semantic_hierarchy_predictions = []
    normal_fault_predictions = []
    for raw, wave, labels in loader:
        raw = raw.to(device, non_blocking=True)
        wave = wave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        operating_state = state_from_labels(labels)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(raw, wave, operating_state)
            loss, _ = criterion(output, labels)
        probability = torch.softmax(output["logits"].float(), dim=1)
        probabilities.append(probability.cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        total_loss += float(loss) * len(labels)
        total += len(labels)
        gate_sum += float(output["fusion_gate"].float().sum())
        accumulate_weighted(
            diagnostic_totals,
            semantic_batch_diagnostics(output, labels),
            len(labels),
        )
        if "prototype_logits" in output:
            prototype_correct += int(
                (output["prototype_logits"].argmax(dim=1) == labels).sum()
            )
            prototype_predictions.append(
                output["prototype_logits"].argmax(dim=1).cpu().numpy()
            )
            semantic_hierarchy_logits = output["semantic_hierarchy_logits"]
            if semantic_hierarchy_logits.float().abs().sum().item() > 0:
                semantic_hierarchy_predictions.append(
                    semantic_hierarchy_logits.argmax(dim=1).cpu().numpy()
                )
            composition_sum += float(
                output["composition_similarity"].float().sum()
            )
            composition_count += output["composition_similarity"].numel()
        if "normal_fault_logits" in output:
            normal_fault_predictions.append(
                output["normal_fault_logits"].argmax(dim=1).cpu().numpy()
            )

    probs = np.concatenate(probabilities)
    labels_numpy = np.concatenate(labels_all)
    metrics = calculate_metrics(labels_numpy, probs)
    metrics["loss"] = total_loss / max(total, 1)
    metrics["fusion_gate_mean"] = gate_sum / max(
        total * model.config.d_model, 1
    )
    for name, value in diagnostic_totals.items():
        metrics[name] = value / max(total, 1)
    metrics.update(semantic_parameter_metrics(model))
    if prototype_predictions:
        prototype_numpy = np.concatenate(prototype_predictions)
        metrics["prototype_only_accuracy"] = float(
            accuracy_score(labels_numpy, prototype_numpy)
        )
        metrics["prototype_only_f1_macro"] = float(
            f1_score(
                labels_numpy,
                prototype_numpy,
                labels=np.arange(len(SIX_CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        )
    if semantic_hierarchy_predictions:
        hierarchy_numpy = np.concatenate(semantic_hierarchy_predictions)
        metrics["semantic_hierarchy_only_accuracy"] = float(
            accuracy_score(labels_numpy, hierarchy_numpy)
        )
        metrics["semantic_hierarchy_only_f1_macro"] = float(
            f1_score(
                labels_numpy,
                hierarchy_numpy,
                labels=np.arange(len(SIX_CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        )
    if normal_fault_predictions:
        binary_labels = np.isin(labels_numpy, FAULT_CLASS_IDS).astype(np.int64)
        head_predictions = np.concatenate(normal_fault_predictions)
        metrics["normal_fault_head"] = binary_detection_metrics(
            binary_labels, head_predictions
        )
        final_binary_predictions = np.isin(
            probs.argmax(axis=1), FAULT_CLASS_IDS
        ).astype(np.int64)
        metrics["normal_fault_head_joint_agreement"] = float(
            np.mean(head_predictions == final_binary_predictions)
        )
    if model.config.use_semantic_prototypes:
        metrics["prototype_alignment_accuracy"] = (
            prototype_correct / max(total, 1)
        )
        metrics["prototype_composition_similarity"] = (
            composition_sum / max(composition_count, 1)
        )
    return metrics


def make_six_class_bundle(
    prototype_path: Path,
) -> SemanticPrototypeBundle:
    bundle = load_semantic_prototypes(prototype_path)
    keep_indices = torch.tensor(KEEP_ORIGINAL_CLASS_IDS, dtype=torch.long)
    metadata = copy.deepcopy(bundle.metadata)
    metadata["class_names"] = list(SIX_CLASS_NAMES)
    metadata["dropped_class"] = ["run_voltage", "run_temperature"]
    metadata["source_class_indices"] = list(KEEP_ORIGINAL_CLASS_IDS)
    return SemanticPrototypeBundle(
        state_embeddings=bundle.state_embeddings,
        fault_embeddings=bundle.fault_embeddings,
        class_embeddings=bundle.class_embeddings[keep_indices],
        metadata=metadata,
    )


def build_model(
    model_config: PredictorConfig,
    prototype_bundle: SemanticPrototypeBundle | None,
    device: torch.device,
) -> HierarchicalCrossGatedPredictor:
    if prototype_bundle is None:
        return HierarchicalCrossGatedPredictor(model_config).to(device)
    return HierarchicalCrossGatedPredictor(
        model_config,
        state_prototypes=prototype_bundle.state_embeddings,
        fault_prototypes=prototype_bundle.fault_embeddings,
        class_prototypes=prototype_bundle.class_embeddings,
    ).to(device)


def filtered_metadata(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(metadata)
    result["class_names"] = list(SIX_CLASS_NAMES)
    result["dropped_original_classes"] = [
        {"id": 4, "name": "run_voltage"},
        {"id": 6, "name": "run_temperature"},
    ]
    result["original_to_six_class"] = {
        str(original): remapped
        for original, remapped in ORIGINAL_TO_SIX_CLASS.items()
    }
    result["counts"] = {}
    result["class_counts"] = {}
    result["source_group_counts"] = {}
    manifest_path = cache_dir / "dataset_split_manifest.csv"
    manifest = (
        pd.read_csv(manifest_path)
        if manifest_path.exists()
        else None
    )
    for split in ("train", "val", "test"):
        labels = mapped_labels_for_split(cache_dir, split)
        result["counts"][split] = int(len(labels))
        result["class_counts"][split] = np.bincount(
            labels, minlength=6
        ).astype(int).tolist()
        if manifest is not None:
            split_manifest = manifest[
                (manifest["split"] == split)
                & manifest["class_id"].isin(KEEP_ORIGINAL_CLASS_IDS)
            ]
            result["source_group_counts"][split] = int(
                split_manifest["source_group_id"].nunique()
            )
    return result


def run(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    cache_dir = Path(cfg.cache_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"V4 cache not found: {metadata_path}. "
            "Run prepare_v4_cache.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("observation_points", 0)) != 64:
        raise ValueError("V4 requires a cache built from exactly 64 points")

    use_semantics = not cfg.disable_semantic_prototypes
    if (
        use_semantics
        and cfg.disable_class_semantic_prototypes
        and cfg.disable_compositional_prototypes
    ):
        raise ValueError(
            "cannot disable both class and compositional semantic prototypes "
            "while semantic prototypes are enabled"
        )
    prototype_bundle = None
    prototype_metadata = None
    if use_semantics:
        prototype_path = Path(cfg.semantic_prototype_path).resolve()
        if not prototype_path.exists():
            raise FileNotFoundError(
                f"semantic prototypes not found: {prototype_path}. "
                "Run semantic_prototypes_v4.py first."
            )
        prototype_bundle = make_six_class_bundle(prototype_path)
        prototype_metadata = prototype_bundle.metadata
        shutil.copy2(
            prototype_path, output_dir / "semantic_prototypes_v4_source.npz"
        )

    train_loader, val_loader, test_loader = make_loaders(cache_dir, cfg)
    train_labels = mapped_labels_for_split(cache_dir, "train")
    semantic_dim = (
        prototype_bundle.embedding_dim if prototype_bundle is not None else 1
    )
    model_config = PredictorConfig(
        n_classes=6,
        wave_feature_dim=int(metadata["wave_feature_dim"]),
        semantic_dim=semantic_dim,
        use_semantic_prototypes=use_semantics,
        use_class_semantic_prototypes=(
            use_semantics and not cfg.disable_class_semantic_prototypes
        ),
        use_compositional_prototypes=(
            use_semantics and not cfg.disable_compositional_prototypes
        ),
        semantic_hierarchy_scale=(
            0.0
            if cfg.disable_compositional_prototypes
            else cfg.semantic_hierarchy_scale
        ),
        use_raw_branch=not cfg.disable_raw_branch,
        use_wave_branch=not cfg.disable_wave_branch,
        use_normal_fault_head=not cfg.disable_normal_fault_head,
        normal_fault_hierarchy_scale=cfg.normal_fault_hierarchy_scale,
        use_state_condition=not cfg.disable_state_condition,
        use_state_logit_mask=not cfg.disable_state_logit_mask,
        class_to_state=SIX_CLASS_TO_STATE,
        class_to_fault=SIX_CLASS_TO_FAULT,
        class_to_normal_fault=SIX_CLASS_TO_NORMAL_FAULT,
        state_logit_run_class_ids=SIX_RUN_CLASS_IDS,
        state_logit_charge_class_ids=SIX_CHARGE_CLASS_IDS,
    )
    device = torch.device(cfg.device)
    model = build_model(model_config, prototype_bundle, device)
    criterion = make_criterion(train_labels, cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=cfg.lr_scheduler_patience,
        factor=0.5,
        min_lr=1e-6,
    )

    best_state = None
    best_metrics = None
    best_score = -math.inf
    stale_epochs = 0
    early_stopping_enabled = cfg.early_stopping_patience > 0
    if not early_stopping_enabled:
        print("Early stopping disabled; training will run for all configured epochs.")
    curves = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, cfg, device
        )
        validation = evaluate(model, val_loader, criterion, device)
        score = score_metrics(validation)
        improved = score > best_score + 1e-4
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = copy.deepcopy(validation)
        stale_epochs = 0 if improved else stale_epochs + 1
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "val_loss": validation["loss"],
            "val_f1_macro": validation["f1_macro"],
            "val_auc_macro": validation["auc_macro"],
            "val_minimum_class_recall": validation["minimum_class_recall"],
            "val_score": score,
        }
        if "prototype_alignment_accuracy" in validation:
            row["val_prototype_alignment_accuracy"] = validation[
                "prototype_alignment_accuracy"
            ]
        row.update(
            {
                f"val_{key}": value
                for key, value in diagnostic_scalar_metrics(
                    validation
                ).items()
            }
        )
        if "normal_fault_head" in validation:
            row["val_normal_fault_head_f1"] = validation[
                "normal_fault_head"
            ]["f1"]
            row["val_normal_fault_head_normal_to_fault_rate"] = validation[
                "normal_fault_head"
            ]["normal_to_fault_rate"]
        curves.append(row)
        pd.DataFrame(curves).to_csv(
            output_dir / "training_curves_v4_6class.csv",
            index=False,
            encoding="utf-8-sig",
        )
        scheduler.step(score)
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.4f} "
            f"val_f1={validation['f1_macro']:.4f} score={score:.5f}"
        )
        if early_stopping_enabled and stale_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "best_predictor_v4_6class.pt")
    validation_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    result = {
        "method": {
            "name": (
                "CC-VOFRWT six-class model without run_voltage/run_temperature + "
                "compositional LLM mechanism prototypes without "
                "structured margin loss"
            ),
            "observation_points": 64,
            "future_prediction": True,
            "dropped_class": ["run_voltage", "run_temperature"],
            "semantic_prototypes": use_semantics,
            "class_semantic_prototypes": (
                model_config.use_class_semantic_prototypes
            ),
            "compositional_prototypes": (
                model_config.use_compositional_prototypes
            ),
            "normal_fault_head": model_config.use_normal_fault_head,
            "semantic_hierarchy_scale": model_config.semantic_hierarchy_scale,
            "normal_fault_hierarchy_scale": (
                model_config.normal_fault_hierarchy_scale
            ),
            "structured_margin_loss": cfg.structured_margin_weight > 0.0,
            "state_condition": model_config.use_state_condition,
            "state_logit_mask": model_config.use_state_logit_mask,
        },
        "train_config": asdict(cfg),
        "predictor_config": asdict(model_config),
        "cache_metadata": filtered_metadata(cache_dir, metadata),
        "prototype_metadata": prototype_metadata,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "validation": metrics_to_json(validation_metrics),
        "test": metrics_to_json(test_metrics),
    }
    (output_dir / "result_v4_6class.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["test"], ensure_ascii=False, indent=2))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="./v4_feature_cache")
    parser.add_argument(
        "--semantic-prototype-path",
        default="./semantic_prototypes_v4.npz",
    )
    parser.add_argument("--output-dir", default="./semantic_training_outputs_v4_6class")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Validation patience for early stopping; set <=0 to disable.",
    )
    parser.add_argument("--lr-scheduler-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--state-loss-weight", type=float, default=0.20)
    parser.add_argument("--fault-loss-weight", type=float, default=0.35)
    parser.add_argument(
        "--normal-fault-loss-weight", type=float, default=0.35
    )
    parser.add_argument(
        "--grouped-normal-fault-loss-weight", type=float, default=0.20
    )
    parser.add_argument(
        "--consistency-loss-weight", type=float, default=0.10
    )
    parser.add_argument("--alignment-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--structured-margin-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--composition-loss-weight", type=float, default=0.05
    )
    parser.add_argument("--structured-margin", type=float, default=0.10)
    parser.add_argument("--disable-semantic-prototypes", action="store_true")
    parser.add_argument(
        "--disable-class-semantic-prototypes", action="store_true"
    )
    parser.add_argument(
        "--disable-compositional-prototypes", action="store_true"
    )
    parser.add_argument("--disable-raw-branch", action="store_true")
    parser.add_argument("--disable-wave-branch", action="store_true")
    parser.add_argument("--disable-normal-fault-head", action="store_true")
    parser.add_argument("--semantic-hierarchy-scale", type=float, default=0.5)
    parser.add_argument(
        "--normal-fault-hierarchy-scale", type=float, default=0.5
    )
    parser.add_argument(
        "--enable-state-condition",
        dest="disable_state_condition",
        action="store_false",
        help="Enable operating-state conditioning in the predictor.",
    )
    parser.add_argument(
        "--disable-state-condition",
        dest="disable_state_condition",
        action="store_true",
        help="Disable operating-state conditioning in the predictor.",
    )
    parser.add_argument(
        "--enable-state-logit-mask",
        dest="disable_state_logit_mask",
        action="store_false",
        help="Enable operating-state logit masking during training/evaluation.",
    )
    parser.add_argument(
        "--disable-state-logit-mask",
        dest="disable_state_logit_mask",
        action="store_true",
        help="Disable operating-state logit masking during training/evaluation.",
    )
    parser.set_defaults(
        disable_state_condition=True,
        disable_state_logit_mask=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
