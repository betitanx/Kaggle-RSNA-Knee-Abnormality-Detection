"""Fusão de probabilidades sem treinar um meta-modelo sobre dados in-sample."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import macro_roc_auc


@dataclass(frozen=True)
class OOFFusionEvaluation:
    weights: np.ndarray
    calibration_scores: dict[str, float]
    evaluation_score: float
    evaluation_per_label: dict[str, float]
    condition_scores: tuple[float, ...]
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray


@dataclass(frozen=True)
class OOFTemperatureEvaluation:
    text_temperatures: np.ndarray
    image_temperatures: np.ndarray
    calibration_log_losses: dict[str, dict[str, float]]
    evaluation_score: float
    evaluation_per_label: dict[str, float]
    condition_scores: tuple[float, ...]
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray


@dataclass(frozen=True)
class OOFAvailabilityEvaluation:
    evaluation_score: float
    evaluation_per_label: dict[str, float]
    condition_scores: tuple[float, ...]
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray


@dataclass(frozen=True)
class OOFCrossfitEvaluation:
    weights: np.ndarray
    fold_weights: np.ndarray
    calibration_scores: tuple[dict[str, float], ...]
    predictions: np.ndarray
    evaluation_score: float
    evaluation_per_label: dict[str, float]
    condition_scores: tuple[float, ...]
    calibration_indices: tuple[np.ndarray, ...]
    evaluation_indices: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class OOFEvidenceEvaluation:
    predictions: np.ndarray
    weights: np.ndarray
    evaluation_score: float
    evaluation_per_label: dict[str, float]
    condition_scores: tuple[float, ...]
    evaluation_indices: np.ndarray


@dataclass(frozen=True)
class OOFRankEvaluation:
    predictions: np.ndarray
    evaluation_score: float
    evaluation_per_label: dict[str, float]
    condition_scores: tuple[float, ...]
    evaluation_indices: np.ndarray


@dataclass(frozen=True)
class OOFLogitEvaluation:
    predictions: np.ndarray
    evaluation_score: float
    evaluation_per_label: dict[str, float]
    condition_scores: tuple[float, ...]
    evaluation_indices: np.ndarray


def _validate_probabilities(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError(f"{name} precisa conter probabilidades finitas em [0, 1].")
    return matrix


def _validate_temperature_grid(grid: Sequence[float]) -> tuple[float, ...]:
    candidate_grid = tuple(float(temperature) for temperature in grid)
    if not candidate_grid or any(not np.isfinite(temperature) or temperature <= 0 for temperature in candidate_grid):
        raise ValueError("grid de temperatura precisa conter valores finitos positivos.")
    return candidate_grid


def _validate_tie_weight(tie_weight: float) -> float:
    resolved = float(tie_weight)
    if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("tie_weight precisa ser finito e estar em [0, 1].")
    return resolved


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return np.log(clipped) - np.log1p(-clipped)


def apply_temperature(predictions: np.ndarray, temperatures: Sequence[float]) -> np.ndarray:
    """Aplica temperatura por rótulo às probabilidades sem alterar sua forma."""
    matrix = _validate_probabilities("predictions", predictions)
    resolved = np.asarray(temperatures, dtype=float)
    if resolved.shape != (matrix.shape[1],) or not np.isfinite(resolved).all() or (resolved <= 0).any():
        raise ValueError("temperatures precisa ter um valor finito positivo por rótulo.")
    logits = _logit(matrix) / resolved[None, :]
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(calibrated, dtype=float)


def _binary_log_loss(truth: np.ndarray, predictions: np.ndarray) -> float:
    clipped = np.clip(predictions, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped)))


def calibrate_temperatures(
    y_true: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
    *,
    grid: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
) -> tuple[np.ndarray, dict[str, float]]:
    """Escolhe temperatura por rótulo minimizando log-loss na dobra de calibração."""
    truth = np.asarray(y_true, dtype=float)
    matrix = _validate_probabilities("predictions", predictions)
    if truth.ndim != 2 or truth.shape != matrix.shape or len(label_names) != matrix.shape[1]:
        raise ValueError("y_true, predictions e label_names precisam ter formas compatíveis.")
    candidate_grid = _validate_temperature_grid(grid)
    temperatures = np.ones(matrix.shape[1], dtype=float)
    scores: dict[str, float] = {}
    for index, label in enumerate(label_names):
        valid = np.isfinite(truth[:, index])
        if np.unique(truth[valid, index]).size < 2:
            continue
        best_temperature = 1.0
        best_loss = np.inf
        for temperature in candidate_grid:
            calibrated = apply_temperature(matrix[valid, index : index + 1], (temperature,))[:, 0]
            loss = _binary_log_loss(truth[valid, index], calibrated)
            if loss < best_loss - 1e-12 or (
                abs(loss - best_loss) <= 1e-12
                and abs(temperature - 1.0) < abs(best_temperature - 1.0)
            ):
                best_temperature = temperature
                best_loss = loss
        temperatures[index] = best_temperature
        scores[str(label)] = float(best_loss)
    if not scores:
        raise ValueError("Nenhum rótulo possui duas classes observáveis para calibrar temperaturas.")
    return temperatures, scores


def _oof_calibration_partition(folds: Sequence[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    if len(folds) < 2:
        raise ValueError("OOF exige pelo menos duas dobras para calibrar e avaliar.")
    calibration_fold_count = max(1, len(folds) // 2)
    calibration_indices = np.asarray(
        sorted(index for fold in folds[:calibration_fold_count] for index in fold["validacao_indices"]),
        dtype=int,
    )
    evaluation_indices = np.asarray(
        sorted(index for fold in folds[calibration_fold_count:] for index in fold["validacao_indices"]),
        dtype=int,
    )
    if not len(calibration_indices) or not len(evaluation_indices):
        raise ValueError("OOF precisa conter estudos nas partes de calibração e avaliação.")
    return calibration_indices, evaluation_indices


def blend_predictions(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    weights: float | Sequence[float],
) -> np.ndarray:
    """Combina texto e imagem com um peso por rótulo."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if text.shape != image.shape:
        raise ValueError("As predições de texto e imagem precisam ter o mesmo formato.")
    resolved_weights = np.asarray(weights, dtype=float)
    if resolved_weights.ndim == 0:
        resolved_weights = np.full(text.shape[1], float(resolved_weights))
    if resolved_weights.shape != (text.shape[1],) or not np.isfinite(resolved_weights).all():
        raise ValueError("weights precisa ter um valor finito por rótulo.")
    if (resolved_weights < 0).any() or (resolved_weights > 1).any():
        raise ValueError("weights precisa estar em [0, 1].")
    return resolved_weights[None, :] * text + (1.0 - resolved_weights[None, :]) * image


def blend_predictions_by_availability(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    text_available: Sequence[bool],
    image_available: Sequence[bool],
) -> np.ndarray:
    """Combina ramos usando somente a modalidade observável em cada estudo."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if text.shape != image.shape:
        raise ValueError("As predições de texto e imagem precisam ter o mesmo formato.")
    text_mask = np.asarray(text_available, dtype=bool)
    image_mask = np.asarray(image_available, dtype=bool)
    expected_shape = (text.shape[0],)
    if text_mask.shape != expected_shape or image_mask.shape != expected_shape:
        raise ValueError("As máscaras de disponibilidade precisam ter uma entrada por estudo.")
    weights = np.full(text.shape[0], 0.5, dtype=float)
    weights[~text_mask & image_mask] = 0.0
    weights[text_mask & ~image_mask] = 1.0
    return weights[:, None] * text + (1.0 - weights[:, None]) * image


def _average_rank_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 1:
        raise ValueError("rank_normalize exige pelo menos um estudo.")
    ranks = np.empty_like(matrix, dtype=float)
    for column in range(matrix.shape[1]):
        order = np.argsort(matrix[:, column], kind="mergesort")
        ordered = matrix[order, column]
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end] == ordered[start]:
                end += 1
            ranks[order[start:end], column] = (start + 1 + end) / 2.0
            start = end
    return (ranks - 0.5) / matrix.shape[0]


def _reference_rank_normalize(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.shape[0] < 1:
        raise ValueError("reference_predictions exige pelo menos um estudo.")
    normalized = np.empty_like(matrix, dtype=float)
    reference_size = reference.shape[0]
    lower = 0.5 / reference_size
    upper = 1.0 - lower
    for column in range(matrix.shape[1]):
        ordered = np.sort(reference[:, column])
        left = np.searchsorted(ordered, matrix[:, column], side="left")
        right = np.searchsorted(ordered, matrix[:, column], side="right")
        average_rank = (left + right + 1.0) / 2.0
        normalized[:, column] = np.clip(
            (average_rank - 0.5) / reference_size,
            lower,
            upper,
        )
    return normalized


def rank_normalize_predictions(
    predictions: np.ndarray,
    reference_predictions: np.ndarray | None = None,
) -> np.ndarray:
    """Converte probabilidades em percentis, preservando somente a ordem."""
    matrix = _validate_probabilities("predictions", predictions)
    if reference_predictions is None:
        return _average_rank_normalize(matrix)
    reference = _validate_probabilities("reference_predictions", reference_predictions)
    if reference.shape[1] != matrix.shape[1]:
        raise ValueError("reference_predictions precisa ter os mesmos rótulos.")
    return _reference_rank_normalize(matrix, reference)


def blend_predictions_by_rank(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    *,
    text_reference: np.ndarray | None = None,
    image_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Combina percentis de texto e imagem sem depender da escala absoluta."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if text.shape != image.shape:
        raise ValueError("As predições de texto e imagem precisam ter o mesmo formato.")
    text_rank = rank_normalize_predictions(text, text_reference)
    image_rank = rank_normalize_predictions(image, image_reference)
    return 0.5 * (text_rank + image_rank)


def blend_predictions_by_rank_shrinkage(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    *,
    text_weight: float = 0.15,
    shrinkage: float = 0.25,
    text_reference: np.ndarray | None = None,
    image_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Mistura a escala original com ranking, retraída a um peso global."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if text.shape != image.shape:
        raise ValueError("As predições de texto e imagem precisam ter o mesmo formato.")
    if not np.isfinite(text_weight) or not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight precisa ser finito e estar em [0, 1].")
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage precisa ser finito e estar em [0, 1].")
    text_rank = rank_normalize_predictions(text, text_reference)
    image_rank = rank_normalize_predictions(image, image_reference)
    raw_blend = text_weight * text + (1.0 - text_weight) * image
    rank_blend = text_weight * text_rank + (1.0 - text_weight) * image_rank
    return (1.0 - shrinkage) * raw_blend + shrinkage * rank_blend


def blend_predictions_by_logit(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
) -> np.ndarray:
    """Combina ramos pela média de logits, ou média geométrica das odds."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if text.shape != image.shape:
        raise ValueError("As predições de texto e imagem precisam ter o mesmo formato.")
    logits = 0.5 * (_logit(text) + _logit(image))
    return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=float)


def _validate_availability_mask(name: str, values: Sequence[bool], expected_length: int) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if mask.shape != (expected_length,):
        raise ValueError(f"{name} precisa ter uma entrada por estudo.")
    return mask


def calculate_evidence_weights(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    text_available: Sequence[bool],
    image_available: Sequence[bool],
    *,
    evidence_power: float = 1.0,
) -> np.ndarray:
    """Calcula pesos por estudo usando confiança como proxy de evidência."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if text.shape != image.shape:
        raise ValueError("As predições de texto e imagem precisam ter o mesmo formato.")
    if not np.isfinite(evidence_power) or evidence_power <= 0:
        raise ValueError("evidence_power precisa ser finito e positivo.")
    text_mask = _validate_availability_mask("text_available", text_available, text.shape[0])
    image_mask = _validate_availability_mask("image_available", image_available, text.shape[0])
    text_evidence = np.abs(2.0 * text - 1.0) ** evidence_power
    image_evidence = np.abs(2.0 * image - 1.0) ** evidence_power
    text_evidence[~text_mask, :] = 0.0
    image_evidence[~image_mask, :] = 0.0
    denominator = text_evidence + image_evidence
    weights = np.full_like(text, 0.5, dtype=float)
    both_available = text_mask & image_mask
    has_evidence = denominator > np.finfo(float).eps
    weights[both_available[:, None] & has_evidence] = (
        text_evidence[both_available[:, None] & has_evidence]
        / denominator[both_available[:, None] & has_evidence]
    )
    weights[text_mask & ~image_mask, :] = 1.0
    weights[~text_mask & image_mask, :] = 0.0
    return weights


def blend_predictions_by_evidence(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    text_available: Sequence[bool],
    image_available: Sequence[bool],
    *,
    evidence_power: float = 1.0,
) -> np.ndarray:
    """Combina ramos dando mais peso ao ramo observável mais confiante."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    weights = calculate_evidence_weights(
        text,
        image,
        text_available,
        image_available,
        evidence_power=evidence_power,
    )
    return weights * text + (1.0 - weights) * image


def blend_predictions_by_confidence_gate(
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    *,
    prior_weight: float = 0.15,
    scale: float = 0.25,
    max_weight: float = 0.30,
) -> np.ndarray:
    """Aplica um gate global por estudo, limitado a uma cadeia conservadora."""
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if text.shape != image.shape:
        raise ValueError("As predições de texto e imagem precisam ter o mesmo formato.")
    if not np.isfinite(prior_weight) or not 0.0 <= prior_weight <= 1.0:
        raise ValueError("prior_weight precisa ser finito e estar em [0, 1].")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale precisa ser finito e não negativo.")
    if not np.isfinite(max_weight) or not prior_weight <= max_weight <= 1.0:
        raise ValueError("max_weight precisa ser finito, maior que prior_weight e estar em [0, 1].")
    text_confidence = np.abs(2.0 * text - 1.0).mean(axis=1)
    image_confidence = np.abs(2.0 * image - 1.0).mean(axis=1)
    weights = np.clip(prior_weight + scale * (text_confidence - image_confidence), 0.0, max_weight)
    return weights[:, None] * text + (1.0 - weights[:, None]) * image


def select_blend_weights(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    label_names: Sequence[str],
    grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    *,
    tie_weight: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Escolhe pesos por rótulo usando somente predições out-of-fold."""
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    if len(label_names) != truth.shape[1]:
        raise ValueError("label_names precisa ter um nome por rótulo.")
    candidate_grid = tuple(float(weight) for weight in grid)
    if not candidate_grid or any(not np.isfinite(weight) or not 0 <= weight <= 1 for weight in candidate_grid):
        raise ValueError("grid precisa conter pesos finitos em [0, 1].")
    resolved_tie_weight = _validate_tie_weight(tie_weight)
    weights = np.full(truth.shape[1], 0.5, dtype=float)
    scores: dict[str, float] = {}
    for index, label in enumerate(label_names):
        valid = np.isfinite(truth[:, index])
        if np.unique(truth[valid, index]).size < 2:
            continue
        best_weight = resolved_tie_weight
        best_score = -np.inf
        for weight in candidate_grid:
            blended = weight * text[valid, index] + (1.0 - weight) * image[valid, index]
            score = float(roc_auc_score(truth[valid, index], blended))
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and abs(weight - resolved_tie_weight) < abs(best_weight - resolved_tie_weight)
            ):
                best_weight = weight
                best_score = score
        weights[index] = best_weight
        scores[str(label)] = best_score
    if not scores:
        raise ValueError("Nenhum rótulo possui duas classes observáveis para selecionar pesos.")
    return weights, scores


def select_blend_weights_by_weakness(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
    *,
    grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    quantile: float = 0.1,
    tie_weight: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Escolhe pesos pelo pior perfil de AUC entre folds de calibração."""
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    if len(label_names) != truth.shape[1]:
        raise ValueError("label_names precisa ter um nome por rótulo.")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile precisa estar entre 0 e 1.")
    candidate_grid = tuple(float(weight) for weight in grid)
    if not candidate_grid or any(not np.isfinite(weight) or not 0 <= weight <= 1 for weight in candidate_grid):
        raise ValueError("grid precisa conter pesos finitos em [0, 1].")
    if not folds:
        raise ValueError("A seleção por fraqueza exige pelo menos uma dobra.")
    resolved_tie_weight = _validate_tie_weight(tie_weight)

    expected_indices = set(range(len(truth)))
    resolved_folds: list[np.ndarray] = []
    seen_indices: set[int] = set()
    for fold in folds:
        raw_indices = tuple(int(index) for index in fold["validacao_indices"])
        indices = np.asarray(sorted(raw_indices), dtype=int)
        fold_index_set = set(indices.tolist())
        if (
            not len(indices)
            or len(fold_index_set) != len(indices)
            or not fold_index_set.issubset(expected_indices)
            or seen_indices.intersection(fold_index_set)
        ):
            raise ValueError("As dobras de calibração precisam ser disjuntas e válidas.")
        seen_indices.update(fold_index_set)
        resolved_folds.append(indices)
    fallback_indices = np.concatenate(resolved_folds)

    weights = np.full(truth.shape[1], 0.5, dtype=float)
    scores: dict[str, float] = {}
    for label_index, label in enumerate(label_names):
        candidates: list[tuple[float, float, float]] = []
        for weight in candidate_grid:
            fold_scores: list[float] = []
            for indices in resolved_folds:
                valid = np.isfinite(truth[indices, label_index])
                if np.unique(truth[indices, label_index][valid]).size < 2:
                    continue
                blended = weight * text[indices, label_index] + (1.0 - weight) * image[indices, label_index]
                fold_scores.append(
                    float(roc_auc_score(truth[indices, label_index][valid], blended[valid]))
                )
            if not fold_scores:
                valid = np.isfinite(truth[fallback_indices, label_index])
                if np.unique(truth[fallback_indices, label_index][valid]).size >= 2:
                    blended = weight * text[fallback_indices, label_index] + (
                        1.0 - weight
                    ) * image[fallback_indices, label_index]
                    fold_scores.append(
                        float(roc_auc_score(truth[fallback_indices, label_index][valid], blended[valid]))
                    )
            if fold_scores:
                candidates.append(
                    (
                        float(np.quantile(fold_scores, quantile)),
                        abs(weight - resolved_tie_weight),
                        weight,
                    )
                )
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        robust_score, _, best_weight = candidates[0]
        weights[label_index] = best_weight
        scores[str(label)] = robust_score
    if not scores:
        raise ValueError("Nenhum rótulo possui duas classes observáveis nas dobras de calibração.")
    return weights, scores


def evaluate_oof_fusion(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
    *,
    grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    tie_weight: float = 0.5,
) -> OOFFusionEvaluation:
    """Escolhe pesos em dobras iniciais e mede em dobras ainda não usadas."""
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    calibration_indices, evaluation_indices = _oof_calibration_partition(folds)
    calibration_fold_count = max(1, len(folds) // 2)
    weights, calibration_scores = select_blend_weights(
        truth[calibration_indices],
        text[calibration_indices],
        image[calibration_indices],
        label_names,
        grid=grid,
        tie_weight=tie_weight,
    )
    blended = blend_predictions(text, image, weights)
    evaluation_score, evaluation_per_label = macro_roc_auc(
        truth[evaluation_indices],
        blended[evaluation_indices],
        label_names,
    )
    condition_scores = [evaluation_score]
    for fold in folds[calibration_fold_count:]:
        indices = np.asarray(fold["validacao_indices"], dtype=int)
        try:
            condition_scores.append(macro_roc_auc(truth[indices], blended[indices], label_names)[0])
        except ValueError:
            continue
    return OOFFusionEvaluation(
        weights,
        calibration_scores,
        evaluation_score,
        evaluation_per_label,
        tuple(float(score) for score in condition_scores),
        calibration_indices,
        evaluation_indices,
    )


def evaluate_oof_temperature_fusion(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
    *,
    grid: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
) -> OOFTemperatureEvaluation:
    """Calibra os dois ramos em OOF e avalia a média 50/50 em dobras futuras."""
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    calibration_indices, evaluation_indices = _oof_calibration_partition(folds)
    text_temperatures, text_losses = calibrate_temperatures(
        truth[calibration_indices],
        text[calibration_indices],
        label_names,
        grid=grid,
    )
    image_temperatures, image_losses = calibrate_temperatures(
        truth[calibration_indices],
        image[calibration_indices],
        label_names,
        grid=grid,
    )
    calibrated_text = apply_temperature(text, text_temperatures)
    calibrated_image = apply_temperature(image, image_temperatures)
    blended = 0.5 * (calibrated_text + calibrated_image)
    evaluation_score, evaluation_per_label = macro_roc_auc(
        truth[evaluation_indices], blended[evaluation_indices], label_names
    )
    condition_scores = [evaluation_score]
    calibration_fold_count = max(1, len(folds) // 2)
    for fold in folds[calibration_fold_count:]:
        indices = np.asarray(fold["validacao_indices"], dtype=int)
        try:
            condition_scores.append(macro_roc_auc(truth[indices], blended[indices], label_names)[0])
        except ValueError:
            continue
    return OOFTemperatureEvaluation(
        text_temperatures=text_temperatures,
        image_temperatures=image_temperatures,
        calibration_log_losses={"texto": text_losses, "imagem": image_losses},
        evaluation_score=evaluation_score,
        evaluation_per_label=evaluation_per_label,
        condition_scores=tuple(float(score) for score in condition_scores),
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
    )


def evaluate_oof_availability_fusion(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    text_available: Sequence[bool],
    image_available: Sequence[bool],
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
) -> OOFAvailabilityEvaluation:
    """Avalia disponibilidade observável em uma parte posterior das dobras OOF."""
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    resolved_text_available = _validate_availability_mask(
        "text_available", text_available, text.shape[0]
    )
    resolved_image_available = _validate_availability_mask(
        "image_available", image_available, text.shape[0]
    )
    calibration_indices, evaluation_indices = _oof_calibration_partition(folds)
    blended = blend_predictions_by_availability(
        text,
        image,
        resolved_text_available,
        resolved_image_available,
    )
    evaluation_score, evaluation_per_label = macro_roc_auc(
        truth[evaluation_indices], blended[evaluation_indices], label_names
    )
    condition_scores = [evaluation_score]
    calibration_fold_count = max(1, len(folds) // 2)
    for fold in folds[calibration_fold_count:]:
        indices = np.asarray(fold["validacao_indices"], dtype=int)
        try:
            condition_scores.append(macro_roc_auc(truth[indices], blended[indices], label_names)[0])
        except ValueError:
            continue
    return OOFAvailabilityEvaluation(
        evaluation_score=evaluation_score,
        evaluation_per_label=evaluation_per_label,
        condition_scores=tuple(float(score) for score in condition_scores),
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
    )


def evaluate_oof_evidence_fusion(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    text_available: Sequence[bool],
    image_available: Sequence[bool],
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
    *,
    evidence_power: float = 1.0,
) -> OOFEvidenceEvaluation:
    """Avalia uma fusão dinâmica sem aprender um gate com os rótulos."""
    if len(folds) < 2:
        raise ValueError("OOF evidence exige pelo menos duas dobras.")
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    expected_indices = set(range(len(truth)))
    seen_indices: set[int] = set()
    resolved_folds: list[np.ndarray] = []
    for fold in folds:
        raw_indices = tuple(int(index) for index in fold["validacao_indices"])
        indices = np.asarray(sorted(raw_indices), dtype=int)
        fold_index_set = set(indices.tolist())
        if (
            not len(indices)
            or len(fold_index_set) != len(indices)
            or not fold_index_set.issubset(expected_indices)
            or seen_indices.intersection(fold_index_set)
        ):
            raise ValueError("As dobras OOF precisam ser disjuntas e vÃ¡lidas.")
        seen_indices.update(fold_index_set)
        resolved_folds.append(indices)
    if seen_indices != expected_indices:
        raise ValueError("As dobras OOF precisam cobrir todos os estudos exatamente uma vez.")
    weights = calculate_evidence_weights(
        text,
        image,
        text_available,
        image_available,
        evidence_power=evidence_power,
    )
    predictions = weights * text + (1.0 - weights) * image
    evaluation_score, evaluation_per_label = macro_roc_auc(truth, predictions, label_names)
    condition_scores = [evaluation_score]
    for indices in resolved_folds:
        try:
            condition_scores.append(macro_roc_auc(truth[indices], predictions[indices], label_names)[0])
        except ValueError:
            continue
    return OOFEvidenceEvaluation(
        predictions=predictions,
        weights=weights,
        evaluation_score=evaluation_score,
        evaluation_per_label=evaluation_per_label,
        condition_scores=tuple(float(score) for score in condition_scores),
        evaluation_indices=np.arange(len(truth), dtype=int),
    )


def evaluate_oof_rank_fusion(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
) -> OOFRankEvaluation:
    """Avalia a média de percentis em predições OOF sem usar rótulos na regra."""
    if len(folds) < 2:
        raise ValueError("OOF rank exige pelo menos duas dobras.")
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    expected_indices = set(range(len(truth)))
    seen_indices: set[int] = set()
    resolved_folds: list[np.ndarray] = []
    for fold in folds:
        raw_indices = tuple(int(index) for index in fold["validacao_indices"])
        indices = np.asarray(sorted(raw_indices), dtype=int)
        fold_index_set = set(indices.tolist())
        if (
            not len(indices)
            or len(fold_index_set) != len(indices)
            or not fold_index_set.issubset(expected_indices)
            or seen_indices.intersection(fold_index_set)
        ):
            raise ValueError("As dobras OOF precisam ser disjuntas e válidas.")
        seen_indices.update(fold_index_set)
        resolved_folds.append(indices)
    if seen_indices != expected_indices:
        raise ValueError("As dobras OOF precisam cobrir todos os estudos exatamente uma vez.")
    predictions = blend_predictions_by_rank(text, image)
    evaluation_score, evaluation_per_label = macro_roc_auc(truth, predictions, label_names)
    condition_scores = [evaluation_score]
    for indices in resolved_folds:
        try:
            condition_scores.append(macro_roc_auc(truth[indices], predictions[indices], label_names)[0])
        except ValueError:
            continue
    return OOFRankEvaluation(
        predictions=predictions,
        evaluation_score=evaluation_score,
        evaluation_per_label=evaluation_per_label,
        condition_scores=tuple(float(score) for score in condition_scores),
        evaluation_indices=np.arange(len(truth), dtype=int),
    )


def evaluate_oof_logit_fusion(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
) -> OOFLogitEvaluation:
    """Avalia a média de logits em predições OOF sem aprender pesos."""
    if len(folds) < 2:
        raise ValueError("OOF logit exige pelo menos duas dobras.")
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    expected_indices = set(range(len(truth)))
    seen_indices: set[int] = set()
    resolved_folds: list[np.ndarray] = []
    for fold in folds:
        raw_indices = tuple(int(index) for index in fold["validacao_indices"])
        indices = np.asarray(sorted(raw_indices), dtype=int)
        fold_index_set = set(indices.tolist())
        if (
            not len(indices)
            or len(fold_index_set) != len(indices)
            or not fold_index_set.issubset(expected_indices)
            or seen_indices.intersection(fold_index_set)
        ):
            raise ValueError("As dobras OOF precisam ser disjuntas e válidas.")
        seen_indices.update(fold_index_set)
        resolved_folds.append(indices)
    if seen_indices != expected_indices:
        raise ValueError("As dobras OOF precisam cobrir todos os estudos exatamente uma vez.")
    predictions = blend_predictions_by_logit(text, image)
    evaluation_score, evaluation_per_label = macro_roc_auc(truth, predictions, label_names)
    condition_scores = [evaluation_score]
    for indices in resolved_folds:
        try:
            condition_scores.append(macro_roc_auc(truth[indices], predictions[indices], label_names)[0])
        except ValueError:
            continue
    return OOFLogitEvaluation(
        predictions=predictions,
        evaluation_score=evaluation_score,
        evaluation_per_label=evaluation_per_label,
        condition_scores=tuple(float(score) for score in condition_scores),
        evaluation_indices=np.arange(len(truth), dtype=int),
    )


def evaluate_oof_crossfit_fusion(
    y_true: np.ndarray,
    text_predictions: np.ndarray,
    image_predictions: np.ndarray,
    folds: Sequence[dict[str, object]],
    label_names: Sequence[str],
    *,
    grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    selection: str = "auc",
    weakness_quantile: float = 0.1,
    tie_weight: float = 0.5,
) -> OOFCrossfitEvaluation:
    """Aprende pesos sem observar o rótulo do fold que será avaliado."""
    if len(folds) < 2:
        raise ValueError("OOF cross-fitted exige pelo menos duas dobras.")
    if selection not in {"auc", "weakness"}:
        raise ValueError("selection precisa ser 'auc' ou 'weakness'.")
    if not 0 <= weakness_quantile <= 1:
        raise ValueError("weakness_quantile precisa estar entre 0 e 1.")
    resolved_tie_weight = _validate_tie_weight(tie_weight)
    truth = np.asarray(y_true, dtype=float)
    text = _validate_probabilities("text_predictions", text_predictions)
    image = _validate_probabilities("image_predictions", image_predictions)
    if truth.ndim != 2 or truth.shape != text.shape or text.shape != image.shape:
        raise ValueError("y_true, texto e imagem precisam ter o mesmo formato 2D.")
    expected_indices = set(range(len(truth)))
    resolved_evaluation_indices: list[np.ndarray] = []
    seen_indices: set[int] = set()
    for fold in folds:
        raw_indices = tuple(int(index) for index in fold["validacao_indices"])
        indices = np.asarray(sorted(raw_indices), dtype=int)
        fold_index_set = set(indices.tolist())
        if (
            not len(indices)
            or len(fold_index_set) != len(indices)
            or not fold_index_set.issubset(expected_indices)
            or seen_indices.intersection(fold_index_set)
        ):
            raise ValueError("As dobras OOF precisam particionar os estudos sem duplicatas.")
        seen_indices.update(fold_index_set)
        resolved_evaluation_indices.append(indices)
    if seen_indices != expected_indices:
        raise ValueError("As dobras OOF precisam cobrir todos os estudos exatamente uma vez.")

    fold_weights: list[np.ndarray] = []
    calibration_scores: list[dict[str, float]] = []
    fold_scores: list[float] = []
    crossfit_predictions = np.full_like(text, np.nan, dtype=float)
    resolved_calibration_indices: list[np.ndarray] = []
    all_indices = np.arange(len(truth), dtype=int)
    for fold_position, evaluation_indices in enumerate(resolved_evaluation_indices):
        evaluation_set = set(evaluation_indices.tolist())
        calibration_indices = np.asarray(
            [index for index in all_indices if int(index) not in evaluation_set],
            dtype=int,
        )
        if selection == "auc":
            weights, scores = select_blend_weights(
                truth[calibration_indices],
                text[calibration_indices],
                image[calibration_indices],
                label_names,
                grid=grid,
                tie_weight=resolved_tie_weight,
            )
        else:
            calibration_folds = tuple(
                fold
                for index, fold in enumerate(folds)
                if index != fold_position
            )
            weights, scores = select_blend_weights_by_weakness(
                truth,
                text,
                image,
                calibration_folds,
                label_names,
                grid=grid,
                quantile=weakness_quantile,
                tie_weight=resolved_tie_weight,
            )
        predictions = blend_predictions(
            text[evaluation_indices],
            image[evaluation_indices],
            weights,
        )
        try:
            fold_score, _ = macro_roc_auc(
                truth[evaluation_indices], predictions, label_names
            )
        except ValueError:
            fold_score = None
        crossfit_predictions[evaluation_indices] = predictions
        fold_weights.append(weights)
        calibration_scores.append(scores)
        if fold_score is not None:
            fold_scores.append(float(fold_score))
        resolved_calibration_indices.append(calibration_indices)
    evaluation_score, evaluation_per_label = macro_roc_auc(
        truth, crossfit_predictions, label_names
    )
    resolved_fold_weights = np.vstack(fold_weights)
    return OOFCrossfitEvaluation(
        weights=np.mean(resolved_fold_weights, axis=0),
        fold_weights=resolved_fold_weights,
        calibration_scores=tuple(calibration_scores),
        predictions=crossfit_predictions,
        evaluation_score=evaluation_score,
        evaluation_per_label=evaluation_per_label,
        condition_scores=tuple(float(score) for score in [evaluation_score, *fold_scores]),
        calibration_indices=tuple(resolved_calibration_indices),
        evaluation_indices=tuple(resolved_evaluation_indices),
    )
