"""Métricas locais robustas para classificação multilabel."""

from __future__ import annotations

from collections.abc import Sequence
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_roc_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    """Calcula AUC macro, ignorando rótulos sem duas classes no fold."""
    truth = np.asarray(y_true)
    predicted = np.asarray(probabilities)
    if truth.shape != predicted.shape or truth.ndim != 2:
        raise ValueError("y_true e probabilities precisam ter o mesmo formato 2D.")
    if truth.shape[1] != len(label_names):
        raise ValueError("label_names precisa ter uma entrada por coluna.")
    if not np.isfinite(predicted).all() or (predicted < 0).any() or (predicted > 1).any():
        raise ValueError("probabilities contém valores não finitos ou fora de [0, 1].")
    per_label: dict[str, float] = {}
    for index, name in enumerate(label_names):
        column_truth = truth[:, index]
        valid = np.isfinite(column_truth)
        if np.unique(column_truth[valid]).size < 2:
            continue
        per_label[str(name)] = float(roc_auc_score(column_truth[valid], predicted[valid, index]))
    if not per_label:
        raise ValueError("Nenhum rótulo possui duas classes observáveis para calcular AUC.")
    return float(np.mean(list(per_label.values()))), per_label


def infer_subgroup_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Encontra colunas de domínio sem confundir ID, texto ou rótulo."""
    tokens = (
        "institution",
        "institu",
        "language",
        "idioma",
        "hospital",
        "center",
        "centre",
        "site",
        "scanner",
        "vendor",
    )
    selected: list[str] = []
    for column in frame.columns:
        normalized = unicodedata.normalize("NFKD", str(column)).encode("ascii", "ignore").decode("ascii").lower()
        if any(token in normalized for token in tokens):
            selected.append(str(column))
    return tuple(selected)


def subgroup_roc_auc(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    subgroup_columns: Sequence[str],
    *,
    min_group_size: int = 4,
) -> dict[str, object]:
    """Calcula diagnóstico por domínio sem participar do ranking padrão."""
    truth = np.asarray(y_true)
    predicted = np.asarray(probabilities)
    if truth.ndim != 2 or truth.shape != predicted.shape or len(frame) != truth.shape[0]:
        raise ValueError("frame, y_true e probabilities precisam estar alinhados.")
    if len(label_names) != truth.shape[1]:
        raise ValueError("label_names precisa ter uma entrada por coluna.")
    if min_group_size < 2:
        raise ValueError("min_group_size precisa ser pelo menos 2.")
    resolved_columns = tuple(str(column) for column in subgroup_columns)
    if len(set(resolved_columns)) != len(resolved_columns):
        raise ValueError("subgroup_columns não pode conter duplicatas.")
    _ = _validate_probability_matrix(predicted)
    diagnostics: dict[str, object] = {}
    for column in resolved_columns:
        if column not in frame.columns:
            raise ValueError(f"Coluna de subgrupo ausente: {column!r}")
        values = frame[column].astype("string").fillna("<ausente>").astype(str).to_numpy()
        groups: dict[str, object] = {}
        scores: list[float] = []
        for group in sorted(pd.unique(values).tolist()):
            indices = np.flatnonzero(values == group)
            entry: dict[str, object] = {"n": int(len(indices))}
            if len(indices) < min_group_size:
                entry["status"] = "ignorado"
                entry["motivo"] = "grupo_pequeno"
            else:
                try:
                    score, per_label = macro_roc_auc(truth[indices], predicted[indices], label_names)
                except ValueError as error:
                    entry["status"] = "ignorado"
                    entry["motivo"] = str(error)
                else:
                    entry["status"] = "concluído"
                    entry["score"] = score
                    entry["per_label"] = per_label
                    scores.append(score)
            groups[str(group)] = entry
        diagnostics[column] = {
            "grupos_total": int(len(groups)),
            "grupos_avaliados": int(len(scores)),
            "grupos": groups,
            "scores": scores,
            "weakness_proxy": weakness_proxy(scores) if scores else None,
            "coverage": condition_coverage(scores, threshold=0.5) if scores else None,
            "min_group_size": int(min_group_size),
        }
    return diagnostics


def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    predicted = np.asarray(probabilities, dtype=float)
    if not np.isfinite(predicted).all() or (predicted < 0).any() or (predicted > 1).any():
        raise ValueError("probabilities contém valores não finitos ou fora de [0, 1].")
    return predicted


def uncertainty_diagnostic(probabilities: np.ndarray) -> float:
    """Mede incerteza preditiva; não deve ser usada isoladamente para ranquear."""
    predicted = np.asarray(probabilities, dtype=float)
    if predicted.size == 0:
        raise ValueError("probabilities não pode ser vazio.")
    if not np.isfinite(predicted).all():
        raise ValueError("probabilities contém valores não finitos.")
    clipped = np.clip(predicted, 0.0, 1.0)
    return float(np.mean(4.0 * clipped * (1.0 - clipped)))


def weakness_proxy(condition_scores: list[float] | np.ndarray, quantile: float = 0.1) -> float:
    """Aproxima generalidade pelo desempenho no pior percentil de condições."""
    scores = np.asarray(condition_scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("condition_scores precisa conter valores finitos.")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile precisa estar entre 0 e 1.")
    return float(np.quantile(scores, quantile))


def condition_coverage(condition_scores: list[float] | np.ndarray, threshold: float) -> float:
    """Retorna a proporção de condições que alcança o piso de desempenho."""
    scores = np.asarray(condition_scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("condition_scores precisa conter valores finitos.")
    return float(np.mean(scores >= threshold))
