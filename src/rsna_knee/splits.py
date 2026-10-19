"""Divisões de validação que preservam grupos de paciente ou estudo."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .data_contract import DataSchema


@dataclass(frozen=True)
class GroupSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def build_group_split(
    frame: pd.DataFrame,
    schema: DataSchema,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> GroupSplit:
    """Cria um holdout determinístico sem compartilhar grupos."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction precisa estar entre 0 e 1.")
    groups = frame[schema.group_column]
    if groups.isna().any():
        raise ValueError("A coluna de grupo não pode conter valores ausentes.")
    if groups.nunique() < 2:
        raise ValueError("São necessários pelo menos dois grupos para validar.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_indices, validation_indices = next(splitter.split(frame, groups=groups))
    return GroupSplit(tuple(sorted(map(int, train_indices))), tuple(sorted(map(int, validation_indices))))


def build_group_splits(
    frame: pd.DataFrame,
    schema: DataSchema,
    validation_fraction: float = 0.2,
    seeds: tuple[int, ...] = (42,),
) -> tuple[GroupSplit, ...]:
    """Cria holdouts independentes por grupo para estimar estabilidade."""
    if not seeds:
        raise ValueError("É necessário fornecer pelo menos uma seed de validação.")
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("As seeds de validação precisam ser únicas.")
    return tuple(
        build_group_split(
            frame,
            schema,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        for seed in normalized
    )
