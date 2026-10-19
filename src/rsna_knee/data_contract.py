"""Inferência e validação mínima do contrato dos CSVs da competição."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


class SchemaError(ValueError):
    """Erro acionável no layout ou no schema dos dados."""


@dataclass(frozen=True)
class DataSchema:
    """Colunas necessárias para separar, treinar e submeter um experimento."""

    id_column: str
    group_column: str
    label_columns: tuple[str, ...]
    text_column: str | None = None
    group_level: str = "custom"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DataTables:
    """Tabelas mínimas encontradas em um diretório de competição."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    train_series: pd.DataFrame | None = None
    test_series: pd.DataFrame | None = None


def discover_tables(data_dir: str | Path) -> DataTables:
    """Lê train/test/template sem baixar ou alterar dados do usuário."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {root}")
    paths: dict[str, Path] = {}
    csv_files = {path.name.lower(): path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".csv"}
    for key in ("train", "test", "sample_submission"):
        exact = csv_files.get(f"{key}.csv")
        if exact is None:
            alternatives = [path for name, path in csv_files.items() if key in name]
            if len(alternatives) == 1:
                exact = alternatives[0]
        if exact is not None:
            paths[key] = exact
    missing = [key for key in ("train", "test", "sample_submission") if key not in paths]
    if missing:
        raise FileNotFoundError(f"CSV(s) ausente(s) em {root}: {missing}")
    train_series = pd.read_csv(csv_files["train_series.csv"]) if "train_series.csv" in csv_files else None
    test_series = pd.read_csv(csv_files["test_series.csv"]) if "test_series.csv" in csv_files else None
    tables = DataTables(
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["test"]),
        pd.read_csv(paths["sample_submission"]),
        train_series,
        test_series,
    )
    if any(frame.empty for frame in (tables.train, tables.test, tables.sample_submission)):
        raise SchemaError("Os CSVs descobertos não podem estar vazios.")
    return tables


def _find_id_column(frame: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred:
        if preferred not in frame.columns:
            raise SchemaError(f"A coluna ID configurada não existe: {preferred}")
        return preferred

    priorities = ("study_id", "exam_id", "case_id", "sample_id", "image_id", "uid", "id")
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for name in priorities:
        if name in lowered:
            return lowered[name]
    contains = [column for column in frame.columns if "study" in str(column).lower() or "exam" in str(column).lower()]
    if len(contains) == 1:
        return str(contains[0])
    raise SchemaError("Não foi possível inferir uma coluna ID; configure id_column explicitamente.")


def _find_group_column(frame: pd.DataFrame, id_column: str, preferred: str | None = None) -> str:
    if preferred:
        if preferred not in frame.columns:
            raise SchemaError(f"A coluna de grupo configurada não existe: {preferred}")
        return preferred

    priorities = ("patient_id", "subject_id", "participant_id", "patient", "subject", "group")
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for name in priorities:
        if name in lowered and lowered[name] != id_column:
            return lowered[name]
    id_name = id_column.lower()
    if any(token in id_name for token in ("study", "exam", "case")):
        return id_column
    raise SchemaError(
        "Não foi possível inferir um grupo de paciente/estudo; configure group_column explicitamente."
    )


def _find_text_column(frame: pd.DataFrame, preferred: str | None = None) -> str | None:
    if preferred:
        if preferred not in frame.columns:
            raise SchemaError(f"A coluna de texto configurada não existe: {preferred}")
        return preferred
    tokens = ("report", "text", "finding", "description", "impression")
    for column in frame.columns:
        lowered = str(column).lower()
        if any(token in lowered for token in tokens):
            return str(column)
    return None


def _infer_group_level(group_column: str, id_column: str) -> str:
    if group_column == id_column:
        return "study"
    lowered = group_column.lower()
    if any(token in lowered for token in ("patient", "subject", "participant")):
        return "patient"
    return "custom"


def infer_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame | None = None,
    overrides: Mapping[str, str] | None = None,
) -> DataSchema:
    """Infere ID, grupo, texto e rótulos sem fixar o schema da competição."""
    if train.empty or test.empty:
        raise SchemaError("Train e test precisam conter pelo menos uma linha.")
    overrides = overrides or {}
    id_column = _find_id_column(train, overrides.get("id_column"))
    if id_column not in test.columns:
        raise SchemaError(f"A coluna ID {id_column!r} não existe no test.")
    group_column = _find_group_column(train, id_column, overrides.get("group_column"))
    text_column = _find_text_column(train, overrides.get("text_column"))

    if "label_columns" in overrides:
        label_columns = tuple(name.strip() for name in overrides["label_columns"].split(",") if name.strip())
    elif sample_submission is not None:
        sample_id = _find_id_column(sample_submission, id_column if id_column in sample_submission.columns else None)
        sample_labels = [str(column) for column in sample_submission.columns if str(column) != sample_id]
        missing_from_train = [column for column in sample_labels if column not in train.columns]
        if missing_from_train:
            raise SchemaError(f"Rótulos do template ausentes no train: {missing_from_train}")
        label_columns = tuple(sample_labels)
    else:
        metadata = {id_column, group_column, *(column for column in train.columns if column == text_column)}
        label_columns = tuple(
            str(column) for column in train.columns
            if str(column) not in metadata and pd.api.types.is_numeric_dtype(train[column])
        )

    if not label_columns:
        raise SchemaError("Nenhum rótulo foi encontrado; forneça sample_submission ou label_columns.")
    missing = [column for column in label_columns if column not in train.columns]
    if missing:
        raise SchemaError(f"Rótulos ausentes no train: {missing}")
    for label in label_columns:
        numeric = pd.to_numeric(train[label], errors="coerce")
        invalid = train[label].notna() & (numeric.isna() | ~numeric.isin((0, 1)))
        if invalid.any():
            raise SchemaError(f"Rótulos precisam ser binários (0, 1 ou ausente): {label!r}")
    return DataSchema(
        id_column,
        group_column,
        label_columns,
        text_column,
        _infer_group_level(group_column, id_column),
    )
