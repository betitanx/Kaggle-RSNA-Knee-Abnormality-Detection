"""Validação de CSV e controle local do limite diário de submissões."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


class SubmissionSchemaError(ValueError):
    """O CSV de predição não é compatível com o template."""


class SubmissionBudgetError(RuntimeError):
    """O limite operacional de submissões foi atingido."""


def _infer_id_column(frame: pd.DataFrame) -> str:
    preferred = ("study_id", "exam_id", "case_id", "sample_id", "id")
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    return str(frame.columns[0])


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Artefato de submissão não encontrado: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    details = dict(metadata or {})
    reserved = {"submission_id", "submitted_at", "artifact_path", "artifact_sha256", "annotated_at"}
    if any(str(key) in reserved for key in details):
        raise ValueError("metadata não pode sobrescrever campos reservados do ledger.")
    for score_name in ("local_score", "public_score"):
        score = details.get(score_name)
        if score is None:
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{score_name} precisa ser numérico.") from error
        if not isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError(f"{score_name} precisa estar em [0, 1].")
        details[score_name] = numeric_score
    try:
        json.dumps(details, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata do ledger precisa ser JSON serializável.") from error
    return details


def validate_submission(
    sample_submission: pd.DataFrame,
    predictions: pd.DataFrame,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Alinha predições ao template e verifica IDs, colunas e probabilidades."""
    if sample_submission.empty or predictions.empty:
        raise SubmissionSchemaError("Template e predições precisam conter linhas.")
    id_column = id_column or _infer_id_column(sample_submission)
    if id_column not in sample_submission.columns or id_column not in predictions.columns:
        raise SubmissionSchemaError(f"A coluna ID {id_column!r} precisa existir nos dois CSVs.")
    expected = list(map(str, sample_submission.columns))
    received = list(map(str, predictions.columns))
    missing = [column for column in expected if column not in received]
    extra = [column for column in received if column not in expected]
    if missing or extra:
        raise SubmissionSchemaError(f"Colunas incompatíveis: ausentes={missing}, extras={extra}.")
    expected_ids = sample_submission[id_column].astype(str)
    received_ids = predictions[id_column].astype(str)
    if expected_ids.duplicated().any() or received_ids.duplicated().any() or set(expected_ids) != set(received_ids):
        raise SubmissionSchemaError("IDs do template e das predições não coincidem.")
    result = predictions.copy()
    result[id_column] = result[id_column].astype(str)
    result = result.set_index(id_column).loc[expected_ids.tolist()].reset_index()
    label_columns = [column for column in expected if column != id_column]
    for column in label_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise SubmissionSchemaError(f"Predições inválidas na coluna {column!r}; esperado [0, 1].")
        result[column] = values.astype(float)
    return result[expected]


class _LedgerLock:
    """Lock simples e atômico por criação exclusiva de arquivo."""

    def __init__(self, path: Path, timeout_seconds: float = 10.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.handle = None

    def __enter__(self) -> "_LedgerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        while True:
            try:
                self.handle = self.path.open("x", encoding="utf-8", newline="\n")
                self.handle.write(str(os.getpid()))
                self.handle.flush()
                return self
            except (FileExistsError, PermissionError):
                if time.monotonic() - started >= self.timeout_seconds:
                    raise SubmissionBudgetError("Não foi possível obter lock do ledger no prazo.")
                time.sleep(0.01)

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            self.handle.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class DailySubmissionLedger:
    """Ledger JSONL que impede exceder o limite diário configurado."""

    def __init__(
        self,
        path: str | Path,
        max_per_day: int = 5,
        timezone_name: str = "America/Sao_Paulo",
        lock_timeout_seconds: float = 10.0,
    ) -> None:
        if max_per_day < 1:
            raise ValueError("max_per_day precisa ser positivo.")
        self.path = Path(path)
        self.max_per_day = max_per_day
        try:
            self.timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Fuso horário desconhecido: {timezone_name}") from error
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.lock_timeout_seconds = lock_timeout_seconds

    def _entries(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        entries: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=self.timezone)
        return timestamp.astimezone(self.timezone)

    def _entry_date(self, entry: Mapping[str, object]) -> date:
        submitted_at = entry.get("submitted_at")
        if not isinstance(submitted_at, str):
            raise ValueError("Entrada de ledger sem submitted_at ISO válido.")
        return self._normalize_timestamp(datetime.fromisoformat(submitted_at)).date()

    def _count_entries(self, entries: list[dict[str, object]], day: date) -> int:
        return sum(self._entry_date(entry) == day for entry in entries)

    def _write_entries(self, entries: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def count(self, day: date) -> int:
        with _LedgerLock(self.lock_path, self.lock_timeout_seconds):
            return self._count_entries(self._entries(), day)

    def register(
        self,
        submission_id: str,
        submitted_at: datetime | None = None,
        *,
        artifact_path: str | Path | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        normalized_id = str(submission_id).strip()
        if not normalized_id:
            raise ValueError("submission_id precisa ser não vazio.")
        artifact = Path(artifact_path) if artifact_path is not None else None
        artifact_hash = _sha256_file(artifact) if artifact is not None else None
        details = _prepare_metadata(metadata)
        timestamp = self._normalize_timestamp(submitted_at or datetime.now(tz=self.timezone))
        with _LedgerLock(self.lock_path, self.lock_timeout_seconds):
            entries = self._entries()
            if any(entry.get("submission_id") == normalized_id for entry in entries):
                raise SubmissionBudgetError(f"A submissão {normalized_id!r} já foi registrada.")
            if self._count_entries(entries, timestamp.date()) >= self.max_per_day:
                raise SubmissionBudgetError(f"Limite de {self.max_per_day} submissões no dia atingido.")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry: dict[str, object] = {
                "submission_id": normalized_id,
                "submitted_at": timestamp.isoformat(),
            }
            if artifact is not None and artifact_hash is not None:
                entry["artifact_path"] = str(artifact)
                entry["artifact_sha256"] = artifact_hash
            entry.update(details)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    def annotate(
        self,
        submission_id: str,
        *,
        annotated_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_id = str(submission_id).strip()
        if not normalized_id:
            raise ValueError("submission_id precisa ser não vazio.")
        details = _prepare_metadata(metadata)
        if not details:
            raise ValueError("A anotação precisa conter pelo menos um campo.")
        timestamp = self._normalize_timestamp(annotated_at or datetime.now(tz=self.timezone))
        with _LedgerLock(self.lock_path, self.lock_timeout_seconds):
            entries = self._entries()
            matches = [entry for entry in entries if entry.get("submission_id") == normalized_id]
            if not matches:
                raise ValueError(f"A submissão {normalized_id!r} não encontrada no ledger.")
            if len(matches) > 1:
                raise ValueError(f"A submissão {normalized_id!r} aparece mais de uma vez no ledger.")
            entry = matches[0]
            entry.update(details)
            entry["annotated_at"] = timestamp.isoformat()
            self._write_entries(entries)
            return dict(entry)
