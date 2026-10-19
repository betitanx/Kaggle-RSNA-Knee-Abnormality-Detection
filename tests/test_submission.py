from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from rsna_knee.submission import (
    DailySubmissionLedger,
    SubmissionBudgetError,
    SubmissionSchemaError,
    validate_submission,
)


def test_submission_preserves_template_columns_and_order() -> None:
    sample = pd.DataFrame({"study_id": ["s1", "s2"], "acl": [0.0, 0.0], "meniscus": [0.0, 0.0]})
    predictions = pd.DataFrame({"meniscus": [0.2, 0.3], "study_id": ["s2", "s1"], "acl": [0.8, 0.1]})

    result = validate_submission(sample, predictions)

    assert list(result.columns) == ["study_id", "acl", "meniscus"]
    assert result["study_id"].tolist() == ["s1", "s2"]
    assert result["acl"].tolist() == [0.1, 0.8]


def test_submission_rejects_missing_or_extra_ids() -> None:
    sample = pd.DataFrame({"study_id": ["s1", "s2"], "acl": [0.0, 0.0]})
    predictions = pd.DataFrame({"study_id": ["s1", "s3"], "acl": [0.1, 0.2]})

    with pytest.raises(SubmissionSchemaError, match="IDs"):
        validate_submission(sample, predictions)


def test_ledger_blocks_sixth_submission_on_same_local_date(tmp_path) -> None:
    ledger = DailySubmissionLedger(tmp_path / "ledger.jsonl", max_per_day=5)
    timestamp = datetime(2026, 8, 6, 12, 0)

    for index in range(5):
        ledger.register(f"s{index}", submitted_at=timestamp)

    with pytest.raises(SubmissionBudgetError, match="5"):
        ledger.register("s5", submitted_at=timestamp)

    assert ledger.count(timestamp.date()) == 5


def test_ledger_normalizes_aware_timestamp_to_configured_local_date(tmp_path) -> None:
    ledger = DailySubmissionLedger(tmp_path / "ledger.jsonl", max_per_day=5, timezone_name="America/Sao_Paulo")
    utc_timestamp = datetime(2026, 8, 7, 1, 30, tzinfo=ZoneInfo("UTC"))

    ledger.register("utc", submitted_at=utc_timestamp)

    assert ledger.count(datetime(2026, 8, 6).date()) == 1
    assert ledger.count(datetime(2026, 8, 7).date()) == 0


def test_ledger_records_artifact_hash_and_experiment_metadata(tmp_path) -> None:
    artifact = tmp_path / "submission.csv"
    artifact.write_text("study_id,acl\ns1,0.7\n", encoding="utf-8")
    ledger = DailySubmissionLedger(tmp_path / "ledger.jsonl", max_per_day=5)

    ledger.register(
        "B0-text-2026-08-07",
        submitted_at=datetime(2026, 8, 7, 12, 0),
        artifact_path=artifact,
        metadata={
            "candidate_id": "B0-text",
            "local_score": 0.72,
            "public_score": 0.68,
            "hypothesis": "O relatório textual fornece o controle mais barato.",
            "decision": "enviar como baseline",
        },
    )

    entry = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["artifact_path"] == str(artifact)
    assert entry["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert entry["candidate_id"] == "B0-text"
    assert entry["local_score"] == 0.72
    assert entry["public_score"] == 0.68
    assert entry["hypothesis"].startswith("O relatório")
    assert entry["decision"] == "enviar como baseline"


def test_ledger_rejects_scores_outside_probability_range(tmp_path) -> None:
    artifact = tmp_path / "submission.csv"
    artifact.write_text("study_id,acl\ns1,0.7\n", encoding="utf-8")
    ledger = DailySubmissionLedger(tmp_path / "ledger.jsonl")

    with pytest.raises(ValueError, match="public_score"):
        ledger.register(
            "invalida",
            artifact_path=artifact,
            metadata={"public_score": 1.1},
        )


def test_ledger_annotates_public_score_without_consuming_new_slot(tmp_path) -> None:
    ledger = DailySubmissionLedger(tmp_path / "ledger.jsonl", max_per_day=1)
    timestamp = datetime(2026, 8, 7, 12, 0)
    ledger.register("B0-text-2026-08-07", submitted_at=timestamp)

    updated = ledger.annotate(
        "B0-text-2026-08-07",
        annotated_at=datetime(2026, 8, 7, 13, 0),
        metadata={"public_score": 0.71, "decision": "manter"},
    )

    entries = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert updated["public_score"] == 0.71
    assert entries[0]["decision"] == "manter"
    assert entries[0]["annotated_at"].startswith("2026-08-07T13:00:00")
    assert ledger.count(timestamp.date()) == 1


def test_ledger_annotation_requires_existing_submission(tmp_path) -> None:
    ledger = DailySubmissionLedger(tmp_path / "ledger.jsonl")

    with pytest.raises(ValueError, match="não encontrada"):
        ledger.annotate("ausente", metadata={"public_score": 0.5})


def test_ledger_is_atomic_under_concurrent_registration(tmp_path) -> None:
    ledger = DailySubmissionLedger(tmp_path / "ledger.jsonl", max_per_day=5)
    timestamp = datetime(2026, 8, 6, 12, 0)

    def register(index: int) -> bool:
        try:
            ledger.register(f"concorrente-{index}", submitted_at=timestamp)
        except SubmissionBudgetError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=10) as executor:
        outcomes = list(executor.map(register, range(10)))

    assert sum(outcomes) == 5
    assert ledger.count(timestamp.date()) == 5
