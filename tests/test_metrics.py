from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.metrics import infer_subgroup_columns, subgroup_roc_auc


def test_subgroup_roc_auc_records_each_group_without_changing_global_metric() -> None:
    frame = pd.DataFrame(
        {
            "institution": ["A"] * 4 + ["B"] * 4,
            "language": ["pt"] * 8,
        }
    )
    truth = np.asarray([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=float)
    predictions = np.asarray([[0.1], [0.9], [0.2], [0.8], [0.1], [0.9], [0.2], [0.8]])

    result = subgroup_roc_auc(
        frame,
        truth,
        predictions,
        ("acl",),
        ("institution", "language"),
        min_group_size=4,
    )

    assert set(result) == {"institution", "language"}
    assert result["institution"]["grupos_avaliados"] == 2
    assert result["institution"]["scores"] == [1.0, 1.0]
    assert result["language"]["grupos_avaliados"] == 1
    assert result["language"]["scores"] == [1.0]


def test_infer_subgroup_columns_uses_metadata_tokens_only() -> None:
    frame = pd.DataFrame(
        columns=["study_id", "institution", "language", "report", "acl"]
    )

    assert infer_subgroup_columns(frame) == ("institution", "language")
