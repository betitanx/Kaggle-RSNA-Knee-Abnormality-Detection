from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.fusion import (
    apply_temperature,
    blend_predictions,
    blend_predictions_by_availability,
    blend_predictions_by_confidence_gate,
    blend_predictions_by_evidence,
    blend_predictions_by_logit,
    blend_predictions_by_rank,
    blend_predictions_by_rank_shrinkage,
    calibrate_temperatures,
    evaluate_oof_availability_fusion,
    evaluate_oof_crossfit_fusion,
    evaluate_oof_evidence_fusion,
    evaluate_oof_fusion,
    evaluate_oof_logit_fusion,
    evaluate_oof_rank_fusion,
    evaluate_oof_temperature_fusion,
    select_blend_weights,
    select_blend_weights_by_weakness,
)


def test_select_blend_weights_chooses_the_stronger_branch_per_label() -> None:
    truth = np.array([[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1]])
    text = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])
    image = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]])

    weights, scores = select_blend_weights(truth, text, image, ("acl", "meniscus"), grid=(0.0, 1.0))

    assert weights.tolist() == [1.0, 0.0]
    assert scores == {"acl": 1.0, "meniscus": 1.0}
    assert np.allclose(blend_predictions(text, image, weights), text * [1.0, 0.0] + image * [0.0, 1.0])


def test_select_blend_weights_uses_the_declared_prior_on_a_score_tie() -> None:
    truth = np.array([[0], [0], [1], [1]], dtype=float)
    text = np.array([[0.2], [0.8], [0.2], [0.8]])
    image = text.copy()

    weights, _ = select_blend_weights(
        truth,
        text,
        image,
        ("acl",),
        grid=(0.0, 0.15, 0.5, 1.0),
        tie_weight=0.15,
    )

    assert weights.tolist() == [0.15]


def test_blend_predictions_by_availability_uses_only_the_present_modality() -> None:
    text = np.array([[0.1], [0.2], [0.3], [0.4]])
    image = np.array([[0.9], [0.8], [0.7], [0.6]])

    fused = blend_predictions_by_availability(
        text,
        image,
        text_available=(True, False, False, True),
        image_available=(True, True, False, False),
    )

    assert np.allclose(fused[:, 0], [0.5, 0.8, 0.5, 0.4])


def test_blend_predictions_by_evidence_weights_the_more_confident_available_branch() -> None:
    text = np.array([[0.9], [0.6], [0.5]])
    image = np.array([[0.6], [0.1], [0.5]])

    fused = blend_predictions_by_evidence(
        text,
        image,
        text_available=[True, True, True],
        image_available=[True, True, True],
    )

    assert np.allclose(fused[:, 0], [0.84, 0.20, 0.50])


def test_blend_predictions_by_evidence_uses_the_only_available_branch() -> None:
    text = np.array([[0.9], [0.2]])
    image = np.array([[0.1], [0.8]])

    fused = blend_predictions_by_evidence(
        text,
        image,
        text_available=[True, False],
        image_available=[False, True],
    )

    assert np.allclose(fused, [[0.9], [0.8]])


def test_blend_predictions_by_confidence_gate_limits_the_chain_weight() -> None:
    chain = np.array([[0.9], [0.1], [0.6]])
    dicom = np.array([[0.6], [0.4], [0.55]])

    fused = blend_predictions_by_confidence_gate(
        chain,
        dicom,
        prior_weight=0.15,
        scale=0.25,
        max_weight=0.30,
    )

    expected_weights = np.array([[0.30], [0.30], [0.175]])
    assert np.allclose(fused, expected_weights * chain + (1.0 - expected_weights) * dicom)


def test_blend_predictions_by_confidence_gate_rejects_invalid_limits() -> None:
    predictions = np.array([[0.2], [0.8]])

    with pytest.raises(ValueError, match="prior_weight"):
        blend_predictions_by_confidence_gate(predictions, predictions, prior_weight=1.1)
    with pytest.raises(ValueError, match="max_weight"):
        blend_predictions_by_confidence_gate(predictions, predictions, max_weight=0.1)


def test_evaluate_oof_evidence_fusion_records_all_oof_conditions() -> None:
    truth = np.array([[0], [1], [0], [1], [0], [1]], dtype=float)
    text = np.array([[0.1], [0.9], [0.4], [0.6], [0.2], [0.8]])
    image = np.array([[0.2], [0.8], [0.3], [0.7], [0.6], [0.4]])
    folds = (
        {"validacao_indices": [0, 1]},
        {"validacao_indices": [2, 3]},
        {"validacao_indices": [4, 5]},
    )

    evaluation = evaluate_oof_evidence_fusion(
        truth,
        text,
        image,
        text_available=[True] * len(truth),
        image_available=[True] * len(truth),
        folds=folds,
        label_names=("acl",),
    )

    assert evaluation.predictions.shape == truth.shape
    assert evaluation.weights.shape == truth.shape
    assert evaluation.evaluation_indices.tolist() == list(range(len(truth)))
    assert len(evaluation.condition_scores) == 4


def test_blend_predictions_by_rank_uses_order_instead_of_probability_scale() -> None:
    text = np.array([[0.01], [0.02], [0.99]])
    image = np.array([[0.20], [0.80], [0.90]])

    fused = blend_predictions_by_rank(text, image)

    assert np.allclose(fused[:, 0], [1 / 6, 0.5, 5 / 6])


def test_blend_predictions_by_rank_maps_test_values_to_oof_reference() -> None:
    fused = blend_predictions_by_rank(
        np.array([[0.2], [0.8]]),
        np.array([[0.3], [0.7]]),
        text_reference=np.array([[0.1], [0.3], [0.9]]),
        image_reference=np.array([[0.2], [0.4], [0.8]]),
    )

    assert np.allclose(fused[:, 0], [1 / 3, 2 / 3])


def test_blend_predictions_by_rank_shrinkage_preserves_the_declared_prior() -> None:
    text = np.array([[0.1], [0.2], [0.9]])
    image = np.array([[0.4], [0.5], [0.6]])

    fused = blend_predictions_by_rank_shrinkage(
        text,
        image,
        text_weight=0.25,
        shrinkage=0.5,
    )
    raw = 0.25 * text + 0.75 * image
    ranked = 0.25 * np.array([[1 / 6], [0.5], [5 / 6]]) + 0.75 * np.array(
        [[1 / 6], [0.5], [5 / 6]]
    )

    assert np.allclose(fused, 0.5 * raw + 0.5 * ranked)


def test_blend_predictions_by_rank_shrinkage_rejects_invalid_hyperparameters() -> None:
    predictions = np.array([[0.2], [0.8]])

    with pytest.raises(ValueError, match="text_weight"):
        blend_predictions_by_rank_shrinkage(predictions, predictions, text_weight=1.1)
    with pytest.raises(ValueError, match="shrinkage"):
        blend_predictions_by_rank_shrinkage(predictions, predictions, shrinkage=-0.1)


def test_evaluate_oof_rank_fusion_records_all_oof_conditions() -> None:
    truth = np.array([[0], [1], [0], [1], [0], [1]], dtype=float)
    text = np.array([[0.1], [0.9], [0.4], [0.6], [0.2], [0.8]])
    image = np.array([[0.2], [0.8], [0.3], [0.7], [0.6], [0.4]])
    folds = (
        {"validacao_indices": [0, 1]},
        {"validacao_indices": [2, 3]},
        {"validacao_indices": [4, 5]},
    )

    evaluation = evaluate_oof_rank_fusion(
        truth,
        text,
        image,
        folds=folds,
        label_names=("acl",),
    )

    assert evaluation.predictions.shape == truth.shape
    assert evaluation.evaluation_indices.tolist() == list(range(len(truth)))
    assert len(evaluation.condition_scores) == 4


def test_blend_predictions_by_logit_uses_geometric_mean_of_odds() -> None:
    fused = blend_predictions_by_logit(
        np.array([[0.8], [0.2]]),
        np.array([[0.6], [0.4]]),
    )

    assert np.allclose(fused[:, 0], [0.7101020514, 0.2898979486])


def test_evaluate_oof_logit_fusion_records_all_oof_conditions() -> None:
    truth = np.array([[0], [1], [0], [1], [0], [1]], dtype=float)
    text = np.array([[0.1], [0.9], [0.4], [0.6], [0.2], [0.8]])
    image = np.array([[0.2], [0.8], [0.3], [0.7], [0.6], [0.4]])
    folds = (
        {"validacao_indices": [0, 1]},
        {"validacao_indices": [2, 3]},
        {"validacao_indices": [4, 5]},
    )

    evaluation = evaluate_oof_logit_fusion(
        truth,
        text,
        image,
        folds=folds,
        label_names=("acl",),
    )

    assert evaluation.predictions.shape == truth.shape
    assert evaluation.evaluation_indices.tolist() == list(range(len(truth)))
    assert len(evaluation.condition_scores) == 4


def test_select_blend_weights_ignores_unknown_labels_but_rejects_invalid_probabilities() -> None:
    truth = np.array([[np.nan], [np.nan]])
    text = np.array([[0.2], [0.8]])
    image = np.array([[0.4], [0.6]])

    with pytest.raises(ValueError, match="duas classes"):
        select_blend_weights(truth, text, image, ("acl",))
    with pytest.raises(ValueError, match="probabilidades"):
        blend_predictions(text, np.array([[1.2], [0.4]]), 0.5)


def test_evaluate_oof_fusion_calibrates_on_earlier_folds_only() -> None:
    truth = np.array([[0], [0], [0], [1], [1], [1], [1], [1]])
    text = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9], [0.6], [0.5]])
    image = np.array([[0.9], [0.8], [0.7], [0.3], [0.2], [0.1], [0.4], [0.5]])
    folds = (
        {"validacao_indices": [0, 3]},
        {"validacao_indices": [1, 4]},
        {"validacao_indices": [2, 5]},
        {"validacao_indices": [6, 7]},
    )

    evaluation = evaluate_oof_fusion(truth, text, image, folds, ("acl",), grid=(0.0, 1.0))

    assert evaluation.weights.tolist() == [1.0]
    assert evaluation.calibration_indices.tolist() == [0, 1, 3, 4]
    assert evaluation.evaluation_indices.tolist() == [2, 5, 6, 7]
    assert evaluation.evaluation_score == 1.0


def test_calibrate_temperatures_uses_only_observable_labels_and_preserves_probabilities() -> None:
    truth = np.array([[0], [0], [0], [1], [1], [1]], dtype=float)
    predictions = np.array([[0.40], [0.45], [0.48], [0.52], [0.55], [0.60]])

    temperatures, losses = calibrate_temperatures(
        truth,
        predictions,
        ("acl",),
        grid=(0.5, 1.0, 2.0),
    )
    calibrated = apply_temperature(predictions, temperatures)

    assert temperatures.shape == (1,)
    assert 0.5 <= temperatures[0] <= 2.0
    assert set(losses) == {"acl"}
    assert losses["acl"] < 1.0
    assert calibrated.shape == predictions.shape
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_evaluate_oof_temperature_fusion_keeps_calibration_and_evaluation_groups_separate() -> None:
    truth = np.array([[0], [0], [0], [1], [1], [1], [0], [1]], dtype=float)
    text = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9], [0.4], [0.6]])
    image = np.array([[0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.5], [0.55]])
    folds = (
        {"validacao_indices": [0, 3]},
        {"validacao_indices": [1, 4]},
        {"validacao_indices": [2, 5]},
        {"validacao_indices": [6, 7]},
    )

    evaluation = evaluate_oof_temperature_fusion(
        truth,
        text,
        image,
        folds,
        ("acl",),
        grid=(0.5, 1.0, 2.0),
    )

    assert evaluation.calibration_indices.tolist() == [0, 1, 3, 4]
    assert evaluation.evaluation_indices.tolist() == [2, 5, 6, 7]
    assert evaluation.text_temperatures.shape == (1,)
    assert evaluation.image_temperatures.shape == (1,)
    assert 0 <= evaluation.evaluation_score <= 1


def test_evaluate_oof_availability_fusion_keeps_earlier_folds_out_of_evaluation() -> None:
    truth = np.array([[0], [0], [0], [1], [1], [1], [0], [1]], dtype=float)
    text = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9], [0.4], [0.6]])
    image = np.array([[0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.5], [0.55]])
    text_available = (True, False, True, True, True, False, False, True)
    image_available = (True, True, False, True, False, True, False, True)
    folds = (
        {"validacao_indices": [0, 3]},
        {"validacao_indices": [1, 4]},
        {"validacao_indices": [2, 5]},
        {"validacao_indices": [6, 7]},
    )

    evaluation = evaluate_oof_availability_fusion(
        truth,
        text,
        image,
        text_available,
        image_available,
        folds,
        ("acl",),
    )

    assert evaluation.calibration_indices.tolist() == [0, 1, 3, 4]
    assert evaluation.evaluation_indices.tolist() == [2, 5, 6, 7]
    assert evaluation.evaluation_score == 1.0


def test_evaluate_oof_crossfit_learns_each_fold_from_the_other_folds() -> None:
    truth = np.array([[0], [0], [0], [1], [1], [1], [0], [1]], dtype=float)
    text = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9], [0.4], [0.6]])
    image = 1.0 - text
    folds = (
        {"validacao_indices": [0, 3]},
        {"validacao_indices": [1, 4]},
        {"validacao_indices": [2, 5]},
        {"validacao_indices": [6, 7]},
    )

    evaluation = evaluate_oof_crossfit_fusion(
        truth,
        text,
        image,
        folds,
        ("acl",),
        grid=(0.0, 1.0),
    )

    assert evaluation.evaluation_score == 1.0
    assert evaluation.weights.tolist() == [1.0]
    assert evaluation.fold_weights.shape == (4, 1)
    assert [indices.tolist() for indices in evaluation.evaluation_indices] == [
        [0, 3],
        [1, 4],
        [2, 5],
        [6, 7],
    ]
    assert evaluation.calibration_indices[0].tolist() == [1, 2, 4, 5, 6, 7]
    assert all(
        set(calibration).isdisjoint(evaluation_indices)
        and set(calibration) | set(evaluation_indices) == set(range(len(truth)))
        for calibration, evaluation_indices in zip(
            evaluation.calibration_indices,
            evaluation.evaluation_indices,
        )
    )
    assert np.isfinite(evaluation.predictions).all()


def test_evaluate_oof_crossfit_propagates_the_conservative_tie_prior() -> None:
    truth = np.array([[0], [1], [0], [1]], dtype=float)
    text = np.array([[0.2], [0.8], [0.2], [0.8]])
    image = text.copy()
    folds = (
        {"validacao_indices": [0, 1]},
        {"validacao_indices": [2, 3]},
    )

    evaluation = evaluate_oof_crossfit_fusion(
        truth,
        text,
        image,
        folds,
        ("acl",),
        grid=(0.0, 0.15, 0.5, 1.0),
        tie_weight=0.15,
    )

    assert evaluation.weights.tolist() == [0.15]
    assert evaluation.fold_weights.tolist() == [[0.15], [0.15]]


def test_blend_weight_selection_rejects_an_invalid_tie_prior() -> None:
    truth = np.array([[0], [1]], dtype=float)
    predictions = np.array([[0.2], [0.8]])

    with pytest.raises(ValueError, match="tie_weight"):
        select_blend_weights(
            truth,
            predictions,
            predictions,
            ("acl",),
            tie_weight=1.5,
        )


def test_select_blend_weights_by_weakness_prefers_the_stable_fold_profile() -> None:
    truth = np.array(
        [[1], [0], [0], [1], [0], [0], [1], [1], [0], [1], [1], [0], [0], [0], [1], [1]],
        dtype=float,
    )
    text = np.array(
        [[0.882637], [0.134680], [0.624869], [0.892552], [0.668919], [0.353123], [0.652074], [0.577422],
         [0.148303], [0.012153], [0.852146], [0.988648], [0.820576], [0.209704], [0.877936], [0.463952]],
    )
    image = np.array(
        [[0.200275], [0.414741], [0.713155], [0.738103], [0.697386], [0.511645], [0.336431], [0.434402],
         [0.236467], [0.266098], [0.198372], [0.600192], [0.807867], [0.802081], [0.910750], [0.498813]],
    )
    folds = tuple(
        {"validacao_indices": list(range(start, start + 4))}
        for start in range(0, len(truth), 4)
    )

    primary_weights, _ = select_blend_weights(truth, text, image, ("acl",), grid=(0.0, 0.5, 1.0))
    robust_weights, robust_scores = select_blend_weights_by_weakness(
        truth,
        text,
        image,
        folds,
        ("acl",),
        grid=(0.0, 0.5, 1.0),
        quantile=0.0,
    )

    assert primary_weights.tolist() == [1.0]
    assert robust_weights.tolist() == [0.5]
    assert robust_scores == {"acl": 0.25}
