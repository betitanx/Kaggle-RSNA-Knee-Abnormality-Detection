from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsna_knee.pipeline_policy import (
    DEFAULT_E74_FIRST_POLICY,
    E74FirstPolicy,
    E60_LOCAL_OOF_SCORE,
    E73_PUBLIC_REFERENCE_SCORE,
    E74_REPRODUCED_OOF_SCORE,
    E74_CONFIRMED_PUBLIC_SCORE,
    E75_PUBLIC_REFERENCE_SCORE,
    evaluate_e74_first_gate,
)


def _valid_gate_kwargs() -> dict[str, object]:
    return {
        "parent_oof": E74_REPRODUCED_OOF_SCORE,
        "fold_gains": [0.001, 0.0, 0.002, 0.0001, 0.003],
        "validation_seeds": [42, 43],
        "csv_validated": True,
        "manifest_validated": True,
        "independent_oof": True,
        "selection_independent": True,
        "selection_reused_validation": False,
        "official_image_labels_complete": True,
    }


def test_politica_e74_first_separa_pai_controle_e_referencia_externa() -> None:
    policy = DEFAULT_E74_FIRST_POLICY

    assert policy.policy_id == "pipeline-e74-first-v1.8"
    assert policy.parent_id == "E74"
    assert policy.first_candidate_id == "E96"
    assert policy.regression_control_id == "E60"
    assert policy.regression_control_oof == E60_LOCAL_OOF_SCORE
    assert policy.oof_floor_reference_id == "E97"
    assert policy.oof_floor == pytest.approx(0.920)
    assert policy.parent_public_score == E74_CONFIRMED_PUBLIC_SCORE
    assert policy.requires_parent_oof is False
    assert policy.oof_floor_is_external_reference is False
    assert policy.oof_floor_authorizes is True
    assert policy.required_folds == 0
    assert policy.minimum_validation_seeds == 0
    assert policy.requires_official_image_labels_complete is True


def test_politica_rejeita_referencia_externa_com_autorizacao() -> None:
    with pytest.raises(ValueError, match="referência externa não pode autorizar"):
        E74FirstPolicy(oof_floor_is_external_reference=True, oof_floor_authorizes=True)
    with pytest.raises(ValueError, match="rótulos oficiais completos por imagem são obrigatórios"):
        E74FirstPolicy(requires_official_image_labels_complete=False)


def test_politica_rejeita_pai_ou_piso_incoerente() -> None:
    with pytest.raises(ValueError, match="pai operacional precisa ser E74"):
        E74FirstPolicy(parent_id="E60")
    with pytest.raises(ValueError, match="piso OOF precisa ser maior que o controle E60"):
        E74FirstPolicy(oof_floor=E60_LOCAL_OOF_SCORE)


def test_gate_aceita_candidato_acima_do_piso_e_da_reproducao_e74() -> None:
    result = evaluate_e74_first_gate(0.940, **_valid_gate_kwargs())

    assert result.promoted is True
    assert result.required_oof == pytest.approx(0.920)
    assert result.reasons == ()


def test_gate_usa_somente_o_piso_inclusivo_sem_exigir_ganho_sobre_e74() -> None:
    kwargs = _valid_gate_kwargs()

    abaixo_do_oof_pai = evaluate_e74_first_gate(0.92005, **kwargs)
    sem_oof_pai_kwargs = _valid_gate_kwargs()
    sem_oof_pai_kwargs["parent_oof"] = None
    sem_oof_pai = evaluate_e74_first_gate(0.920, **sem_oof_pai_kwargs)

    assert abaixo_do_oof_pai.promoted is True
    assert abaixo_do_oof_pai.required_oof == pytest.approx(0.920)
    assert sem_oof_pai.promoted is True
    assert sem_oof_pai.required_oof == pytest.approx(0.920)


def test_gate_aceita_empate_no_piso_inclusivo_e_rejeita_abaixo() -> None:
    floor_tie = evaluate_e74_first_gate(0.9202, **_valid_gate_kwargs())
    below_floor = evaluate_e74_first_gate(0.919999, **_valid_gate_kwargs())

    assert floor_tie.promoted is True
    assert "oof_nao_atinge_piso_oof" not in floor_tie.reasons
    assert below_floor.promoted is False
    assert "oof_nao_atinge_piso_oof" in below_floor.reasons


def test_gate_aceita_empate_exato_com_oof_reproduzido_do_e74() -> None:
    result = evaluate_e74_first_gate(
        E74_REPRODUCED_OOF_SCORE,
        **_valid_gate_kwargs(),
    )

    assert result.promoted is True
    assert "oof_nao_supera_margem_do_pai" not in result.reasons


def test_gate_rejeita_dobra_regressiva_e_falta_de_sementes() -> None:
    kwargs = _valid_gate_kwargs()
    kwargs["fold_gains"] = [0.001, -0.0001, 0.002, 0.0001, 0.003]
    kwargs["validation_seeds"] = [42]

    result = evaluate_e74_first_gate(0.9202, **kwargs)

    assert result.promoted is True
    assert "dobra_regressiva" not in result.reasons
    assert "sementes_insuficientes" not in result.reasons


def test_gate_rejeita_artefatos_ou_oof_nao_independente() -> None:
    kwargs = _valid_gate_kwargs()
    kwargs["csv_validated"] = False
    kwargs["manifest_validated"] = False
    kwargs["independent_oof"] = False

    result = evaluate_e74_first_gate(0.920, **kwargs)

    assert result.promoted is False
    assert "csv_invalido" in result.reasons
    assert "manifesto_invalido" in result.reasons
    assert "oof_nao_independente" in result.reasons


def test_gate_rejeita_oof_cujo_gold_foi_reutilizado_na_selecao() -> None:
    kwargs = _valid_gate_kwargs()
    kwargs["selection_reused_validation"] = True

    result = evaluate_e74_first_gate(0.959, **kwargs)

    assert result.promoted is False
    assert result.checks["selecao_independente"] is False
    assert result.checks["validacao_nao_reutilizada_na_selecao"] is False
    assert "selecao_nao_independente" in result.reasons
    assert "validacao_reutilizada_na_selecao" in result.reasons


def test_gate_rejeita_oof_sem_rotulos_oficiais_completos_baseados_em_imagem() -> None:
    kwargs = _valid_gate_kwargs()
    kwargs["official_image_labels_complete"] = False

    result = evaluate_e74_first_gate(0.99, **kwargs)

    assert result.promoted is False
    assert result.checks["rotulos_oficiais_completos_por_imagem"] is False
    assert "oof_sem_rotulos_oficiais_completos_por_imagem" in result.reasons


def test_configuracao_versionada_reflete_o_contrato_do_gate() -> None:
    config_path = Path("configs/pipeline_e74_first.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["linhagem"]["pai"] == "E74"
    assert config["linhagem"]["primeiro_candidato"] == "E96"
    assert config["linhagem"]["controle_regressao"] == "E60"
    assert config["referencias"]["E60"]["oof_local"] == pytest.approx(E60_LOCAL_OOF_SCORE)
    assert config["referencias"]["E73"]["score_publico_externo"] == pytest.approx(
        E73_PUBLIC_REFERENCE_SCORE
    )
    assert config["referencias"]["E74"]["score_publico_confirmado"] == pytest.approx(
        E74_CONFIRMED_PUBLIC_SCORE
    )
    assert config["referencias"]["E74"]["oof_local"] == pytest.approx(
        E74_REPRODUCED_OOF_SCORE
    )
    assert E73_PUBLIC_REFERENCE_SCORE > E60_LOCAL_OOF_SCORE
    assert config["referencias"]["E73"]["score_publico_externo"] == pytest.approx(
        E73_PUBLIC_REFERENCE_SCORE
    )
    assert config["referencias"]["E75"]["score_publico_externo"] == pytest.approx(
        E75_PUBLIC_REFERENCE_SCORE
    )
    assert config["gate_local"]["piso_triagem"]["referencia"] == "E97"
    assert config["gate_local"]["piso_triagem"]["score"] == pytest.approx(0.920)
    assert config["gate_local"]["piso_triagem"]["comparacao"] == "maior_ou_igual"
    assert config["gate_local"]["piso_triagem"]["necessario"] is True
    assert config["gate_local"]["piso_triagem"]["suficiente_sozinho"] is False
    assert config["gate_local"]["requer_selecao_independente"] is True
    assert config["gate_local"]["requer_rotulos_oficiais_completos_por_imagem"] is True
    assert config["gate_local"]["origem_rotulos_oficiais"] == "train.csv"
    assert config["gate_local"]["requer_todas_as_colunas_do_template"] is True
    assert config["gate_local"]["oof_recalculado_pelo_cli"] is True
    assert config["gate_local"]["independencia_treino_verificada_pelo_manifesto"] is True
    assert config["gate_local"]["independencia_selecao_requer_auditoria_de_proveniencia"] is True
    assert config["gate_local"]["ganho_minimo_sobre_oof_pai"] == pytest.approx(0.0)
    assert config["gate_local"]["comparacao_ganho_pai"] == "informativa_nao_autorizadora"
    assert config["gate_local"]["requer_oof_pai_e74"] is False
    assert config["gate_local"]["dobras_nao_regressivas"] == 0
    assert config["gate_local"]["sementes_minimas"] == 0
    assert config["submissao"]["score_publico_campeao"] == pytest.approx(0.936)
    assert config["implementacao"]["comando_autorizador"] == "submission gate"
    assert config["implementacao"]["registro_exige_relatorio_gate"] is True
