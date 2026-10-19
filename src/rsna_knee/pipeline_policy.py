"""Política de linhagem e gate local para a família E74-first."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

E60_LOCAL_OOF_SCORE = 0.6312198835466454
E73_PUBLIC_REFERENCE_SCORE = 0.914
E75_PUBLIC_REFERENCE_SCORE = 0.935
E74_REPRODUCED_OOF_SCORE = 0.9200900503799733
E74_CONFIRMED_PUBLIC_SCORE = 0.936


def _validate_score(value: float, name: str) -> float:
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} precisa ser finito e estar em [0, 1].")
    return score


@dataclass(frozen=True)
class E74FirstPolicy:
    """Contrato operacional para candidatos derivados do E74."""

    policy_id: str = "pipeline-e74-first-v1.8"
    parent_id: str = "E74"
    first_candidate_id: str = "E96"
    regression_control_id: str = "E60"
    regression_control_oof: float = E60_LOCAL_OOF_SCORE
    oof_floor_reference_id: str = "E97"
    oof_floor: float = 0.920
    parent_public_score: float = E74_CONFIRMED_PUBLIC_SCORE
    min_gain_over_parent_oof: float = 0.0
    required_folds: int = 0
    minimum_validation_seeds: int = 0
    oof_floor_is_external_reference: bool = False
    oof_floor_authorizes: bool = True
    requires_parent_oof: bool = False
    requires_fold_stability: bool = False
    requires_official_image_labels_complete: bool = True

    def __post_init__(self) -> None:
        if not self.requires_official_image_labels_complete:
            raise ValueError("rótulos oficiais completos por imagem são obrigatórios na política E74-first.")
        if self.parent_id != "E74":
            raise ValueError("o pai operacional precisa ser E74.")
        if self.first_candidate_id != "E96":
            raise ValueError("o primeiro candidato precisa ser E96.")
        if self.regression_control_id != "E60":
            raise ValueError("o controle de regressão precisa ser E60.")
        if self.oof_floor_reference_id != "E97":
            raise ValueError("a referência do piso OOF precisa ser E97.")
        for name in (
            "policy_id",
            "parent_id",
            "first_candidate_id",
            "regression_control_id",
            "oof_floor_reference_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} não pode ser vazio.")
        _validate_score(self.regression_control_oof, "regression_control_oof")
        _validate_score(self.oof_floor, "oof_floor")
        _validate_score(self.parent_public_score, "parent_public_score")
        if self.oof_floor <= self.regression_control_oof:
            raise ValueError("o piso OOF precisa ser maior que o controle E60.")
        if not isfinite(float(self.min_gain_over_parent_oof)) or float(self.min_gain_over_parent_oof) < 0.0:
            raise ValueError("min_gain_over_parent_oof precisa ser não negativo e finito.")
        if int(self.required_folds) < 0:
            raise ValueError("required_folds não pode ser negativo.")
        if int(self.minimum_validation_seeds) < 0:
            raise ValueError("minimum_validation_seeds não pode ser negativo.")
        if self.oof_floor_is_external_reference and self.oof_floor_authorizes:
            raise ValueError("A referência externa não pode autorizar o gate.")

    def required_oof(self, parent_oof: float | None = None) -> float:
        """Retorna o OOF mínimo necessário para liberar o candidato."""
        if parent_oof is not None:
            _validate_score(parent_oof, "parent_oof")
        required = float(self.oof_floor)
        if self.requires_parent_oof and parent_oof is not None:
            required = max(required, float(parent_oof) + float(self.min_gain_over_parent_oof))
        return required

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class E74GateResult:
    """Resultado auditável da avaliação do gate E74-first."""

    promoted: bool
    required_oof: float | None
    reasons: tuple[str, ...]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


DEFAULT_E74_FIRST_POLICY = E74FirstPolicy()


def evaluate_e74_first_gate(
    candidate_oof: float,
    *,
    parent_oof: float | None,
    fold_gains: Sequence[float] | None,
    validation_seeds: Sequence[int] | None,
    csv_validated: bool,
    manifest_validated: bool,
    independent_oof: bool,
    selection_independent: bool,
    selection_reused_validation: bool,
    official_image_labels_complete: bool,
    policy: E74FirstPolicy = DEFAULT_E74_FIRST_POLICY,
) -> E74GateResult:
    """Decide se um candidato atende ao gate OOF da linhagem E74.

    A política v1.8 usa o piso inclusivo de 0,920 e exige rótulos oficiais
    completos para cada estudo do OOF, além de independência tanto do treino quanto
    da seleção;
    o OOF do E74 é contexto comparativo, não uma barreira adicional. O score
    público do Kaggle permanece a validação externa e decide eventual promoção
    do campeão. O CLI verifica exclusão do estudo e, quando disponível, do
    paciente em cada fold; `independent_oof` recebe esse resultado já auditado.
    `selection_independent` também exige que o resultado não tenha sido usado
    para escolher modelo, geometria, pesos ou limiar.
    """
    candidate_score = _validate_score(candidate_oof, "candidate_oof")
    reasons: list[str] = []
    checks: dict[str, bool] = {}
    required_oof: float | None = None

    required_oof = policy.required_oof(parent_oof)
    checks["oof_supera_piso_triagem"] = candidate_score >= policy.oof_floor
    if not checks["oof_supera_piso_triagem"]:
        reasons.append("oof_nao_atinge_piso_oof")

    if policy.requires_parent_oof:
        if parent_oof is None:
            checks["oof_pai_e74_presente"] = False
            reasons.append("oof_pai_e74_ausente")
            checks["oof_supera_margem_do_pai"] = False
        else:
            checks["oof_pai_e74_presente"] = True
            checks["oof_supera_margem_do_pai"] = candidate_score > float(parent_oof) + policy.min_gain_over_parent_oof
            if not checks["oof_supera_margem_do_pai"]:
                reasons.append("oof_nao_supera_margem_do_pai")
    else:
        checks["oof_pai_e74_presente"] = True
        checks["oof_supera_margem_do_pai"] = True

    if not policy.requires_fold_stability:
        checks["dobras_suficientes"] = True
        checks["dobras_nao_regressivas"] = True
    else:
        if fold_gains is None:
            checks["dobras_suficientes"] = False
            checks["dobras_nao_regressivas"] = False
            reasons.append("dobras_ausentes")
        else:
            gains = [float(value) for value in fold_gains]
            if not all(isfinite(value) for value in gains):
                raise ValueError("fold_gains precisa conter somente valores finitos.")
            checks["dobras_suficientes"] = len(gains) >= policy.required_folds
            checks["dobras_nao_regressivas"] = bool(gains) and all(value >= 0.0 for value in gains)
            if not checks["dobras_suficientes"]:
                reasons.append("dobras_insuficientes")
            if not checks["dobras_nao_regressivas"]:
                reasons.append("dobra_regressiva")

    if policy.minimum_validation_seeds == 0:
        checks["sementes_suficientes"] = True
    else:
        if validation_seeds is None:
            seed_count = 0
        else:
            seed_count = len({int(seed) for seed in validation_seeds})
        checks["sementes_suficientes"] = seed_count >= policy.minimum_validation_seeds
        if not checks["sementes_suficientes"]:
            reasons.append("sementes_insuficientes")

    checks["csv_valido"] = bool(csv_validated)
    checks["manifesto_valido"] = bool(manifest_validated)
    checks["oof_independente"] = bool(independent_oof)
    checks["validacao_nao_reutilizada_na_selecao"] = selection_reused_validation is False
    checks["selecao_independente"] = bool(selection_independent) and checks[
        "validacao_nao_reutilizada_na_selecao"
    ]
    checks["rotulos_oficiais_completos_por_imagem"] = (
        official_image_labels_complete is True
        if policy.requires_official_image_labels_complete
        else True
    )
    if not checks["csv_valido"]:
        reasons.append("csv_invalido")
    if not checks["manifesto_valido"]:
        reasons.append("manifesto_invalido")
    if not checks["oof_independente"]:
        reasons.append("oof_nao_independente")
    if not checks["selecao_independente"]:
        reasons.append("selecao_nao_independente")
    if not checks["validacao_nao_reutilizada_na_selecao"]:
        reasons.append("validacao_reutilizada_na_selecao")
    if not checks["rotulos_oficiais_completos_por_imagem"]:
        reasons.append("oof_sem_rotulos_oficiais_completos_por_imagem")

    return E74GateResult(
        promoted=not reasons,
        required_oof=required_oof,
        reasons=tuple(reasons),
        checks=checks,
    )
