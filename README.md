# Kaggle RSNA Knee Abnormality Detection

Um laboratório de pesquisa aplicada para classificação de anormalidades em ressonâncias magnéticas do joelho: hipóteses, experimentos, validação e aprendizado com resultados que nem sempre melhoraram.

**Melhor score público registrado: 0,936 ROC-AUC macro · 205 candidatos/variações registrados · ≈187 horas estimadas de execução.**

![Evolução dos experimentos e volume diário de candidatos](docs/imagens/progresso.png)

## Sobre o projeto

Desenvolvi este projeto para explorar a competição [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), construindo com assistência de agentes de IA uma esteira de pesquisa que registra hipóteses, compara candidatos, valida artefatos e acompanha resultados no Kaggle.

O trabalho começou com baselines de texto, metadados e estatísticas de pixels e evoluiu para reprodução e adaptação de ensembles públicos, experimentos de supervisão fraca e auditorias de validação. A melhor execução pública registrada foi **E74, com 0,936**, baseada na solução pública DINOsaur V4.9. A contribuição deste projeto está na engenharia da esteira, nas reproduções, nas ablações e na análise crítica; a arquitetura campeã e os pesos públicos têm seus próprios autores, relacionados em [Referências e atribuição](docs/referencias.md).

Esta é uma seleção dos recursos mais relevantes para portfólio. Os resultados são o último estado confirmado nos registros, em **23/09/2026**. Não foi verificado aqui o score privado final nem a colocação após o encerramento da competição.

## Resultados selecionados

| Experimento | Abordagem | Score público |
|---|---|---:|
| E44 | Metadados DICOM e combinação em cadeia | 0,557 |
| E48 | Estatísticas de pixels | 0,595 |
| E55 | Regressão com supervisão soft de pixels | 0,627 |
| **E74** | **Reprodução DINOsaur V4.9, ensemble visual** | **0,936** |
| E75 | Reprodução de outra versão pública | 0,935 |
| E204 | Transformer 85% + Raptor 15% | 0,928 |
| E120 | Consenso logit E74–E118 | 0,934 |
| E121 | Consenso logit por alvo | 0,933 |
| E122 / E125 | Consenso fraco / shrinkage de Fracture | 0,932 |
| E123 | Proteção de alvos estáveis | 0,931 |

E109, E130, E207 e E208 também chegaram a 0,936, sem superar o primeiro máximo do E74. Os números acima são públicos, não resultados clínicos nem scores privados finais.

Os 205 registros representam IDs de candidatos e variações, inclusive tentativas sem score; não equivalem a 205 modelos treinados ou submissões. As ≈187 horas combinam 169 h 40 min de duas execuções longas com aproximadamente 17 h de runtime deduplicado; pode haver sobreposição e o trabalho manual não foi contabilizado.

## O que este repositório demonstra

- Organização de pesquisa iterativa com hipóteses e resultados rastreáveis.
- Separação de pacientes/estudos na validação, métricas por alvo e análise de estabilidade.
- Fusão de probabilidades, ranks e logits, com seleção e avaliação em conjuntos separados.
- Validação de CSVs, hashes de artefatos e controle local de orçamento de submissões.
- Auditoria de vazamento, reutilização do conjunto de seleção e limitações da supervisão por laudos.

O aprendizado central foi que **melhorar um diagnóstico local não garante melhora no leaderboard**. As cinco últimas sondagens tiveram diagnósticos locais entre 0,9670 e 0,9729 nos mesmos estudos gold já expostos, mas resultados públicos de 0,931 a 0,934. Esses diagnósticos não são OOF independente. A distinção é detalhada em [Metodologia e limites](docs/metodologia.md).

## Estrutura

```text
docs/                 Metodologia, reprodução, referências e gráfico
resultados/           Histórico agregado e resultados selecionados
notebooks/            Demonstração executável com dados sintéticos
src/rsna_knee/        Núcleo selecionado de validação, fusão e submissões
tests/                Testes dos módulos selecionados
scripts/              Regeneração do gráfico
lab.md                Síntese dos experimentos e aprendizados
```

## Executar localmente

Requer Python 3.11 ou superior. O núcleo e os exemplos usam CPU e não precisam dos exames da competição.

```bash
python -m pip install -e ".[dev,visualizacao,notebooks]"
python -m pytest
python scripts/gerar_progresso.py
jupyter notebook notebooks/01_validacao_e_fusao.ipynb
```

O notebook utiliza exclusivamente dados sintéticos para demonstrar divisão por grupo, avaliação e validação de submissão. **Não reproduz o score 0,936.** A reprodução da inferência E74 depende de dados, checkpoints e ambiente Kaggle externos: veja o [guia de reprodução](docs/reproducao.md).

## Recursos

- [Diário resumido](lab.md)
- [Metodologia e limitações](docs/metodologia.md)
- [Reprodução e proveniência do E74](docs/reproducao.md)
- [Fontes e créditos](docs/referencias.md)
- [Histórico agregado usado no gráfico](resultados/historico.json)

Dados de exames, laudos, identificadores de pacientes, checkpoints, credenciais e logs operacionais não fazem parte desta publicação. O código é apresentado como material de pesquisa e portfólio, sem validação para uso clínico.
