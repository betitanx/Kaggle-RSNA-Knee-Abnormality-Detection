# Top 3 soluções preservadas

As três primeiras soluções a alcançar o máximo público confirmado do projeto foram E74, E109 e E130, todas com **0,936**. A classificação é editorial, ordenada por confirmação cronológica. Não representa um pódio da competição nem diferenças de desempenho entre resultados empatados. E207 e E208 atingiram o mesmo score posteriormente.

## 1. E74 — DINOsaur V4.9

[Notebook de inferência](../notebooks/01_e74_dinosaur_v49.ipynb) · [Execução Kaggle, versão 345735626](https://www.kaggle.com/code/betitanx/rsna-knee-dinosaur-v4?scriptVersionId=345735626)

**Score público 0,936; confirmado em 29/08/2026.** O controle campeão reúne múltiplos braços visuais e agregação de rankings, com encoders DINO e braços CoAtNet/Raptor. O nome histórico do experimento não identifica sozinho todos os backbones efetivamente usados. A inferência preserva o pré-processamento e os checkpoints do snapshot, em vez de reconstruir a arquitetura a partir do nome.

Foi a primeira solução do projeto a alcançar esse resultado e permaneceu como referência. Sua base é o notebook público DINOsaur de Roman Tamrazov e a linhagem pública Bend the Knee; os créditos estão no [aviso de atribuição](../notebooks/NOTICE.md).

## 2. E109 — Transformer adaptativo

[Notebook de inferência](../notebooks/02_e109_transformer_adaptativo.ipynb) · versão Kaggle registrada **346147820**

**Score público 0,936; confirmado em 31/08/2026.** Preserva a inferência completa do E74 e reduz o peso do Raptor segundo o desacordo entre rankings dos braços. A inclinação fixa de 0,20 desloca a combinação em direção ao Transformer nos casos de maior discordância.

O diagnóstico local de 0,9612756846 foi escolhido em uma busca sobre os mesmos 58 estudos gold já observados. É resultado de desenvolvimento pós-seleção, não OOF independente. O empate público mostrou que esse ganho local não se traduziu em melhora confirmada no leaderboard.

## 3. E130 — margem majoritária

[Notebook de inferência](../notebooks/03_e130_margem_majoritaria.ipynb) · versão Kaggle registrada **346774795**

**Score público 0,936; confirmado em 02/09/2026.** Captura as previsões do controle E74, do Transformer e do CoAtNet; agrega preferências par a par usando a margem média positiva, exige dois votos e desempata com ranking médio e epsilon de 1e-6.

O diagnóstico de referência foi 0,9582783690 contra 0,9574425109 do controle no mesmo escopo de desenvolvimento. Na execução de teste observada, alterou empates de Effusion e Contusion sem superar o score público do E74. Esse diagnóstico também não deve ser apresentado como validação independente da seleção.

## O que foi preservado

Cada notebook contém as células reais de inferência, sem outputs, pesos ou dados de pacientes. As células de código têm hashes registrados em `resultados/top3.json` e são idênticas aos respectivos snapshots locais. Uma introdução em português contextualiza o resultado e os créditos. Não houve novo treino, inferência de GPU ou submissão na preparação deste portfólio.

As dependências externas e as limitações estão no [guia de reprodução](reproducao.md). Score privado final e colocação não foram documentados.
