# Metodologia e limites

## Ciclo de pesquisa

O ciclo organizava uma hipótese, identificava o controle, preparava a execução, validava os artefatos e registrava a decisão. A família E74 passou a ser a referência após alcançar 0,936 no leaderboard público. Ideias posteriores incluíram fusão por logits/ranks, reponderação dos braços, densidade de janelas, supervisão fraca e treinamento de estudantes visuais.

A proposta de Bennett em [arXiv:2301.12987v4](https://arxiv.org/pdf/2301.12987v4) inspirou a busca por hipóteses que resistissem a diferentes condições de avaliação. O uso de métricas de pior percentil e cobertura no código é uma operacionalização experimental do projeto, não uma reprodução formal do artigo ou uma garantia de generalização.

## Quatro medidas diferentes

| Medida | O que permite afirmar |
|---|---|
| OOF com exclusão do treino | O estudo não participou do treino daquele fold, se o manifesto estiver correto |
| Validação independente da seleção | A hipótese e os parâmetros não foram escolhidos observando aqueles resultados |
| Diagnóstico no gold exposto | Desempenho descritivo no conjunto já consultado, sujeito a sobreajuste de seleção |
| Score público Kaggle | Resultado no subconjunto público do teste; não é score privado final |

Excluir um estudo do treino não o torna independente se ele já participou de seleção de checkpoints, pesos ou hipóteses. Também não existe uma relação fixa entre OOF e score público.

## Gargalo encontrado

A auditoria do CSV de treino disponível no projeto registrou 4.407 estudos, dos quais 58 tinham os 12 alvos oficiais completos. Esses 58 já haviam sido utilizados na história de seleção/avaliação; os outros 4.349 não forneciam os 12 alvos oficiais completos. Não havia uma coorte oficial intocada disponível para certificar novas melhorias.

Laudos possibilitam supervisão fraca, mas rótulos extraídos deles não substituem as anotações oficiais baseadas em imagem. Treino e avaliação contra um mesmo professor lexical podem medir concordância com esse professor, sem demonstrar melhora na tarefa oficial.

## Critério de promoção

O estado final da política exigia OOF ≥0,920, independência do treino e da seleção, rastreabilidade e controles de regressão. O módulo `pipeline_policy.py` preserva essa lógica de decisão do projeto; suas declarações precisam ser sustentadas por proveniência real. Ele não consegue provar sozinho que os folds ou as alegações de independência são verdadeiros.

Os 58 estudos expostos impediam uma nova aprovação independente. Cinco sondagens públicas foram feitas por autorização explícita como experimentos exploratórios; todas ficaram abaixo de E74. Essa exceção não transformou os diagnósticos locais em OOF independente.

## Métrica e inferência

O núcleo calcula AUC por alvo e média macro. `macro_roc_auc` ignora alvos sem duas classes observáveis em uma amostra: isso facilita diagnósticos de folds pequenos, mas exige registrar quais alvos entraram na média. Não se deve comparar médias com conjuntos de alvos diferentes como se fossem equivalentes ao avaliador oficial.

Ensembles dependem de pré-processamento consistente com os checkpoints, ordenação geométrica DICOM e agregação por estudo. A auditoria também investigou normalização, densidade de slices e complementaridade entre modelos; ganhos em arquiteturas de terceiros não foram tratados como garantia de ganho no E74.

## Estado dos resultados

O encerramento da competição foi informado pelo autor do projeto. Este portfólio preserva resultados públicos confirmados até 23/09/2026; não atribui colocação, medalha ou score privado final sem evidência. A estimativa de tempo é operacional, não uma medição de horas humanas ou de GPU.
