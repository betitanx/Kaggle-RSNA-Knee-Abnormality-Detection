# Diário resumido — RSNA Knee

Seleção editorial dos registros de 07/08 a 23/09/2026. O arquivo original de pesquisa continha também operações, rascunhos e auditorias; este documento mantém as decisões que ajudam a compreender o projeto.

## 1. Baselines e primeiros controles

Foram explorados texto, metadados DICOM, estatísticas espaciais e de pixels, modelos leves e supervisão fraca. E44, E48 e E55 chegaram respectivamente a 0,557, 0,595 e 0,627 no público. Esses controles ajudaram a revelar a diferença entre proxies disponíveis no treino e os alvos oficiais.

## 2. Reprodução de soluções públicas

A reprodução E74 da família DINOsaur V4.9 alcançou 0,936 público em 29/08. Esse salto veio da adoção de uma solução visual pública com pesos e ensemble existentes, não de uma arquitetura inédita criada no projeto. E74 tornou-se o controle para descendentes posteriores.

## 3. Fusão e reponderação

Foram investigadas fusão por probabilidades, ranks e logits, pesos por alvo, proteção de alvos e diferentes densidades de janelas. Várias variantes empataram em 0,936; nenhuma melhoria pública acima de E74 foi confirmada nos registros.

E204 teve diagnóstico local de aproximadamente 0,972755, mas score público 0,928. A seleção havia observado os mesmos estudos gold: a métrica local não representava uma estimativa independente de generalização.

## 4. Treino e supervisão por laudos

Os experimentos de estudantes visuais e mistura de professores investigaram como extrair sinal dos estudos sem os 12 alvos oficiais completos. E211 registrou aproximadamente 0,734949 contra rótulos fracos e 0,688913 no gold diagnóstico. H212 registrou 0,719171 no report-OOF e 0,684874 no gold para o estudante misto. Não são métricas intercambiáveis nem evidência de melhora sobre 0,936 público.

## 5. Geometria e auditoria de independência

H214 comparou uma geometria Fine Spacing com o controle, mantendo os braços visuais fixos. O delta de +0,001645 nos 58 gold reutilizados permaneceu exploratório. O teste não foi submetido, e a análise levou a reforçar a independência da seleção como requisito separado da exclusão do treino.

As auditorias encontraram 58 estudos com os 12 alvos oficiais completos, todos já expostos ao processo de seleção. Sem nova coorte oficial independente, a esteira não podia certificar uma nova promoção.

## 6. Cinco sondagens públicas finais — 23/09

| Candidato | Diagnóstico local no gold exposto | Público |
|---|---:|---:|
| E120 | 0,967026 | 0,934 |
| E121 | 0,972928 | 0,933 |
| E122 | 0,972385 | 0,932 |
| E125 | 0,972926 | 0,932 |
| E123 | 0,972579 | 0,931 |

As sondagens foram autorizadas como exploração pública e não passaram por promoção independente. E74 permaneceu no máximo registrado, 0,936. O resultado mostrou que melhorar o ajuste local nos mesmos estudos não garantia melhorar a ordenação no teste público.

## Aprendizados e caminhos futuros

- Registrar hipóteses e tentativas sem score foi tão importante para interpretar o percurso quanto guardar as melhores submissões.
- OOF precisa de auditoria de proveniência, folds e história de seleção; o número isolado é insuficiente.
- Complementaridade deve ser medida entre previsões de modelos realmente diferentes, sem duplicar um checkpoint já presente no ensemble.
- Dados externos exigem compatibilidade de rótulos e termos de acesso/uso claros.
- Um próximo estudo exigiria uma coorte intocada e um contraste pré-especificado, como atenção anatômica ou uma única alteração de supervisão, antes de ajustar pesos por alvo.

**Estado do portfólio:** melhor score público confirmado 0,936; 205 IDs de candidatos/variações; ≈187 horas de execução estimadas. Score privado final e colocação não documentados.
