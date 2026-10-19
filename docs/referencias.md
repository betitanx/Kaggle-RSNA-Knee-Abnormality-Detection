# Referências e atribuição

## Competição e soluções públicas

- [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection): dados, regras e avaliação oficial.
- [Roman Tamrazov — RSNA Knee | DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4): base pública da reprodução E74.
- [Mattia Angeli — Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs?scriptVersionId=349077359): referência de ensembles visuais e inferência.
- [Mattia Angeli — Bend the Knee to DinoV3 (ensembled)](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled?scriptVersionId=344206476): outra referência estudada.
- [Dread Development — Raptor Knee WideDense](https://www.kaggle.com/dreaddevelopment/raptor-knee-widedense): referência dos braços CoAtNet.
- [Discussão 735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304): hipóteses de supervisão, ensemble e validação investigadas no projeto. Relatos de terceiros não são resultados próprios.

## Pesquisa

- [Michael Timothy Bennett — The Optimal Choice of Hypothesis Is the Weakest, Not the Shortest](https://arxiv.org/pdf/2301.12987v4): inspiração conceitual para seleção orientada à generalização.

## Dados externos avaliados

MRNet, fastMRI+, KneeMRI e SKM-TEA foram examinados como possibilidades. Acesso, licenças, domínio e compatibilidade de rótulos limitaram sua adoção. A presença de uma referência no diário não significa que seus dados foram usados na solução campeã. Nenhum desses datasets é redistribuído aqui.

## Autoria e uso

Curadoria, experimentação e engenharia da esteira: **Bernardo Braga Perobeli — [betitanx](https://github.com/betitanx)**, com assistência de agentes de IA. Créditos das arquiteturas, notebooks públicos, bibliotecas e checkpoints pertencem aos respectivos autores.

Os três notebooks de inferência são derivados de código público Apache 2.0, com atribuição e alterações em [NOTICE](../notebooks/NOTICE.md) e uma cópia da [licença](../licencas/Apache-2.0.txt). A página pública [DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4/notebook?scriptVersionId=347433881) e a referência [Bend the Knee to DinoV3](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled/notebook) exibem essa licença; ela cobre o código dessas fontes, não automaticamente os dados, modelos ou checkpoints anexados.

Essa licença não foi estendida ao núcleo próprio em `src/`, aos textos ou aos datasets externos. Consulte o autor para reutilização do material próprio e as fontes originais para os demais recursos.
