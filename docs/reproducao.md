# Reprodução e proveniência

## Núcleo local

Os módulos em `src/rsna_knee` foram selecionados da esteira original: contrato de dados, separação por grupo, métricas, fusão e validação de submissão. Os testes correspondentes são executáveis em CPU. O manifesto `resultados/proveniencia.json` registra os hashes dos arquivos selecionados.

O CLI completo e o registro de exposição com identificadores de estudos não integram esta seleção. A política histórica está descrita em [Metodologia e limites](metodologia.md).

## Execução campeã E74

- Notebook da conta: [rsna-knee-dinosaur-v4, versão 345735626](https://www.kaggle.com/code/betitanx/rsna-knee-dinosaur-v4?scriptVersionId=345735626).
- Origem pública: [RSNA Knee | DINOsaur V4, Roman Tamrazov](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4).
- Resultado registrado: 0,936 no score público.
- Ambiente de referência: Kaggle, GPU T4 ×2 e internet desligada; checkpoints e dependências precisam estar anexados previamente.
- A referência arquivada contém múltiplos braços visuais, incluindo DINOv2 e CoAtNet; o nome histórico de vários experimentos menciona DINOv3 e não basta para identificar todos os backbones efetivamente carregados.

Para tentar reproduzir, abra a versão fixada na conta Kaggle, confira os inputs originais, as permissões de acesso e o contrato de pré-processamento, então execute uma cópia com os mesmos pesos/configuração. Links e pesos podem ter mudado ou deixado de estar disponíveis. A reprodução do resultado no teste oculto depende do avaliador Kaggle e não é garantida por este repositório.

O código de inferência dos três notebooks selecionados está incluído, com fontes, licença e alterações documentadas em [NOTICE](../notebooks/NOTICE.md). Os pesos e dados permanecem nas fontes originais, sob seus respectivos termos. O material disponível também não permitiu provar independência de seleção de todos os componentes.

## Inputs dos três notebooks

1. Anexe os dados da competição RSNA Knee, incluindo `test.csv`, `sample_submission.csv` e séries DICOM. O código reconhece os caminhos Kaggle curtos e os montados sob `competitions/`.
2. Anexe o pacote de ativos `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`, cujo caminho está fixado nas células de inferência. Ele precisa fornecer os checkpoints/configurações esperados; permissões e disponibilidade não são garantidas pelo repositório.
3. Anexe o modelo `metaresearch/dinov2/pytorch/small/1` e os checkpoints Raptor exigidos pelo código. O runtime procura, entre outros, `raptor_ft_coatnet_v5_full_swa.pt` e `raptor_ft_coatnet_v4_full.pt` nos inputs anexados. Confira os nomes e os caminhos no notebook antes de executar.
4. Use GPU T4 ×2 e internet desligada, como nas execuções registradas. Dependências como PyTorch, timm, transformers e pydicom devem estar disponíveis no ambiente Kaggle ou em inputs de instalação offline compatíveis.
5. Importe um dos três `.ipynb` e execute as células em ordem. E109 e E130 já incluem o código-base completo E74, sem depender de executar outro notebook deste repositório.
6. Confira o `submission.csv` produzido contra o template. A quantidade de estudos acessível no teste de desenvolvimento não é uma estimativa da quantidade de exames do teste oculto.

As cópias de portfólio preservam exatamente as células de código dos snapshots locais. Foram acrescentadas células introdutórias, retirados outputs e metadados de sessões. Compilação sintática e integridade foram verificadas; a curadoria não reexecutou inferência em GPU nem produziu novos scores. As dependências de modelo não estão fixadas em um lockfile histórico completo, portanto mudanças no ambiente podem exigir reconstrução do runtime original.

## Reproduzir a análise histórica

`resultados/historico.json` contém somente IDs de experimentos, datas e scores agregados extraídos do snapshot do projeto. Não contém exames nem identificadores de estudos/pacientes. Os cinco resultados mais recentes foram reconciliados com o readback H284 do diário.

```bash
python scripts/gerar_progresso.py
```

O script gera PNG e SVG em `docs/imagens/`, preservando a estrela em E74, primeira solução a atingir o máximo público registrado. As barras contam a criação de IDs, sem contar novamente um candidato quando sua submission termina.
