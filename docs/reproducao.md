# Reprodução e proveniência

## Núcleo local

Os módulos em `src/rsna_knee` foram selecionados da esteira original: contrato de dados, separação por grupo, métricas, fusão, política de promoção e validação de submissão. Os testes correspondentes são executáveis em CPU. O manifesto `resultados/proveniencia.json` registra os hashes dos arquivos selecionados.

A configuração em `configs/` é um registro histórico da política. O CLI completo e o registro de exposição com identificadores de estudos não integram esta seleção; referências a eles documentam o fluxo original, sem constituir um pipeline completo executável neste portfólio.

O notebook `01_validacao_e_fusao.ipynb` demonstra o uso com dados sintéticos. Os valores desse exemplo não são resultados da competição.

## Execução campeã E74

- Notebook da conta: [rsna-knee-dinosaur-v4, versão 345735626](https://www.kaggle.com/code/betitanx/rsna-knee-dinosaur-v4?scriptVersionId=345735626).
- Origem pública: [RSNA Knee | DINOsaur V4, Roman Tamrazov](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4).
- Resultado registrado: 0,936 no score público.
- Ambiente de referência: Kaggle, GPU T4 ×2 e internet desligada; checkpoints e dependências precisam estar anexados previamente.
- A referência arquivada contém múltiplos braços visuais, incluindo DINOv2 e CoAtNet; o nome histórico de vários experimentos menciona DINOv3 e não basta para identificar todos os backbones efetivamente carregados.

Para tentar reproduzir, abra a versão fixada na conta Kaggle, confira os inputs originais, as permissões de acesso e o contrato de pré-processamento, então execute uma cópia com os mesmos pesos/configuração. Links e pesos podem ter mudado ou deixado de estar disponíveis. A reprodução do resultado no teste oculto depende do avaliador Kaggle e não é garantida por este repositório.

O código e os pesos integrais de terceiros não foram incorporados a esta seleção: permanecem nas fontes originais e sob seus respectivos termos. O material disponível também não permitiu provar independência de seleção de todos os componentes.

## Reproduzir a análise histórica

`resultados/historico.json` contém somente IDs de experimentos, datas e scores agregados extraídos do snapshot do projeto. Não contém exames nem identificadores de estudos/pacientes. Os cinco resultados mais recentes foram reconciliados com o readback H284 do diário.

```bash
python scripts/gerar_progresso.py
```

O script gera PNG e SVG em `docs/imagens/`, preservando a estrela em E74, primeira solução a atingir o máximo público registrado. As barras contam a criação de IDs, sem contar novamente um candidato quando sua submission termina.
