# PA1 — Segmentação de instâncias com as arquiteturas da aula

Disciplina Aprendizado Profundo (FGV) · Prof. Dario Oliveira · Monitor Erick Brito

Fazemos as arquiteturas de segmentação **semântica** da aula (U-Net, SegNet, DeepLab)
produzirem rótulos **instance-aware** sem detectores de região, no **DSB2018 (Opção A)**. Trilha escolhida:
**A — fronteiras + distância + watershed**. Decoder, perdas, matching e pós-processamento
são nossos (`pa1_instseg/`); só usamos `scipy.ndimage`, `skimage.segmentation.watershed`
e `scipy.optimize.linear_sum_assignment`.

## Ambiente

```bash
python -m venv .venv && source .venv/bin/activate      # testado com Python 3.12
pip install -r requirements.txt        # torch, numpy, scipy, scikit-image, scikit-learn, matplotlib, Pillow
pip install jupyter                    # só para abrir inferencia.ipynb
```

Os resultados do DSB2018 foram gerados num **Apple M4 Pro (GPU via MPS, `--device mps`)**;
em NVIDIA/Colab use `--device cuda` (ou `DEV=cuda` nos scripts), e `--device cpu` funciona
em qualquer máquina. A Parte 0 (sintético) foi rodada em 1 núcleo de CPU.

## Dados

* **Opção A — DSB2018 / BBBC038v1** (dataset principal, Partes 1–6):

  ```bash
  mkdir -p data/stage1_train
  curl -L -o data/stage1_train.zip https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip
  unzip -q data/stage1_train.zip -d data/stage1_train     # 670 pastas <id>/{images,masks}
  ```

  **Split estratificado por modalidade** (`data/dsb2018.py`). O DSB não rotula modalidade;
  usamos uma heurística fotométrica: imagem colorida → `histology`; cinza com fundo escuro →
  `fluorescence`; cinza com fundo claro → `brightfield`. Resultado: 546 / 108 / 16 imagens,
  que batem com a composição conhecida do stage1. Split 70/15/15 com
  `train_test_split(stratify=modalidade, random_state=0)` → treino 469, val 100, teste 101, as
  três com as mesmas proporções (sem isso, com só 16 brightfield, o teste pode ficar sem a
  modalidade mais difícil). O teste é avaliado na **imagem inteira** (até 1040×1388).
  `--holdout-modality` treina sem uma modalidade (variante da Parte 6).
* **Parte 0 — sintético**: gerado na hora (`pa1_instseg/data/synthetic.py`), sem download.
  Amostra *i* é determinística (seed `base + i`); train/val/test usam faixas de seed
  disjuntas (0 / 1e6 / 2e6).

## Um comando que treina, um comando que avalia

```bash
# treina o modelo final (DSB2018): U-Net, cabeça fronteira+distância, focal balanceada γ=2 (~6 min no M4 Pro)
python -m pa1_instseg.train --dataset dsb --data-root data --arch unet --base 32 --depth 4 \
    --head boundary --loss balanced_focal --gamma 2 --alpha-max 3 --epochs 30 --val-limit 50 \
    --device mps --out runs/dsb_p2_unet_boundary

# avalia no teste: IoU/Dice, AP@[.50:.95], mAP, erro de contagem, figura mAP×densidade
python -m pa1_instseg.evaluate --run runs/dsb_p2_unet_boundary --split test
```

Opções úteis do `evaluate`: `--naive` (limiar + componentes conexos), `--rule hungarian`,
`--marker-source dist` (marcadores pelos picos da distância). `inferencia.ipynb` recebe o
caminho de uma imagem qualquer e devolve a máscara colorida e a contagem, usando o checkpoint
`runs/dsb_p2_unet_boundary/model.pt` (30 MB, versionado).

## Reproduzir cada parte

| Parte | Comando | Saída |
|---|---|---|
| 0 — teste unitário sintético | `scripts/run_main_synthetic.sh` (treina em < 5 min de CPU) | `runs/p2_unet_boundary/`, `runs/p1_unet_binary/` |
| 1, 2, 3 — treinos no DSB2018 | `scripts/run_dsb.sh` (baseline binário, modelo final, Eixos 1 e 2 com 2 seeds) | `runs/dsb_p1_unet_binary/`, `runs/dsb_p2_unet_boundary/`, `runs/dsb_ablation/axis*_table.md` |
| 1, 2, 4, 5, 6 — avaliações no DSB2018 | `scripts/run_dsb_parts.sh` | `eval_test*.json`, `*_density.png`, `part4_*`, `part5_*`, `part6_*` em `runs/dsb_p2_unet_boundary/` |
| tabelas | `python scripts/collect_results.py` | `RESULTS.md` |

## Decisões de projeto (as três coisas que a aula não entregou)

1. **O que a rede prevê** — 4 canais: 3 classes (fundo / interior / *fronteira entre
   instâncias*) + distância ao fundo normalizada **por instância** (∈ [0,1]). A fronteira é
   gerada do mapa de ids com max/min-filter: pixel de objeto cuja vizinhança (2t+1)² contém
   outro id (t = 2 px). Fundo não conta — é exatamente o pixel que a semântica não separa.
2. **Qual perda** — CE / CE balanceada (α = frequência inversa, EMA no batch, com teto
   `--alpha-max`) / focal / focal balanceada para as classes; L1 para a distância.
3. **Como decodificar** — marcadores = componentes do interior (ou picos da distância,
   `--marker-source dist`); máscara = interior ∪ fronteira; elevação = −distância; watershed.
   Foreground sem marcador vira instância própria (não perde objetos pequenos).

**Matching (Parte 1, item 4)** — padrão *guloso por IoU decrescente*: ordena todos os pares
(pred, GT) por IoU e aceita se ambos estão livres e IoU ≥ τ. Alternativa *húngara*
(`linear_sum_assignment` maximizando a soma de IoU, depois filtra IoU ≥ τ). Nos nossos
testes os dois dão o mesmo mAP até a 3ª casa. **AP_τ = TP/(TP+FP+FN)** (definição do
DSB2018; sem score por instância não há curva PR), mAP = média em τ ∈ {0,50,…,0,95}.

## Estrutura

```
pa1_instseg/
  data/synthetic.py   dados sintéticos (Parte 0)      data/dsb2018.py  loader + split por modalidade
  data/targets.py     ids → sem2 / sem3 / distância    models/nets.py   U-Net, SegNet, DeepLab(+ASPP), ResNet34-UNet
  models/blocks.py    ConvBlock, ParseNet, PSP, ASPP   losses.py        CE, balanceada, focal, L1/L2
  postproc.py         CC ingênuo, watershed            metrics.py       IoU matrix, matching, AP, mAP, contagem
  tiling.py           tiles + 2 correções (Parte 4)    receptive_field.py  RF teórico (slides 35–38)
  train.py / evaluate.py / ablation.py / infer.py / inference.py / viz.py
scripts/  run_dsb.sh  run_dsb_parts.sh  run_main_synthetic.sh  part4_mosaic.py  part5_failures.py
          part5_postproc_sweep.py  part6_stress.py  collect_results.py
runs/     dsb_* = DSB2018 (Partes 1–6); p1_*, p2_*, ablation, p5_* = sintético (Parte 0)
```

Referências de ideias (reescritas, não clonadas): rótulo de fronteira entre células e
watershed com marcadores (Ronneberger et al. 2015, U-Net; kernels do DSB2018 "3-class");
regressão de distância por instância (Naylor et al. 2019); mAP no formato DSB2018.
