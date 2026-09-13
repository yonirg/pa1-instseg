# PA1 — Segmentação de instâncias com as arquiteturas da aula

Disciplina Aprendizado Profundo (FGV) · Prof. Dario Oliveira · Monitor Erick Brito

Fazemos as arquiteturas de segmentação **semântica** da aula (U-Net, SegNet, DeepLab)
produzirem rótulos **instance-aware** sem detectores de região. Trilha escolhida:
**A — fronteiras + distância + watershed**. Decoder, perdas, matching e pós-processamento
são nossos (`pa1_instseg/`); só usamos `scipy.ndimage`, `skimage.segmentation.watershed`
e `scipy.optimize.linear_sum_assignment`.

## Ambiente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, numpy, scipy, scikit-image, scikit-learn, matplotlib, Pillow
```

Testado com Python 3.12, PyTorch 2.x (CPU e CUDA). Todos os números deste README foram
gerados em **1 núcleo de CPU** com orçamento de tempo por treino — em GPU basta subir
`--epochs`, `--base` e `--time-limit`.

## Dados

* **Parte 0 — sintético**: gerado na hora (`pa1_instseg/data/synthetic.py`), sem download.
  Amostra *i* é determinística (seed `base + i`); train/val/test usam faixas de seed
  disjuntas (0 / 1e6 / 2e6).
* **Opção A — DSB2018 / BBBC038v1**: baixe `stage1_train` de
  <https://bbbc.broadinstitute.org/BBBC038> (ou `kaggle competitions download -c data-science-bowl-2018`)
  e descompacte em `data/stage1_train/<id>/{images,masks}`. O split é estratificado por
  **modalidade** (heurística fotométrica: `histology` / `fluorescence` / `brightfield`,
  ver `data/dsb2018.py`), com `--holdout-modality` para a Parte 6.

## Um comando que treina, um comando que avalia

```bash
# treina (Parte 2, modelo final): U-Net, cabeça fronteira+distância, focal balanceada γ=2
python -m pa1_instseg.train --dataset synthetic --arch unet --base 16 --depth 3 \
    --head boundary --loss balanced_focal --gamma 2 --epochs 12 --time-limit 270 \
    --out runs/p2_unet_boundary

# avalia no teste: IoU/Dice, AP@[.50:.95], mAP, erro de contagem, figura mAP×densidade
python -m pa1_instseg.evaluate --run runs/p2_unet_boundary --split test
```

Para DSB2018: `--dataset dsb --data-root data --crop 256 --arch resnet34_unet --pretrained true`.
Opções úteis do `evaluate`: `--naive` (limiar + componentes conexos), `--rule hungarian`,
`--marker-source dist` (correção da Parte 5). `inferencia.ipynb` recebe o caminho de uma
imagem qualquer e devolve a máscara colorida e a contagem (checkpoint em `runs/p2_unet_boundary/model.pt`).

## Reproduzir cada parte

| Parte | Comando | Saída |
|---|---|---|
| 0 — teste unitário sintético | `scripts/run_main_synthetic.sh` (treina em < 5 min de CPU) | `runs/p2_unet_boundary/history.json` |
| 1 — baseline semântico + CC | `train ... --head binary --loss ce` → `evaluate` | `runs/p1_unet_binary/eval_test*.json`, `*_density.png` |
| 2 — cabeça de instâncias | (acima) + `evaluate --naive` para isolar o efeito do pós-processamento | `runs/p2_unet_boundary/eval_test*.json` |
| 3 — ablações (Eixos 1 e 2, 2 seeds) | `python -m pa1_instseg.ablation --axis 1` / `--axis 2` (ver flags em `RESULTS.md`) | `runs/ablation/axis*_table.md`, `axis*_map.png` |
| 4 — mosaico | `python scripts/part4_mosaic.py --run runs/p2_unet_boundary` | `part4_mosaic.json`, `part4_border_object.png` |
| 5 — galeria + campo receptivo + correção | `scripts/part5_failures.py`, `scripts/part5_postproc_sweep.py`, `train --boundary-thickness 3` | `part5_*.json/png` |
| 6 — estresse (corrupções + escala) | `python scripts/part6_stress.py --run runs/p2_unet_boundary` | `part6_*.json/png` |

`python scripts/collect_results.py` regenera `RESULTS.md` a partir dos JSONs.

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
scripts/  part4_mosaic.py  part5_failures.py  part5_postproc_sweep.py  part6_stress.py  collect_results.py
runs/     checkpoints, JSONs e figuras (tudo do README sai daqui)
```

Referências de ideias (reescritas, não clonadas): rótulo de fronteira entre células e
watershed com marcadores (Ronneberger et al. 2015, U-Net; kernels do DSB2018 "3-class");
regressão de distância por instância (Naylor et al. 2019); mAP no formato DSB2018.
