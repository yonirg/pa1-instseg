#!/usr/bin/env bash
# DSB2018 — avaliação das Partes 1, 2, 4, 5 e 6 a partir dos checkpoints de run_dsb.sh (CPU basta).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
P1=runs/dsb_p1_unet_binary
P2=runs/dsb_p2_unet_boundary
# Parte 1: baseline binário (limiar + componentes conexos), as duas regras de matching
$PY -m pa1_instseg.evaluate --run $P1 --split test
$PY -m pa1_instseg.evaluate --run $P1 --split test --rule hungarian
# Partes 1 e 2: métricas lado a lado, as duas regras de matching, decodificação ingênua na rede da Parte 2
$PY -m pa1_instseg.evaluate --run $P2 --split test
$PY -m pa1_instseg.evaluate --run $P2 --split test --rule hungarian
$PY -m pa1_instseg.evaluate --run $P2 --split test --naive
$PY -m pa1_instseg.evaluate --run $P2 --split test --marker-source dist
# Parte 4: tiles 256 com sobreposição 64 nas imagens grandes do teste (lado menor ≥ 500 px)
$PY scripts/part4_mosaic.py --run $P2 --scene dsb_large --tile 256 --overlap 64 --n-mosaics 40
# Parte 5: campo receptivo × tamanhos dos núcleos, galeria das 5 piores imagens, taxonomia de erros
$PY scripts/part5_failures.py --run $P2
# Parte 6: corrupções (3 intensidades) e escala 0,5× / 2×
$PY scripts/part6_stress.py --run $P2 --limit 101
# Parte 5: antes (α automático) × depois (α fixo 1/1/3) — mesma galeria, mesma taxonomia
$PY -m pa1_instseg.evaluate --run runs/dsb_p2_unet_boundary_autoalpha --split test
$PY scripts/part5_failures.py --run runs/dsb_p2_unet_boundary_autoalpha --fixed $P2
$PY scripts/part5_correction_dsb.py
$PY scripts/dsb_breakdown.py
$PY -c "import json,torch; from pa1_instseg.ablation import class_iou; from pa1_instseg.evaluate import load_run; from pa1_instseg.data import get_datasets; json.dump({rd: class_iou(*[(m, get_datasets(c)[0][\"test\"]) for m, c in [load_run(rd)]][0], \"boundary\", 3) for rd in (\"runs/dsb_p2_unet_boundary\", \"runs/dsb_p2_unet_boundary_autoalpha\")}, open(\"runs/dsb_class_iou_final.json\", \"w\"), indent=1)"
