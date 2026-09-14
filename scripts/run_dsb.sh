#!/usr/bin/env bash
# DSB2018 (Opção A) — Partes 1, 2 e 3 na GPU (MPS do Apple M4 Pro; troque DEV=cuda/cpu).
# Requer data/stage1_train (ver README). Tudo em sequência: no MPS, 2 processos
# dividindo a GPU ficam ~3,5× mais lentos cada (42 s/época contra 12 s sozinho).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-mps}
mkdir -p logs
COMMON="--dataset dsb --data-root data --arch unet --base 32 --depth 4 --epochs 30 --time-limit 900 --device $DEV --threads 4 --val-limit 50 --alpha-max 3"
ABL="--seeds 0 1 --out runs/dsb_ablation --test-limit 101 --dataset dsb --data-root data --base 16 --depth 4 --epochs 15 --time-limit 150 --alpha-max 3 --device $DEV --threads 4 --val-limit 40 --eval-every 3"
$PY -m pa1_instseg.train $COMMON --head boundary --loss balanced_focal --gamma 2 --out runs/dsb_p2_unet_boundary > logs/dsb_p2.log 2>&1
$PY -m pa1_instseg.train $COMMON --head binary   --loss ce                --out runs/dsb_p1_unet_binary   > logs/dsb_p1.log 2>&1
$PY -m pa1_instseg.ablation --axis 1 $ABL > logs/dsb_axis1.log 2>&1
$PY -m pa1_instseg.ablation --axis 2 $ABL > logs/dsb_axis2.log 2>&1
