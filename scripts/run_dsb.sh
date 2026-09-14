#!/usr/bin/env bash
# DSB2018 (Opção A) — treinos das Partes 1, 2, 3 e 5 na GPU (MPS do Apple M4 Pro; troque DEV=cuda/cpu).
# Requer data/stage1_train (ver README). Tudo em sequência: no MPS, processos dividindo a GPU
# ficam bem mais lentos que em fila (4 em paralelo: 92 s/época contra 12 s sozinho).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-mps}
mkdir -p logs
COMMON="--dataset dsb --data-root data --arch unet --base 32 --depth 4 --epochs 30 --time-limit 900 --device $DEV --threads 4 --val-limit 50 --alpha-max 3"
ABL="--seeds 0 1 --out runs/dsb_ablation --test-limit 101 --dataset dsb --data-root data --base 16 --depth 4 --epochs 15 --time-limit 150 --alpha 1,1,3 --device $DEV --threads 4 --val-limit 40 --eval-every 3"
# Parte 2, 1ª versão (= "antes" da Parte 5): α automático por frequência inversa
$PY -m pa1_instseg.train $COMMON --head boundary --loss balanced_focal --gamma 2 --out runs/dsb_p2_unet_boundary_autoalpha > logs/dsb_p2_autoalpha.log 2>&1
# Parte 2, modelo final (= "depois" da Parte 5): α fixo fundo/interior/fronteira = 1/1/3
$PY -m pa1_instseg.train $COMMON --head boundary --loss balanced_focal --gamma 2 --alpha 1,1,3 --out runs/dsb_p2_unet_boundary > logs/dsb_p2.log 2>&1
# Parte 1: baseline binário, mesmo encoder-decoder e orçamento
$PY -m pa1_instseg.train $COMMON --head binary --loss ce --out runs/dsb_p1_unet_binary > logs/dsb_p1.log 2>&1
# Parte 3: Eixos 1 e 2, 2 seeds
$PY -m pa1_instseg.ablation --axis 1 $ABL > logs/dsb_axis1.log 2>&1
$PY -m pa1_instseg.ablation --axis 2 $ABL > logs/dsb_axis2.log 2>&1
