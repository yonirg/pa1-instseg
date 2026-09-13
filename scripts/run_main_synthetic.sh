#!/usr/bin/env bash
# Parte 0/2: modelo final (cabeça de fronteira+distância) e Parte 1: baseline binário.
# Mesmo encoder-decoder, mesmo orçamento (< 5 min de CPU cada) — comparação justa.
set -e
cd "$(dirname "$0")/.."
COMMON="--dataset synthetic --arch unet --base 16 --depth 3 --epochs 12 --time-limit 270 --n-train 400 --n-val 64"
python -m pa1_instseg.train $COMMON --head boundary --loss balanced_focal --gamma 2 --seed 0 --out runs/p2_unet_boundary
python -m pa1_instseg.train $COMMON --head binary   --loss ce                --seed 0 --out runs/p1_unet_binary
python -m pa1_instseg.evaluate --run runs/p1_unet_binary   --split test
python -m pa1_instseg.evaluate --run runs/p2_unet_boundary --split test
python -m pa1_instseg.evaluate --run runs/p2_unet_boundary --split test --naive
python -m pa1_instseg.evaluate --run runs/p2_unet_boundary --split test --rule hungarian
