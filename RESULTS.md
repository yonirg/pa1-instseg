# RESULTS.md — números reproduzíveis (1 núcleo de CPU)

Todos gerados pelos comandos do README; JSONs de origem indicados em cada seção.

## Partes 0, 1 e 2 — baseline vs. cabeça de instâncias (teste sintético, 128 imagens)

| modelo | decodificação | matching | mAP | AP50 | AP75 | erro contagem | IoU fg | Dice |
|---|---|---|---|---|---|---|---|---|
| U-Net binário (Parte 1) | limiar + CC | greedy | 0.117 | 0.162 | 0.110 | 8.922 | 0.965 | 0.982 |
| U-Net binário (Parte 1) | limiar + CC | hungarian | 0.117 | 0.162 | 0.110 | 8.922 | 0.965 | 0.982 |
| U-Net fronteira+dist (Parte 2) | limiar + CC (ingênuo) | greedy | 0.083 | 0.132 | 0.092 | 9.008 | 0.898 | 0.946 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = interior) | greedy | 0.547 | 0.847 | 0.632 | 0.891 | 0.898 | 0.946 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = interior) | hungarian | 0.547 | 0.847 | 0.632 | 0.891 | 0.898 | 0.946 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = distância, Parte 5) | greedy | 0.557 | 0.855 | 0.659 | 0.969 | 0.898 | 0.946 |

`p2_unet_boundary`: 12 épocas em 247.2 s de CPU, 482k parâmetros, melhor val mAP 0.536 (Parte 0: treina em < 5 min).

`p1_unet_binary`: 12 épocas em 236.6 s de CPU, 482k parâmetros, melhor val mAP 0.089 (Parte 0: treina em < 5 min).

Figura mAP × densidade: `runs/p1_unet_binary/eval_test_density.png` e `runs/p2_unet_boundary/eval_test_density.png`.

## Parte 3 — Eixo 1 — como recuperar resolução (mesmo encoder base=8, depth=3)

Configuração: `--epochs 12 --time-limit 100 --base 8 --n-train 300 --lr 2e-3 --alpha-max 3`, 2 seeds, média ± desvio, teste em 96 imagens.

| config | mAP | AP50 | erro contagem | IoU fg | IoU por classe (bg/int/fronteira) | épocas |
|---|---|---|---|---|---|---|
| segnet | 0.203 ± 0.120 | 0.403 ± 0.268 | 6.21 ± 2.43 | 0.845 | 0.97±0.00 / 0.35±0.35 / 0.37±0.16 | 12 |
| unet | 0.386 ± 0.034 | 0.725 ± 0.005 | 2.61 ± 1.04 | 0.822 | 0.97±0.00 / 0.66±0.14 / 0.46±0.13 | 12 |
| deeplab | 0.239 ± 0.001 | 0.577 ± 0.012 | 2.61 ± 0.02 | 0.801 | 0.96±0.00 / 0.66±0.00 / 0.43±0.01 | 12 |

Figura: `runs/ablation/axis1_map.png`

## Parte 3 — Eixo 2 — função de perda (U-Net)

Configuração: `--epochs 12 --time-limit 100 --base 8 --n-train 300 --lr 2e-3 --alpha-max 3`, 2 seeds, média ± desvio, teste em 96 imagens.

| config | mAP | AP50 | erro contagem | IoU fg | IoU por classe (bg/int/fronteira) | épocas |
|---|---|---|---|---|---|---|
| ce | 0.367 ± 0.188 | 0.529 ± 0.231 | 4.15 ± 2.23 | 0.918 | 0.98±0.01 / 0.77±0.09 / 0.32±0.30 | 12 |
| bal_ce | 0.314 ± 0.114 | 0.595 ± 0.200 | 2.22 ± 0.05 | 0.863 | 0.97±0.01 / 0.76±0.04 / 0.54±0.05 | 12 |
| focal_g1 | 0.333 ± 0.179 | 0.466 ± 0.248 | 4.90 ± 2.68 | 0.943 | 0.99±0.00 / 0.80±0.05 / 0.30±0.28 | 12 |
| focal_g2 | 0.345 ± 0.171 | 0.538 ± 0.214 | 3.57 ± 1.70 | 0.919 | 0.98±0.01 / 0.77±0.07 / 0.28±0.27 | 12 |
| focal_g5 | 0.424 ± 0.009 | 0.668 ± 0.032 | 2.16 ± 0.15 | 0.901 | 0.99±0.00 / 0.75±0.03 / 0.15±0.15 | 12 |
| bal_focal_g1 | 0.225 ± 0.138 | 0.449 ± 0.304 | 4.18 ± 1.07 | 0.817 | 0.97±0.00 / 0.61±0.18 / 0.41±0.17 | 12 |
| bal_focal_g2 | 0.386 ± 0.034 | 0.725 ± 0.005 | 2.61 ± 1.04 | 0.822 | 0.97±0.00 / 0.66±0.14 / 0.46±0.13 | 12 |
| bal_focal_g5 | 0.273 ± 0.027 | 0.569 ± 0.026 | 3.33 ± 1.91 | 0.704 | 0.98±0.00 / 0.55±0.01 / 0.32±0.00 | 12 |

Figura: `runs/ablation/axis2_map.png` e `axis2_boundary_iou.png`

## Parte 4 — inferência em mosaico (cena 3×3 tiles de 128 px, sobreposição 32, 6 cenas)

| estratégia | mAP | AP50 | erro contagem | objetos na borda quebrados |
|---|---|---|---|---|
| imagem inteira (referência) | 0.486 | 0.775 | 5.8 | 24/352 |
| tiles: colar miolo (slide 83) | 0.303 | 0.527 | 49.2 | 144/352 |
| correção A: fusão por IoU na sobreposição | 0.480 | 0.764 | 6.8 | 23/352 |
| correção B: costurar mapas, watershed global | 0.488 | 0.778 | 5.8 | 24/352 |

Figuras: `part4_border_object.png`, `part4_mosaic_full.png`.

## Parte 5 — galeria de falhas, campo receptivo e correção

Campo receptivo teórico do encoder (unet): **68 px** (jump 8). Objetos: mediana 17.8 px, p90 25.2, máx 31.5 → 0% maiores que o RF. DeepLab OS8: **196 px com atrous vs 84 px sem**, mesma resolução de saída. Figura: `part5_rf_vs_objects.png`.

Taxonomia no teste inteiro: {'fusao': 127, 'fragmento': 33, 'perdido': 47, 'fantasma': 39}. Galeria: `part5_failure1..5.png`.

| falha | idx | mAP | fusões | fragmentos | perdidos | fantasmas | contraste fg/bg | ruído | p(fronteira) no contato GT |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 95 | 0.10 | 2 | 0 | 0 | 9 | 0.14 | 0.10 | 0.552 |
| 2 | 43 | 0.16 | 2 | 1 | 0 | 0 | 0.18 | 0.05 | 0.78 |
| 3 | 84 | 0.24 | 1 | 0 | 0 | 2 | 0.14 | 0.05 | 0.686 |
| 4 | 19 | 0.30 | 4 | 2 | 2 | 0 | 0.36 | 0.08 | 0.712 |
| 5 | 123 | 0.30 | 5 | 0 | 1 | 0 | 0.40 | 0.06 | 0.723 |

**Correção 1 (fronteira 3 px no rótulo, retreino mesmo orçamento):**
antes mAP 0.547 / fusões 127 → depois mAP 0.545 / fusões 166 — **não funcionou**; o diagnóstico 'fronteira fina' estava errado (`part5_before_after.json`).

**Correção 2 (limiar do interior / tamanho do marcador):** melhor {'thr_int': 0.5, 'min_marker': 8, 'map': 0.5497729597850001, 'ap50': 0.850318688557177, 'count_err': 1.09375, 'fusao': 145, 'fragmento': 29, 'perdido': 54, 'fantasma': 24} — fusões não caem; subir o limiar perde marcadores e cria órfãos fundidos (`part5_postproc_sweep.png`).

**Correção 3 (marcadores pelos picos da distância):** objetos sem interior no rótulo = 50/1506; melhor {'marker': 'dist', 'thr_d': 0.5, 'map': 0.556, 'ap50': 0.853, 'count_err': 0.93, 'fusao': 131, 'fragmento': 25, 'perdido': 46, 'fantasma': 15} — fantasmas caem, fusões (~130) persistem em todas as variantes → contatos sem evidência fotométrica.

## Parte 6 — teste de estresse (96 imagens de teste)

| modelo | limpo | blur σ=1/2/3 | ruído σ=.05/.10/.20 | contraste ×.7/.5/.35 (+brilho) | escala 0.5× / 2× |
|---|---|---|---|---|---|
| unet | 0.550 | 0.46 / 0.42 / 0.29 | 0.53 / 0.47 / 0.30 | 0.46 / 0.27 / 0.08 | 0.23 / 0.08 |

Figuras: `part6_blur.png`, `part6_ruido.png`, `part6_brilho_contraste.png`, `part6_escala.png`.

