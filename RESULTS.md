# RESULTS.md — números reproduzíveis

Todos gerados pelos comandos do README; JSONs de origem indicados em cada seção.

# DSB2018 / BBBC038v1 (Opção A) — Partes 1 a 6

Split estratificado por modalidade (469/100/101 imagens). Teste em imagem inteira. GPU Apple M4 Pro (MPS).

## Partes 1 e 2 — baseline vs. cabeça de instâncias (DSB2018, teste = 101 imagens inteiras)

| modelo | decodificação | matching | mAP | AP50 | AP75 | erro contagem | IoU fg | Dice |
|---|---|---|---|---|---|---|---|---|
| U-Net binário (Parte 1) | limiar + CC | greedy | 0.477 | 0.656 | 0.528 | 9.673 | 0.832 | 0.896 |
| U-Net binário (Parte 1) | limiar + CC | hungarian | 0.477 | 0.656 | 0.528 | 9.673 | 0.832 | 0.896 |
| U-Net fronteira+dist (Parte 2) | limiar + CC (ingênuo) | greedy | 0.442 | 0.620 | 0.486 | 11.515 | 0.832 | 0.902 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = interior) | greedy | 0.502 | 0.706 | 0.552 | 7.168 | 0.832 | 0.902 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = interior) | hungarian | 0.502 | 0.706 | 0.552 | 7.168 | 0.832 | 0.902 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = distância, Parte 5) | greedy | 0.504 | 0.719 | 0.551 | 7.040 | 0.832 | 0.902 |

`dsb_p2_unet_boundary`: 30 épocas em 350.3 s (GPU MPS), 7763k parâmetros, melhor val mAP 0.358 .

`dsb_p1_unet_binary`: 30 épocas em 384.7 s (GPU MPS), 7763k parâmetros, melhor val mAP 0.342 .

Figura mAP × densidade: `runs/dsb_p1_unet_binary/eval_test_density.png` e `runs/dsb_p2_unet_boundary/eval_test_density.png`.

### mAP por densidade de núcleos e por modalidade (Parte 1, item 5)

| modelo | mAP | AP50 | erro contagem | mAP 0–9 núcleos | mAP 10–24 núcleos | mAP 25–49 núcleos | mAP 50–99 núcleos | mAP ≥100 núcleos | fluorescence | histology | brightfield |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Parte 1: U-Net binária + limiar/CC | 0.477 | 0.656 | 9.7 | 0.56 | 0.57 | 0.45 | 0.30 | 0.50 | 0.54 | 0.24 | 0.00 |
| Parte 2: mesma rede, decodificação ingênua (CC) | 0.442 | 0.620 | 11.5 | 0.51 | 0.55 | 0.40 | 0.29 | 0.43 | 0.50 | 0.21 | 0.03 |
| Parte 2: fronteira+distância + watershed | 0.502 | 0.706 | 7.2 | 0.54 | 0.59 | 0.47 | 0.35 | 0.55 | 0.57 | 0.23 | 0.02 |
| Parte 2: watershed com marcadores da distância | 0.504 | 0.719 | 7.0 | 0.54 | 0.60 | 0.47 | 0.36 | 0.55 | 0.57 | 0.25 | 0.02 |
| Parte 2 com α automático (antes da Parte 5) | 0.234 | 0.574 | 9.3 | 0.19 | 0.30 | 0.20 | 0.20 | 0.27 | 0.26 | 0.14 | 0.09 |

Imagens por faixa: {'0–9': 12, '10–24': 32, '25–49': 27, '50–99': 20, '≥100': 10}; por modalidade: {'fluorescence': 82, 'histology': 16, 'brightfield': 3}.

Figura: `runs/dsb_breakdown_density.png`.

IoU por classe no teste (fundo / interior / fronteira): modelo final 0.98 / 0.83 / 0.13; com α automático 0.95 / 0.70 / 0.08 (`runs/dsb_class_iou_final.json`).

## Parte 3 — Eixo 1 — como recuperar resolução (mesmo encoder)

Configuração: `--dataset dsb --base 16 --depth 4 --epochs 15 --time-limit 150 --alpha 1,1,3 (modelo final: base 32, 30 épocas)`, 2 seeds, média ± desvio.

| config | mAP | AP50 | erro contagem | IoU fg | IoU por classe (bg/int/fronteira) | épocas |
|---|---|---|---|---|---|---|
| segnet | 0.326 ± 0.010 | 0.558 ± 0.001 | 11.37 ± 0.84 | 0.750 | 0.97±0.00 / 0.79±0.00 / 0.00±0.00 | 15 |
| unet | 0.370 ± 0.000 | 0.600 ± 0.013 | 13.73 ± 3.55 | 0.768 | 0.97±0.00 / 0.80±0.00 / 0.00±0.00 | 15 |
| deeplab | 0.349 ± 0.005 | 0.574 ± 0.003 | 10.90 ± 0.96 | 0.762 | 0.97±0.00 / 0.79±0.00 / 0.00±0.00 | 15 |

Figura: `runs/dsb_ablation/axis1_map.png`

**Leitura do Eixo 1.** Com o mesmo encoder e o mesmo orçamento, **skip connections (U-Net) > atrous + ASPP (DeepLab OS8) > índices de pooling (SegNet)**, e as duas seeds concordam (desvio ≤ 0,01 no mAP). O que decide é recuperar detalhe fino: o contato entre dois núcleos tem 1–3 px. A U-Net traz de volta, pela skip, os mapas de alta resolução do encoder. A SegNet só devolve *onde* estava o máximo, não o conteúdo. A DeepLab decodifica em 1/8 e sobe ×8 por interpolação bilinear, borrando exatamente esses pixels. O campo receptivo maior da DeepLab (356 px) não ajuda: os núcleos têm mediana de 21 px e máximo de 88 px. Nesse regime curto (base 16, 15 épocas), a classe fronteira fica com IoU ≈ 0 nas três arquiteturas, e a separação vem do mapa de distância + watershed. No modelo final (base 32, 30 épocas), o IoU da fronteira chega a 0,13.

## Parte 4 — inferência em mosaico (23 imagens grandes do teste (lado menor ≥ 500 px); tiles de 256 px, sobreposição 64)

| estratégia | mAP | AP50 | erro contagem | objetos na borda quebrados |
|---|---|---|---|---|
| imagem inteira (referência) | 0.493 | 0.674 | 15.3 | 18/650 |
| tiles: colar miolo (slide 83) | 0.356 | 0.541 | 31.1 | 202/650 |
| correção A: fusão por IoU na sobreposição | 0.492 | 0.669 | 15.2 | 15/650 |
| correção B: costurar mapas, watershed global | 0.493 | 0.670 | 14.9 | 17/650 |

Figuras: `part4_border_object.png`, `part4_mosaic_full.png`.

## Parte 5 — galeria de falhas, campo receptivo e correção

Campo receptivo teórico do encoder (unet): **140 px** (jump 16). Objetos: mediana 20.7 px, p90 36.0, máx 87.7 → 0% maiores que o RF. DeepLab OS8: **356 px com atrous vs 116 px sem**, mesma resolução de saída. Figura: `part5_rf_vs_objects.png`.

Taxonomia no teste inteiro: {'fusao': 339, 'fragmento': 68, 'perdido': 134, 'fantasma': 685}. Galeria: `part5_failure1..5.png`.

| falha | idx | mAP | fusões | fragmentos | perdidos | fantasmas | contraste fg/bg | ruído | p(fronteira) no contato GT |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 98 | 0.01 | 6 | 1 | 45 | 71 | -0.42 | 0.12 | 0.15 |
| 2 | 16 | 0.02 | 7 | 2 | 4 | 78 | -0.41 | 0.12 | 0.355 |
| 3 | 40 | 0.03 | 9 | 2 | 7 | 68 | -0.38 | 0.10 | 0.34 |
| 4 | 42 | 0.03 | 4 | 2 | 0 | 19 | -0.27 | 0.05 | 0.32 |
| 5 | 27 | 0.11 | 8 | 0 | 3 | 2 | -0.22 | 0.05 | 0.264 |

**Diagnóstico de cada falha** (modelo final; figuras `runs/dsb_p2_unet_boundary/part5_failure1..5.png`). Campo receptivo de 140 px contra núcleos de no máximo 88 px: nenhuma das cinco falhas é de campo receptivo.

1. **idx 98 — brightfield 1024×1024 (células escamosas).** Núcleos escuros de 10–35 px sobre citoplasma também escuro; 71 fantasmas e 45 perdidos. O brightfield tem 16 imagens no DSB inteiro (11 no treino, 2,3%), com contraste invertido em relação à fluorescência (contraste fg/bg −0,42). A rede nunca aprendeu a separar "núcleo escuro" de "citoplasma escuro": é mudança de modalidade, não de arquitetura.
2. **idx 16 — brightfield 1024×1024.** Mesmo quadro: 78 fantasmas em detritos e bordas de citoplasma. p(fronteira) quase zero nos núcleos verdadeiros (0,36 no contato), então nem a cabeça de fronteira ajuda: o erro acontece antes, no foreground.
3. **idx 40 — brightfield 1024×1024.** 68 fantasmas; além disso, o recorte de 256 px usado no treino, numa imagem de 1024 com poucos núcleos pequenos, muitas vezes não contém núcleo nenhum. Com só 11 imagens, a rede vê poucos exemplos positivos dessa modalidade.
4. **idx 42 — histologia 256×320, núcleos alongados de até 79 px.** 19 fantasmas e 4 fusões. A textura roxa dentro dos núcleos grandes gera vários máximos locais na distância prevista, e cada máximo vira marcador (fragmentos e fantasmas). No contato entre núcleos, p(fronteira) é só 0,32.
5. **idx 27 — histologia fora de foco.** 8 fusões e p(fronteira) de 0,26 no contato: núcleos borrados e sobrepostos não têm borda visível entre si. Sem evidência fotométrica, a cabeça de fronteira não tem o que detectar.

**Padrão:** as 5 piores imagens do teste são das duas modalidades minoritárias (brightfield e histologia). A tabela por modalidade confirma: mAP 0,57 em fluorescência, 0,23 em histologia e 0,02 em brightfield (`runs/dsb_breakdown.md`).


**Correção (α automático zera o peso do fundo; a rede chama o núcleo inteiro de fronteira):** frequência das classes no treino (fundo/interior/fronteira) = [0.8613, 0.1345, 0.0042]; α automático = [0.017, 0.111, 2.872] → α fixo = [1, 1, 3], mesma U-Net, mesmo orçamento (`--alpha 1,1,3`).

| | mAP | AP50 | erro contagem | fusões | fragmentos | perdidos | fantasmas |
|---|---|---|---|---|---|---|---|
| antes | 0.234 | 0.574 | 9.3 | 283 | 128 | 97 | 914 |
| depois | 0.502 | 0.706 | 7.2 | 339 | 68 | 134 | 685 |

mAP 0.234 → 0.502; fantasmas 914 → 685, fragmentos 128 → 68, fusões 283 → 339. A correção funcionou: o diagnóstico (peso do fundo) estava certo. Figuras: `runs/dsb_p2_unet_boundary_autoalpha/part5_failure1..5.png` (antes) e `part5_failure1_after.png`, `part5_failure2_after.png` (depois).

## Parte 6 — teste de estresse (DSB2018, teste = 101 imagens inteiras)

| modelo | limpo | blur σ=1/2/3 | ruído σ=.05/.10/.20 | contraste ×.7/.5/.35 (+brilho) | escala 0.5× / 2× |
|---|---|---|---|---|---|
| unet | 0.502 | 0.39 / 0.28 / 0.21 | 0.20 / 0.05 / 0.01 | 0.48 / 0.38 / 0.20 | 0.36 / 0.41 |

Figuras: `part6_blur.png`, `part6_ruido.png`, `part6_brilho_contraste.png`, `part6_escala.png`.

**Leitura (Parte 6, corrupções escolhidas; escala como extra).**
- **Ruído é o que mais derruba (σ=0,05 → mAP 0,20).** Na fluorescência (82 das 101 imagens de teste) o fundo fica em ~0,02–0,05 e muitos núcleos são tênues, então σ=0,05 já tem a ordem do contraste do objeto. O treino não teve augmentação de ruído.
- **Blur σ=1 → 0,39.** O contato entre dois núcleos tem 1–3 px, e o blur apaga exatamente a evidência que a cabeça de fronteira usa. As fusões voltam.
- **Contraste/brilho é o mais robusto (×0,7 → 0,48).** O treino já tinha jitter de contraste ±20% e de brilho ±0,1.
- **Escala (0,5× → 0,36; 2× → 0,41).** A rede é totalmente convolucional: filtros e campo receptivo são fixos em pixels, e o rótulo de fronteira tem espessura fixa (2 px). Em 0,5×, o núcleo mediano cai de 21 para ~10 px, e fronteira e interior quase somem. Em 2×, cabem núcleos de ~42 px (ainda abaixo do campo receptivo de 140 px), mas textura e espessura de borda mudam de escala. O ASPP amostra várias dilatações em paralelo e dá contexto em mais de uma escala ao encoder, mas não muda a escala da representação de saída nem os limiares do decodificador. Não medimos a DeepLab em escala por falta de tempo.


# Parte 0 — teste unitário sintético (elipses 128×128, 1 núcleo de CPU)

## Partes 1 e 2 — baseline vs. cabeça de instâncias (sintético, teste = 128 imagens)

| modelo | decodificação | matching | mAP | AP50 | AP75 | erro contagem | IoU fg | Dice |
|---|---|---|---|---|---|---|---|---|
| U-Net binário (Parte 1) | limiar + CC | greedy | 0.117 | 0.162 | 0.110 | 8.922 | 0.965 | 0.982 |
| U-Net binário (Parte 1) | limiar + CC | hungarian | 0.117 | 0.162 | 0.110 | 8.922 | 0.965 | 0.982 |
| U-Net fronteira+dist (Parte 2) | limiar + CC (ingênuo) | greedy | 0.083 | 0.132 | 0.092 | 9.008 | 0.898 | 0.946 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = interior) | greedy | 0.547 | 0.847 | 0.632 | 0.891 | 0.898 | 0.946 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = interior) | hungarian | 0.547 | 0.847 | 0.632 | 0.891 | 0.898 | 0.946 |
| U-Net fronteira+dist (Parte 2) | watershed (marcador = distância, Parte 5) | greedy | 0.557 | 0.855 | 0.659 | 0.969 | 0.898 | 0.946 |

`p2_unet_boundary`: 12 épocas em 247.2 s (CPU, 1 núcleo), 482k parâmetros, melhor val mAP 0.536 .

`p1_unet_binary`: 12 épocas em 236.6 s (CPU, 1 núcleo), 482k parâmetros, melhor val mAP 0.089 .

Figura mAP × densidade: `runs/p1_unet_binary/eval_test_density.png` e `runs/p2_unet_boundary/eval_test_density.png`.

## Parte 3 — Eixo 1 — como recuperar resolução (mesmo encoder)

Configuração: `--epochs 12 --time-limit 100 --base 8 --n-train 300 --lr 2e-3 --alpha-max 3`, 2 seeds, média ± desvio.

| config | mAP | AP50 | erro contagem | IoU fg | IoU por classe (bg/int/fronteira) | épocas |
|---|---|---|---|---|---|---|
| segnet | 0.203 ± 0.120 | 0.403 ± 0.268 | 6.21 ± 2.43 | 0.845 | 0.97±0.00 / 0.35±0.35 / 0.37±0.16 | 12 |
| unet | 0.386 ± 0.034 | 0.725 ± 0.005 | 2.61 ± 1.04 | 0.822 | 0.97±0.00 / 0.66±0.14 / 0.46±0.13 | 12 |
| deeplab | 0.239 ± 0.001 | 0.577 ± 0.012 | 2.61 ± 0.02 | 0.801 | 0.96±0.00 / 0.66±0.00 / 0.43±0.01 | 12 |

Figura: `runs/ablation/axis1_map.png`

## Parte 3 — Eixo 2 — função de perda (U-Net)

Configuração: `--epochs 12 --time-limit 100 --base 8 --n-train 300 --lr 2e-3 --alpha-max 3`, 2 seeds, média ± desvio.

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

## Parte 4 — inferência em mosaico (cena 3×3, 6 cenas; tiles de 128 px, sobreposição 32)

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

## Parte 6 — teste de estresse (sintético, teste = 128 imagens)

| modelo | limpo | blur σ=1/2/3 | ruído σ=.05/.10/.20 | contraste ×.7/.5/.35 (+brilho) | escala 0.5× / 2× |
|---|---|---|---|---|---|
| unet | 0.550 | 0.46 / 0.42 / 0.29 | 0.53 / 0.47 / 0.30 | 0.46 / 0.27 / 0.08 | 0.23 / 0.08 |

Figuras: `part6_blur.png`, `part6_ruido.png`, `part6_brilho_contraste.png`, `part6_escala.png`.

