| modelo | mAP | AP50 | erro contagem | mAP 0–9 núcleos | mAP 10–24 núcleos | mAP 25–49 núcleos | mAP 50–99 núcleos | mAP ≥100 núcleos | fluorescence | histology | brightfield |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Parte 1: U-Net binária + limiar/CC | 0.477 | 0.656 | 9.7 | 0.56 | 0.57 | 0.45 | 0.30 | 0.50 | 0.54 | 0.24 | 0.00 |
| Parte 2: mesma rede, decodificação ingênua (CC) | 0.442 | 0.620 | 11.5 | 0.51 | 0.55 | 0.40 | 0.29 | 0.43 | 0.50 | 0.21 | 0.03 |
| Parte 2: fronteira+distância + watershed | 0.502 | 0.706 | 7.2 | 0.54 | 0.59 | 0.47 | 0.35 | 0.55 | 0.57 | 0.23 | 0.02 |
| Parte 2: watershed com marcadores da distância | 0.504 | 0.719 | 7.0 | 0.54 | 0.60 | 0.47 | 0.36 | 0.55 | 0.57 | 0.25 | 0.02 |
| Parte 2 com α automático (antes da Parte 5) | 0.234 | 0.574 | 9.3 | 0.19 | 0.30 | 0.20 | 0.20 | 0.27 | 0.26 | 0.14 | 0.09 |

Imagens por faixa: {'0–9': 12, '10–24': 32, '25–49': 27, '50–99': 20, '≥100': 10}; por modalidade: {'fluorescence': 82, 'histology': 16, 'brightfield': 3}.
