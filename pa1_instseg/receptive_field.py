"""Campo receptivo teórico (slides 35–38).

Para uma sequência de camadas (k_l, s_l, d_l):

    r_0 = 1, j_0 = 1
    r_l = r_{l-1} + (k_l − 1) · d_l · j_{l-1}       (tamanho do campo, em px de entrada)
    j_l = j_{l-1} · s_l                              (salto entre neurônios vizinhos)

Pooling 2×2/2 entra como (k=2, s=2, d=1). Uma conv 3×3 com dilatação d conta
como kernel efetivo 2d+1, que é exatamente (k−1)·d + 1.
"""
from __future__ import annotations

import numpy as np


def receptive_field(layers) -> tuple[int, int]:
    r, j = 1, 1
    for k, s, d in layers:
        r += (k - 1) * d * j
        j *= s
    return int(r), int(j)


def rf_trace(layers) -> list[tuple[int, int, int, int, int]]:
    """Tabela (k, s, d, r_l, j_l) camada a camada, para o slide."""
    r, j, rows = 1, 1, []
    for k, s, d in layers:
        r += (k - 1) * d * j
        j *= s
        rows.append((k, s, d, r, j))
    return rows


def object_diameters(inst_maps) -> np.ndarray:
    """Diâmetro equivalente 2·sqrt(área/π) de todos os objetos."""
    ds = []
    for inst in inst_maps:
        areas = np.bincount(inst.ravel())[1:]
        areas = areas[areas > 0]
        ds.append(2 * np.sqrt(areas / np.pi))
    return np.concatenate(ds) if ds else np.zeros(0)


def rf_report(model, inst_maps) -> dict:
    layers = model.rf_layers()
    r, j = receptive_field(layers)
    diam = object_diameters(inst_maps)
    out = {"rf": r, "jump": j, "n_layers": len(layers),
           "diam_median": float(np.median(diam)), "diam_p90": float(np.percentile(diam, 90)),
           "diam_max": float(diam.max()), "frac_objects_larger_than_rf": float((diam > r).mean())}
    if hasattr(model, "atrous"):
        out["rf_without_atrous"] = receptive_field(model.rf_layers(atrous=False))[0]
    return out
