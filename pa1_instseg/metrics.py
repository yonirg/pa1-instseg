"""Métricas — tudo implementado aqui (sem AP de biblioteca).

Semântico: IoU e Dice do foreground, por imagem.

Instância (Parte 1, item 3):
  para cada limiar τ ∈ {0.50, 0.55, …, 0.95}
    1. matriz de IoU (n_pred × n_gt) via ``np.bincount`` sobre os pares de ids;
    2. **matching** 1-para-1 (regra explícita, item 4):
       - ``greedy``    — ordena todos os pares por IoU decrescente e aceita o par
                         se ambos ainda estão livres e IoU ≥ τ;
       - ``hungarian`` — ``scipy.optimize.linear_sum_assignment`` maximizando a
                         soma de IoU; depois descarta pares com IoU < τ.
       Diferença prática: o guloso pode "roubar" um GT com um pred grande que
       depois deixa outro pred sem par; o húngaro maximiza a soma total. Nos
       nossos testes a diferença de mAP fica < 0.01; **padrão = greedy** (é o
       que o Kaggle DSB2018 usa na prática, e é determinístico e barato).
    3. TP/FP/FN;
    4. AP_τ = TP / (TP + FP + FN)   — definição do DSB2018 (não há score de
       confiança por instância em Trilha A, então não há curva PR a integrar).
  mAP = média das AP_τ nos 10 limiares.
  Também: erro absoluto de contagem |n_pred − n_gt| por imagem.

Agregação: reportamos (a) média por imagem das AP_τ e (b) o "dataset-level"
(soma TP/FP/FN e depois razão). Imagens sem GT e sem predição contam AP=1.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.05), 2)


# ----------------------------------------------------------------------------- semântico
def semantic_metrics(pred_fg: np.ndarray, gt_fg: np.ndarray) -> dict:
    pred_fg, gt_fg = pred_fg.astype(bool), gt_fg.astype(bool)
    inter = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    s = pred_fg.sum() + gt_fg.sum()
    return {"iou": float(inter / union) if union else 1.0,
            "dice": float(2 * inter / s) if s else 1.0}


# ----------------------------------------------------------------------------- instância
def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(n_pred, n_gt) com IoU entre cada instância prevista e cada verdadeira."""
    n_p, n_g = int(pred.max()), int(gt.max())
    if n_p == 0 or n_g == 0:
        return np.zeros((n_p, n_g), dtype=np.float64)
    pair = pred.astype(np.int64).ravel() * (n_g + 1) + gt.astype(np.int64).ravel()
    inter = np.bincount(pair, minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
    area_p = inter.sum(1, keepdims=True)
    area_g = inter.sum(0, keepdims=True)
    union = area_p + area_g - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou[1:, 1:]


def match(iou: np.ndarray, thr: float, rule: str = "greedy") -> list[tuple[int, int]]:
    n_p, n_g = iou.shape
    if n_p == 0 or n_g == 0:
        return []
    if rule == "greedy":
        ii, jj = np.nonzero(iou >= thr)
        order = np.argsort(-iou[ii, jj], kind="stable")
        used_p, used_g, pairs = set(), set(), []
        for k in order:
            i, j = int(ii[k]), int(jj[k])
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j); pairs.append((i, j))
        return pairs
    if rule == "hungarian":
        rows, cols = linear_sum_assignment(-iou)
        return [(int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] >= thr]
    raise ValueError(rule)


def instance_metrics(pred: np.ndarray, gt: np.ndarray, rule: str = "greedy",
                     thresholds=THRESHOLDS) -> dict:
    iou = iou_matrix(pred, gt)
    n_p, n_g = iou.shape
    tp = np.zeros(len(thresholds), dtype=int)
    for k, t in enumerate(thresholds):
        tp[k] = len(match(iou, t, rule))
    fp, fn = n_p - tp, n_g - tp
    denom = tp + fp + fn
    ap = np.where(denom > 0, tp / np.maximum(denom, 1), 1.0)
    return {"ap": ap, "map": float(ap.mean()), "tp": tp, "fp": fp, "fn": fn,
            "n_pred": n_p, "n_gt": n_g, "count_err": abs(n_p - n_g),
            "ap50": float(ap[0]), "ap75": float(ap[5])}


def aggregate(results: list[dict]) -> dict:
    """Média por imagem + agregado dataset-level."""
    if not results:
        return {}
    ap = np.stack([r["ap"] for r in results])
    tp = np.stack([r["tp"] for r in results]).sum(0)
    fp = np.stack([r["fp"] for r in results]).sum(0)
    fn = np.stack([r["fn"] for r in results]).sum(0)
    ap_ds = tp / np.maximum(tp + fp + fn, 1)
    out = {
        "map": float(ap.mean()), "ap50": float(ap[:, 0].mean()), "ap75": float(ap[:, 5].mean()),
        "map_dataset": float(ap_ds.mean()),
        "ap_per_thr": ap.mean(0).round(4).tolist(),
        "count_err": float(np.mean([r["count_err"] for r in results])),
        "n_images": len(results),
    }
    if "iou" in results[0]:
        out["iou"] = float(np.mean([r["iou"] for r in results]))
        out["dice"] = float(np.mean([r["dice"] for r in results]))
    return out
