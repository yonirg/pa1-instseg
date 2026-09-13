"""Parte 5 — galeria de falhas + campo receptivo + correção.

    python scripts/part5_failures.py --run runs/p2_unet_boundary            # galeria + RF + diagnósticos
    python scripts/part5_failures.py --run runs/p2_unet_boundary --fixed runs/p5_fix   # antes/depois

Diagnóstico automático por imagem (o texto final é de vocês, mas os números
saem daqui): para cada objeto de GT, classificamos o erro como
  * ``fusao``    — um pred cobre ≥ 2 GTs (≥ 30% de cada);       → fronteira não detectada
  * ``fragmento``— um GT coberto por ≥ 2 preds (≥ 30% cada);   → interior quebrado / marcador duplo
  * ``perdido``  — GT sem nenhum pred com IoU ≥ 0.1;           → contraste / tamanho
  * ``fantasma`` — pred sem GT (IoU < 0.1 com todos).
E medimos: diâmetro dos objetos vs. campo receptivo, contraste fg/bg, nº de pares encostados.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa1_instseg.data import get_datasets  # noqa: E402
from pa1_instseg.data.targets import touching_boundary  # noqa: E402
from pa1_instseg.evaluate import evaluate_dataset, load_run  # noqa: E402
from pa1_instseg.metrics import iou_matrix  # noqa: E402
from pa1_instseg.models.nets import DeepLab  # noqa: E402
from pa1_instseg.receptive_field import object_diameters, receptive_field, rf_trace  # noqa: E402
from pa1_instseg.viz import panel, rf_histogram  # noqa: E402


def diagnose(img, gt, pred, maps, rf: int) -> dict:
    iou = iou_matrix(pred, gt)
    n_p, n_g = iou.shape
    cov_p = np.zeros_like(iou)  # fração do GT j coberta pelo pred i
    for j in range(n_g):
        g = gt == j + 1
        for i in np.nonzero(iou[:, j] > 0)[0]:
            cov_p[i, j] = ((pred == i + 1) & g).sum() / g.sum()
    fusao = int(((cov_p >= 0.3).sum(1) >= 2).sum())
    fragmento = int(((cov_p >= 0.3).sum(0) >= 2).sum())
    perdido = int((iou.max(0) < 0.1).sum()) if n_p else n_g
    fantasma = int((iou.max(1) < 0.1).sum()) if n_g else n_p
    diam = object_diameters([gt])
    fg = gt > 0
    im = img[0]
    contraste = float(im[fg].mean() - im[~fg].mean()) if fg.any() and (~fg).any() else 0.0
    ruido = float(im[~fg].std()) if (~fg).any() else 0.0
    tb = touching_boundary(gt, 1)
    return {"n_gt": n_g, "n_pred": n_p, "fusao": fusao, "fragmento": fragmento, "perdido": perdido,
            "fantasma": fantasma, "diam_min": float(diam.min()), "diam_max": float(diam.max()),
            "diam_maior_que_rf": int((diam > rf).sum()), "contraste_fg_bg": round(contraste, 3),
            "ruido_bg": round(ruido, 3), "px_fronteira_gt": int(tb.sum()),
            "p_fronteira_media_na_fronteira_gt": round(float(maps[2][tb].mean()), 3) if tb.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p2_unet_boundary")
    ap.add_argument("--fixed", default=None, help="run corrigido para o antes/depois")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    model, cfg = load_run(args.run)
    dsets, _ = get_datasets(cfg)
    test = dsets["test"]
    rd = Path(args.run)

    # --- campo receptivo teórico (obrigatório) ---
    layers = model.rf_layers()
    rf, jump = receptive_field(layers)
    insts = [test.raw(i)[1] for i in range(len(test))]
    diam = object_diameters(insts)
    rf_lines = {f"{cfg['arch']} (encoder)": rf}
    rf_info = {"arch": cfg["arch"], "rf": rf, "jump": jump, "trace": rf_trace(layers),
               "diam_median": float(np.median(diam)), "diam_p90": float(np.percentile(diam, 90)),
               "diam_max": float(diam.max()), "frac_objetos_maiores_que_rf": float((diam > rf).mean())}
    # DeepLab com e sem atrous, mesma resolução de saída (mesmo base/depth do modelo final)
    dl = DeepLab(cfg["in_ch"], cfg["out_ch"], cfg["base"], cfg["depth"], output_stride=8)
    rf_info["deeplab_os8_com_atrous"] = receptive_field(dl.rf_layers(atrous=True))[0]
    rf_info["deeplab_os8_sem_atrous"] = receptive_field(dl.rf_layers(atrous=False))[0]
    rf_lines["DeepLab OS8 c/ atrous"] = rf_info["deeplab_os8_com_atrous"]
    rf_lines["DeepLab OS8 s/ atrous"] = rf_info["deeplab_os8_sem_atrous"]
    rf_histogram(diam, rf_lines, rd / "part5_rf_vs_objects.png")
    (rd / "part5_receptive_field.json").write_text(json.dumps(rf_info, indent=1))
    print("campo receptivo:", {k: v for k, v in rf_info.items() if k != "trace"})

    # --- galeria: as n piores imagens ---
    agg, per, samples = evaluate_dataset(model, test, cfg["head"], keep_maps=True)
    order = np.argsort([r["map"] for r in per])[: args.n]
    gallery = []
    for k, i in enumerate(order):
        img, gt, pred, maps = samples[i]
        d = diagnose(img, gt, pred, maps, rf)
        d.update({"index": per[i]["index"], "map": round(per[i]["map"], 3), "ap50": round(per[i]["ap50"], 3)})
        gallery.append(d)
        panel(img, gt, pred, maps, head=cfg["head"],
              title=f"falha {k + 1}: idx {d['index']} mAP={d['map']:.2f} | fusões={d['fusao']} fragmentos={d['fragmento']} perdidos={d['perdido']} fantasmas={d['fantasma']} | contraste={d['contraste_fg_bg']:.2f} ruído={d['ruido_bg']:.2f}",
              path=rd / f"part5_failure{k + 1}.png")
    # taxonomia no teste inteiro (para escolher a correção)
    tax = {"fusao": 0, "fragmento": 0, "perdido": 0, "fantasma": 0}
    for img, gt, pred, maps in samples:
        d = diagnose(img, gt, pred, maps, rf)
        for key in tax:
            tax[key] += d[key]
    report = {"test_map": agg["map"], "taxonomia_teste": tax, "galeria": gallery}
    (rd / "part5_failures.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"taxonomia_teste": tax}, indent=1))
    for d in gallery:
        print(d)

    # --- antes/depois da correção ---
    if args.fixed:
        model2, cfg2 = load_run(args.fixed)
        agg2, per2, samples2 = evaluate_dataset(model2, test, cfg2["head"], keep_maps=True)
        tax2 = {"fusao": 0, "fragmento": 0, "perdido": 0, "fantasma": 0}
        for img, gt, pred, maps in samples2:
            d = diagnose(img, gt, pred, maps, rf)
            for key in tax2:
                tax2[key] += d[key]
        changed = {k: v for k, v in cfg2.items() if cfg.get(k) != v and k not in ("out", "n_params", "seed")}
        ba = {"antes": {"map": agg["map"], "ap50": agg["ap50"], "count_err": agg["count_err"], **tax},
              "depois": {"map": agg2["map"], "ap50": agg2["ap50"], "count_err": agg2["count_err"], **tax2},
              "mudanca": changed}
        (rd / "part5_before_after.json").write_text(json.dumps(ba, indent=1, default=str))
        print(json.dumps(ba, indent=1, default=str))
        for k, i in enumerate(order[:2]):
            idx = per[i]["index"]
            j = [r["index"] for r in per2].index(idx)
            img, gt, pred, maps = samples2[j]
            panel(img, gt, pred, maps, head=cfg2["head"],
                  title=f"DEPOIS da correção — idx {idx}: mAP {per[i]['map']:.2f} → {per2[j]['map']:.2f}",
                  path=rd / f"part5_failure{k + 1}_after.png")


if __name__ == "__main__":
    main()
