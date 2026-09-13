"""Parte 5 — diagnóstico revisado: varredura do pós-processamento.

    python scripts/part5_postproc_sweep.py --run runs/p2_unet_boundary

Se engrossar a fronteira no rótulo não reduz fusões, a hipótese seguinte é que
o *interior* previsto vaza pelo contato (p_int > 0.5 numa faixa de 1–2 px)
e junta dois marcadores. A mudança sugerida é no decodificador: subir o limiar
do interior (``thr_int``) e/ou exigir marcadores maiores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa1_instseg.data import CachedDataset, get_datasets  # noqa: E402
from pa1_instseg.evaluate import load_run  # noqa: E402
from pa1_instseg.inference import predict_maps  # noqa: E402
from pa1_instseg.metrics import aggregate, instance_metrics  # noqa: E402
from pa1_instseg.postproc import watershed_instances  # noqa: E402
from pa1_instseg.viz import curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from part5_failures import diagnose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p2_unet_boundary")
    ap.add_argument("--thr-int", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--min-marker", type=int, nargs="+", default=[3, 8])
    args = ap.parse_args()
    model, cfg = load_run(args.run)
    dsets, _ = get_datasets(cfg)
    test = CachedDataset(dsets["test"])
    maps = [(s["image"], s["inst"], predict_maps(model, s["image"], cfg["head"])) for s in test.items]

    rows = []
    for mm in args.min_marker:
        for t in args.thr_int:
            rs, tax = [], {"fusao": 0, "fragmento": 0, "perdido": 0, "fantasma": 0}
            for img, gt, m in maps:
                pred = watershed_instances(m[:3], m[3], thr_int=t, min_marker=mm)
                rs.append(instance_metrics(pred, gt))
                d = diagnose(img, gt, pred, m, rf=68)
                for k in tax:
                    tax[k] += d[k]
            a = aggregate(rs)
            rows.append({"thr_int": t, "min_marker": mm, "map": a["map"], "ap50": a["ap50"],
                         "count_err": a["count_err"], **tax})
            print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["map"])
    out = Path(args.run) / "part5_postproc_sweep.json"
    out.write_text(json.dumps({"rows": rows, "best": best}, indent=1))
    series = {f"min_marker={mm}": [r["map"] for r in rows if r["min_marker"] == mm] for mm in args.min_marker}
    curve(args.thr_int, series, Path(args.run) / "part5_postproc_sweep.png", "limiar do interior (thr_int)",
          title="Correção no decodificador: mAP × limiar do interior")
    fus = {f"min_marker={mm}": [r["fusao"] for r in rows if r["min_marker"] == mm] for mm in args.min_marker}
    curve(args.thr_int, fus, Path(args.run) / "part5_postproc_sweep_fusoes.png", "limiar do interior (thr_int)",
          ylabel="nº de fusões (teste)", title="Fusões × limiar do interior")
    print("melhor:", best)


if __name__ == "__main__":
    main()
