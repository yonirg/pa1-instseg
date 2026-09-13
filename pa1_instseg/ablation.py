"""Parte 3 — ablações, cada configuração com N seeds, média ± desvio.

    python -m pa1_instseg.ablation --axis 1 --seeds 0 1 --time-limit 60 --out runs/ablation
    python -m pa1_instseg.ablation --axis 2 ...     (perda: CE → CE bal. → focal γ → focal bal. γ)
    python -m pa1_instseg.ablation --axis 3 ...     (contexto global: none / image pooling / PSP)

Eixo 2 também reporta IoU da classe *fronteira* (a minoritária), que é onde o
desbalanceamento aparece. Todos os hiperparâmetros não variados vêm de
``train.DEFAULTS`` sobrescritos pelos argumentos extras da linha de comando.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CachedDataset, get_datasets
from .evaluate import evaluate_dataset, load_run
from .inference import predict_maps
from .train import DEFAULTS, run

AXES = {
    1: [("segnet", {"arch": "segnet"}), ("unet", {"arch": "unet"}), ("deeplab", {"arch": "deeplab"})],
    2: [("ce", {"loss": "ce", "gamma": 0}), ("bal_ce", {"loss": "balanced_ce", "gamma": 0}),
        ("focal_g1", {"loss": "focal", "gamma": 1}), ("focal_g2", {"loss": "focal", "gamma": 2}),
        ("focal_g5", {"loss": "focal", "gamma": 5}),
        ("bal_focal_g1", {"loss": "balanced_focal", "gamma": 1}),
        ("bal_focal_g2", {"loss": "balanced_focal", "gamma": 2}),
        ("bal_focal_g5", {"loss": "balanced_focal", "gamma": 5})],
    3: [("none", {"context": "none"}), ("image_pooling", {"context": "image_pooling"}), ("psp", {"context": "psp"})],
}


def class_iou(model, ds, head: str, n_classes: int = 3) -> list[float]:
    inter = np.zeros(n_classes); union = np.zeros(n_classes)
    for i in range(len(ds)):
        s = ds[i]
        maps = predict_maps(model, s["image"], head)
        pred = maps[:n_classes].argmax(0)
        gt = s["sem3"] if head == "boundary" else s["sem2"]
        for c in range(n_classes):
            p, g = pred == c, gt == c
            inter[c] += (p & g).sum(); union[c] += (p | g).sum()
    return (inter / np.maximum(union, 1)).round(4).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--test-limit", type=int, default=96)
    ap.add_argument("--max-new-runs", type=int, default=999, help="treina no máximo N configs por chamada (retomável)")
    args, extra = ap.parse_known_args()
    # sobrescritas livres: --base 8 --time-limit 60 --epochs 6 ...
    overrides = {}
    for i in range(0, len(extra), 2):
        k = extra[i].lstrip("-").replace("-", "_"); v = extra[i + 1]
        d = DEFAULTS.get(k)
        if isinstance(d, bool):
            overrides[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(d, (int, float, str)):
            overrides[k] = type(d)(v)
        else:  # None → tenta int, float, senão string
            try:
                overrides[k] = int(v)
            except ValueError:
                try:
                    overrides[k] = float(v)
                except ValueError:
                    overrides[k] = v

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    new_runs = 0
    for name, cfg_axis in AXES[args.axis]:
        for seed in args.seeds:
            cfg = {**overrides, **cfg_axis, "seed": seed}
            full = {**DEFAULTS, **cfg}
            # nome canônico: configs idênticas em eixos diferentes compartilham o treino
            rd = out / f"{full['arch']}_{full['loss']}_g{full['gamma']:g}_{full['context']}_s{seed}"
            cfg["out"] = str(rd)
            if not (rd / "summary.json").exists():
                if new_runs >= args.max_new_runs:
                    print("limite de treinos desta chamada atingido; rode de novo para retomar"); return
                run(cfg, verbose=False); new_runs += 1
            model, full_cfg = load_run(rd)
            dsets, _ = get_datasets(full_cfg)
            test = CachedDataset(dsets["test"])
            agg, _ = evaluate_dataset(model, test, full_cfg["head"], limit=args.test_limit)
            agg["class_iou"] = class_iou(model, test, full_cfg["head"], 3 if full_cfg["head"] == "boundary" else 2)
            agg["train_seconds"] = json.loads((rd / "summary.json").read_text())["train_seconds"]
            agg["epochs_run"] = json.loads((rd / "summary.json").read_text())["epochs_run"]
            results.setdefault(name, []).append(agg)
            print(name, seed, {k: agg[k] for k in ("map", "ap50", "count_err", "class_iou")}, flush=True)

    # agrega média ± desvio
    table = {}
    for name, rs in results.items():
        row = {}
        for k in ("map", "ap50", "ap75", "count_err", "iou", "dice"):
            v = np.array([r[k] for r in rs]); row[k] = (float(v.mean()), float(v.std()))
        ci = np.array([r["class_iou"] for r in rs]); row["class_iou"] = (ci.mean(0).tolist(), ci.std(0).tolist())
        row["epochs_run"] = float(np.mean([r["epochs_run"] for r in rs]))
        table[name] = row
    (out / f"axis{args.axis}_results.json").write_text(json.dumps({"runs": results, "table": table}, indent=1))

    lines = [f"| config | mAP | AP50 | erro contagem | IoU fg | IoU por classe (bg/int/fronteira) | épocas |",
             "|---|---|---|---|---|---|---|"]
    for name, r in table.items():
        ci = " / ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(*r["class_iou"]))
        lines.append(f"| {name} | {r['map'][0]:.3f} ± {r['map'][1]:.3f} | {r['ap50'][0]:.3f} ± {r['ap50'][1]:.3f} | "
                     f"{r['count_err'][0]:.2f} ± {r['count_err'][1]:.2f} | {r['iou'][0]:.3f} | {ci} | {r['epochs_run']:.0f} |")
    md = "\n".join(lines)
    (out / f"axis{args.axis}_table.md").write_text(md + "\n")
    print(md)
    from .viz import ablation_bars
    ablation_bars({n: r["map"] for n, r in table.items()}, out / f"axis{args.axis}_map.png",
                  title={1: "Eixo 1 — como recuperar resolução", 2: "Eixo 2 — função de perda",
                         3: "Eixo 3 — contexto global"}[args.axis] + f" (n={len(args.seeds)} seeds)")
    if args.axis == 2:
        ablation_bars({n: (r["class_iou"][0][2], r["class_iou"][1][2]) for n, r in table.items()},
                      out / "axis2_boundary_iou.png", title="Eixo 2 — IoU da classe fronteira (minoritária)", ylabel="IoU fronteira")


if __name__ == "__main__":
    main()
