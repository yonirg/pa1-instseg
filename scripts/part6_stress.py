"""Parte 6 — teste de estresse: corrupções em 3 intensidades + mudança de escala.

    python scripts/part6_stress.py --run runs/p2_unet_boundary [--run2 runs/ablation/axis1_deeplab_s0]

Corrupções (aplicadas só no teste; o modelo nunca as viu):
  blur      σ ∈ {1, 2, 3} px
  ruído     σ ∈ {0.05, 0.10, 0.20}
  brilho/contraste: contraste × {0.7, 0.5, 0.35} e brilho + {0.1, 0.2, 0.3}

Escala: imagem reamostrada por 0.5× e 2×; a predição volta ao tamanho original
(nearest) antes das métricas. Uma FCN não é invariante a escala porque o campo
receptivo e os filtros são fixos em pixels — em 2× o objeto ocupa 4× mais
pixels e o interior/fronteira aprendidos com espessura fixa não escalam. O
ASPP amostra várias taxas de dilatação ao mesmo tempo, o que dá *alguma*
robustez a escala no encoder, mas não muda os rótulos de espessura fixa.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa1_instseg.data import CachedDataset, get_datasets  # noqa: E402
from pa1_instseg.evaluate import evaluate_dataset, load_run  # noqa: E402
from pa1_instseg.inference import decode_maps, predict_maps  # noqa: E402
from pa1_instseg.metrics import aggregate, instance_metrics  # noqa: E402
from pa1_instseg.viz import curve  # noqa: E402

CORRUPTIONS = {
    "blur": [1.0, 2.0, 3.0],
    "ruido": [0.05, 0.10, 0.20],
    "brilho_contraste": [1, 2, 3],
}


def corrupt(kind: str, level, seed: int = 0):
    rng = np.random.default_rng(seed)
    def f(img):
        x = img.copy()
        if kind == "blur":
            x = np.stack([ndi.gaussian_filter(c, level) for c in x])
        elif kind == "ruido":
            x = x + rng.normal(0, level, x.shape).astype(np.float32)
        elif kind == "brilho_contraste":
            c = {1: 0.7, 2: 0.5, 3: 0.35}[level]; b = {1: 0.1, 2: 0.2, 3: 0.3}[level]
            x = (x - x.mean()) * c + x.mean() + b
        return np.clip(x, 0, 1).astype(np.float32)
    return f


def scale_eval(model, ds, head: str, factor: float, limit: int) -> dict:
    rs = []
    for i in range(min(limit, len(ds))):
        s = ds[i]
        img = ndi.zoom(s["image"], (1, factor, factor), order=1)
        pred = decode_maps(predict_maps(model, img, head), head)
        pred = ndi.zoom(pred, (1 / factor, 1 / factor), order=0)[: s["inst"].shape[0], : s["inst"].shape[1]]
        if pred.shape != s["inst"].shape:  # ajuste de arredondamento
            pad = [(0, s["inst"].shape[k] - pred.shape[k]) for k in range(2)]
            pred = np.pad(pred, pad)
        rs.append(instance_metrics(pred, s["inst"]))
    return aggregate(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p2_unet_boundary")
    ap.add_argument("--run2", default=None, help="segundo modelo para comparar (ex.: DeepLab/ASPP)")
    ap.add_argument("--limit", type=int, default=96)
    args = ap.parse_args()
    runs = {"unet": args.run} if args.run2 is None else {"unet": args.run, "deeplab": args.run2}
    out_dir = Path(args.run)
    result = {}
    for name, rd in runs.items():
        model, cfg = load_run(rd)
        dsets, _ = get_datasets(cfg)
        test = CachedDataset(dsets["test"])
        base = evaluate_dataset(model, test, cfg["head"], limit=args.limit)[0]["map"]
        res = {"limpo": base, "corrupcoes": {}, "escala": {}}
        for kind, levels in CORRUPTIONS.items():
            res["corrupcoes"][kind] = [evaluate_dataset(model, test, cfg["head"], limit=args.limit,
                                                        transform=corrupt(kind, lv))[0]["map"] for lv in levels]
            print(name, kind, [round(v, 3) for v in res["corrupcoes"][kind]], flush=True)
        for f in (0.5, 1.0, 2.0):
            res["escala"][str(f)] = scale_eval(model, test, cfg["head"], f, args.limit)["map"]
        print(name, "escala", {k: round(v, 3) for k, v in res["escala"].items()}, flush=True)
        result[name] = res
    (out_dir / "part6_stress.json").write_text(json.dumps(result, indent=1))

    for kind, levels in CORRUPTIONS.items():
        series = {n: [result[n]["limpo"]] + result[n]["corrupcoes"][kind] for n in result}
        curve([0] + levels, series, out_dir / f"part6_{kind}.png",
              xlabel=f"intensidade ({kind})", title=f"Degradação do mAP — {kind}")
    curve([0.5, 1.0, 2.0], {n: [result[n]["escala"][k] for k in ("0.5", "1.0", "2.0")] for n in result},
          out_dir / "part6_escala.png", xlabel="fator de escala", title="mAP × escala (0.5×, 1×, 2×)")


if __name__ == "__main__":
    main()
