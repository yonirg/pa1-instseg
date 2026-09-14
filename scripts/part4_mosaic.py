"""Parte 4 — inferência em mosaico.

    python scripts/part4_mosaic.py --run runs/p2_unet_boundary --grid 3 --tile 128 --overlap 32 --n-mosaics 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from pa1_instseg.data import get_datasets  # noqa: E402
from pa1_instseg.evaluate import load_run  # noqa: E402
from pa1_instseg.inference import decode_maps, make_mosaic, predict_maps  # noqa: E402
from pa1_instseg.metrics import aggregate, instance_metrics, iou_matrix  # noqa: E402
from pa1_instseg.tiling import (core_slices, instances_from_fused_maps, instances_tiled_fused,  # noqa: E402
                                instances_tiled_naive, tile_positions)
from pa1_instseg.viz import color_instances, to_display  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p2_unet_boundary")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--n-mosaics", type=int, default=6)
    ap.add_argument("--iou-thr", type=float, default=0.25)
    ap.add_argument("--scene", default="large", choices=["large", "mosaic", "dsb_large"],
                    help="large: cena sintética grande gerada direto (sem emendas); mosaic: colagem de imagens do teste; "
                         "dsb_large: imagens grandes do teste do DSB2018 (lado menor ≥ --min-side), sem emendas")
    ap.add_argument("--min-side", type=int, default=500)
    args = ap.parse_args()

    model, cfg = load_run(args.run)
    head = cfg["head"]
    dsets, _ = get_datasets(cfg)
    test = dsets["test"]
    predict = lambda im: predict_maps(model, im, head)
    decode = lambda maps: decode_maps(maps, head)
    strategies = {
        "imagem inteira (referência)": lambda im: decode(predict(im)),
        "tiles: colar miolo (slide 83)": lambda im: instances_tiled_naive(predict, decode, im, args.tile, args.overlap),
        "correção A: fusão por IoU na sobreposição": lambda im: instances_tiled_fused(predict, decode, im, args.tile, args.overlap, args.iou_thr),
        "correção B: costurar mapas, watershed global": lambda im: instances_from_fused_maps(predict, decode, im, args.tile, args.overlap),
    }
    per = {k: [] for k in strategies}
    border_stats = {k: {"split": 0, "n_border": 0} for k in strategies}
    example = None
    k = args.grid ** 2
    big = [i for i in range(len(test)) if min(test.raw(i)[1].shape) >= args.min_side] if args.scene == "dsb_large" else []
    n_scenes = min(args.n_mosaics, len(big)) if args.scene == "dsb_large" else args.n_mosaics
    for m in range(n_scenes):
        if args.scene == "dsb_large":
            s = test[big[m]]
            img, gt = s["image"], s["inst"].astype(np.int32)
        elif args.scene == "mosaic":
            idxs = list(range(m * k, (m + 1) * k))
            img, gt = make_mosaic(test, idxs, (args.grid, args.grid))
        else:  # cena grande com a mesma densidade de objetos por área do treino
            from pa1_instseg.data.synthetic import make_synthetic_sample
            size = args.grid * cfg.get("size", 128)
            rng = np.random.default_rng(3_000_000 + m)
            im, gt = make_synthetic_sample(rng, size=size, n_min=cfg.get("n_min", 5) * k, n_max=cfg.get("n_max", 20) * k)
            img = im[None].astype(np.float32)
        H, W = gt.shape
        # objetos de GT que cruzam a fronteira entre miolos de tiles
        border = np.zeros_like(gt, dtype=bool)
        for y0, x0 in tile_positions(H, W, args.tile, args.overlap):
            ys, xs = core_slices(y0, x0, H, W, args.tile, args.overlap)
            for e in (ys.start, ys.stop):
                if 0 < e < H: border[e - 1:e + 1, :] = True
            for e in (xs.start, xs.stop):
                if 0 < e < W: border[:, e - 1:e + 1] = True
        border_ids = np.unique(gt[border]); border_ids = border_ids[border_ids > 0]
        preds = {}
        for name, fn in strategies.items():
            pred = fn(img); preds[name] = pred
            per[name].append(instance_metrics(pred, gt))
            # um objeto de GT "quebrado" = coberto (≥20% cada) por ≥2 instâncias previstas
            iou = iou_matrix(pred, gt)
            inter = np.zeros_like(iou)
            for j in border_ids:
                g = gt == j
                for i in np.nonzero(iou[:, j - 1] > 0)[0]:
                    inter[i, j - 1] = ((pred == i + 1) & g).sum() / g.sum()
                border_stats[name]["split"] += int((inter[:, j - 1] >= 0.2).sum() >= 2)
            border_stats[name]["n_border"] += len(border_ids)
        if example is None and len(border_ids):
            example = (img, gt, preds, border, border_ids)

    summary = {name: {"map": aggregate(rs)["map"], "ap50": aggregate(rs)["ap50"],
                      "count_err": aggregate(rs)["count_err"],
                      "objetos_na_borda_quebrados": f"{border_stats[name]['split']}/{border_stats[name]['n_border']}"}
               for name, rs in per.items()}
    out = Path(args.run) / "part4_mosaic.json"
    out.write_text(json.dumps({"args": vars(args), "n_cenas": n_scenes, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    # figura: zoom num objeto que cai na fronteira entre dois miolos
    img, gt, preds, border, border_ids = example
    j = int(border_ids[np.argmax([(gt == j).sum() for j in border_ids])])
    ys, xs = np.nonzero(gt == j)
    cy, cx = int(ys.mean()), int(xs.mean()); r = 40
    sl = (slice(max(cy - r, 0), cy + r), slice(max(cx - r, 0), cx + r))
    names = list(preds)
    fig, ax = plt.subplots(1, 2 + len(names), figsize=(2.6 * (2 + len(names)), 2.9))
    ax[0].imshow(to_display(img)[sl], cmap="gray", vmin=0, vmax=1); ax[0].set_title("imagem (zoom)")
    ax[1].imshow(color_instances(gt)[sl]); ax[1].set_title("GT")
    for a in ax[:2]:
        a.contour(border[sl], levels=[0.5], colors="cyan", linewidths=0.8)
    for a, n in zip(ax[2:], names):
        a.imshow(color_instances(preds[n], 1)[sl]); a.contour(border[sl], levels=[0.5], colors="cyan", linewidths=0.8)
        a.set_title(n.split(":")[0] if ":" in n else n, fontsize=8)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Objeto na fronteira entre tiles (linha ciano = borda dos miolos)", fontsize=9)
    fig.tight_layout(); fig.savefig(Path(args.run) / "part4_border_object.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    for a, n in zip(ax, names):
        a.imshow(color_instances(preds[n], 1)); a.contour(border, levels=[0.5], colors="cyan", linewidths=0.5)
        a.set_title(f"{n}\nmAP={summary[n]['map']:.3f}", fontsize=8); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(Path(args.run) / "part4_mosaic_full.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
