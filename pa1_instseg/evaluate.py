"""Avaliação de um checkpoint.

    python -m pa1_instseg.evaluate --run runs/p2_unet --split test [--naive] [--rule hungarian]

Escreve ``<run>/eval_<split>[_naive].json`` (agregado + por imagem), a figura
mAP × densidade (Parte 1, item 5) e alguns painéis de exemplo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import get_datasets
from .inference import decode_maps, foreground_of, predict_maps
from .metrics import aggregate, instance_metrics, semantic_metrics
from .models.nets import build_model


def load_run(run_dir: str | Path, device="cpu"):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["arch"], cfg["in_ch"], cfg["out_ch"], base=cfg["base"], depth=cfg["depth"],
                        context=cfg["context"], pretrained=False,
                        **({"output_stride": cfg["output_stride"], "aspp_rates": tuple(cfg["aspp_rates"])}
                           if cfg["arch"] == "deeplab" else {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg


def evaluate_dataset(model, ds, head: str, naive: bool = False, rule: str = "greedy",
                     device="cpu", limit: int | None = None, transform=None, keep_maps: bool = False,
                     decode_kw: dict | None = None):
    """``transform(image)`` opcional (corrupções da Parte 6); ``decode_kw`` vai para o decodificador."""
    decode_kw = decode_kw or {}
    per_image, maps_out = [], []
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        s = ds[i]
        img = s["image"] if transform is None else transform(s["image"])
        maps = predict_maps(model, img, head, device)
        pred = decode_maps(maps, head, naive=naive, **decode_kw)
        r = instance_metrics(pred, s["inst"], rule)
        r.update(semantic_metrics(foreground_of(maps, head), s["inst"] > 0))
        r["index"] = int(s["index"])
        per_image.append(r)
        if keep_maps:
            maps_out.append((img, s["inst"], pred, maps))
    return (aggregate(per_image), per_image, maps_out) if keep_maps else (aggregate(per_image), per_image)


def _jsonable(r: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--naive", action="store_true", help="limiar + componentes conexos (Parte 1)")
    ap.add_argument("--rule", default="greedy", choices=["greedy", "hungarian"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-panels", type=int, default=4)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--marker-source", default="interior", choices=["interior", "dist"])
    ap.add_argument("--marker-opening", type=int, default=0)
    args = ap.parse_args()
    decode_kw = ({"marker_source": args.marker_source, "marker_opening": args.marker_opening}
                 if not args.naive else {})

    model, cfg = load_run(args.run)
    if args.data_root:
        cfg["data_root"] = args.data_root
    dsets, _ = get_datasets(cfg)
    ds = dsets[args.split]
    t0 = time.time()
    agg, per_image, samples = evaluate_dataset(model, ds, cfg["head"], naive=args.naive, rule=args.rule,
                                               limit=args.limit, keep_maps=True,
                                               decode_kw=decode_kw if cfg["head"] == "boundary" else None)
    agg["rule"], agg["naive"], agg["seconds"] = args.rule, args.naive, round(time.time() - t0, 1)
    tag = (f"eval_{args.split}" + ("_naive" if args.naive else "") + (f"_{args.rule}" if args.rule != "greedy" else "")
           + ("_distmarkers" if args.marker_source == "dist" else "") + (f"_open{args.marker_opening}" if args.marker_opening else ""))
    out = Path(args.run) / f"{tag}.json"
    out.write_text(json.dumps({"aggregate": agg, "per_image": [_jsonable(r) for r in per_image]}, indent=1))
    print(json.dumps(agg, indent=1))

    from .viz import map_vs_density, panel
    map_vs_density(per_image, Path(args.run) / f"{tag}_density.png",
                   label=f"{cfg['arch']}/{cfg['head']}{' (ingênuo)' if args.naive else ''}: mAP × densidade")
    worst = np.argsort([r["map"] for r in per_image])
    for k, i in enumerate(list(worst[: args.n_panels // 2]) + list(worst[-(args.n_panels - args.n_panels // 2):])):
        img, gt, pred, maps = samples[i]
        panel(img, gt, pred, maps, head=cfg["head"], title=f"idx {per_image[i]['index']} — mAP {per_image[i]['map']:.2f}",
              path=Path(args.run) / f"{tag}_panel{k}.png")


if __name__ == "__main__":
    main()
