"""Treino — um comando para qualquer configuração.

    python -m pa1_instseg.train --dataset synthetic --head boundary --arch unet \
        --loss balanced_focal --gamma 2 --seed 0 --time-limit 240 --out runs/p2_unet

Salva em ``--out``: ``config.json``, ``model.pt`` (melhor val mAP), ``history.json``.
``run(cfg)`` é reutilizado por ``ablation.py``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import CachedDataset, get_datasets, iterate
from .evaluate import evaluate_dataset
from .losses import MultiHeadLoss
from .models.nets import build_model

DEFAULTS = dict(
    dataset="synthetic", data_root="data", head="boundary", arch="unet", base=16, depth=3,
    context="none", loss="balanced_focal", gamma=2.0, alpha="auto", alpha_max=None, dist_loss="l1", dist_weight=1.0,
    output_stride=8, aspp_rates=(2, 4, 8), pretrained=True,
    seed=0, epochs=30, time_limit=240.0, batch_size=8, lr=1e-3, weight_decay=1e-4,
    n_train=400, n_val=64, n_test=128, size=128, n_min=5, n_max=20,
    boundary_thickness=2, boundary_mode="touching", crop=256, holdout_modality=None,
    eval_every=1, out="runs/tmp", device="cpu", threads=1, cache=True, val_limit=None,
)


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)


def run(cfg: dict, verbose: bool = True) -> dict:
    cfg = {**DEFAULTS, **cfg}
    torch.set_num_threads(cfg["threads"])
    set_seed(cfg["seed"])
    out_dir = Path(cfg["out"]); out_dir.mkdir(parents=True, exist_ok=True)

    dsets, in_ch = get_datasets(cfg)
    # DSB: sem cache no treino — o recorte aleatório 256 e a augmentação são refeitos a cada época
    use_cache = cfg["cache"] and cfg["dataset"] == "synthetic"
    train_ds = CachedDataset(dsets["train"], augment=True, seed=cfg["seed"]) if use_cache else dsets["train"]
    val_ds = CachedDataset(dsets["val"]) if cfg["cache"] else dsets["val"]

    alpha = cfg["alpha"]
    if isinstance(alpha, str) and "," in alpha:   # --alpha 1,1,3 → pesos fixos por classe (fundo, interior, fronteira)
        alpha = [float(a) for a in alpha.split(",")]
    criterion = MultiHeadLoss(cfg["head"], cfg["loss"], cfg["gamma"], alpha,
                              cfg["dist_loss"], cfg["dist_weight"], alpha_max=cfg["alpha_max"]).to(cfg["device"])
    cfg["in_ch"], cfg["out_ch"] = in_ch, criterion.out_channels
    extra = {"output_stride": cfg["output_stride"], "aspp_rates": tuple(cfg["aspp_rates"])} if cfg["arch"] == "deeplab" else {}
    model = build_model(cfg["arch"], in_ch, cfg["out_ch"], base=cfg["base"], depth=cfg["depth"],
                        context=cfg["context"], pretrained=cfg["pretrained"], **extra).to(cfg["device"])
    cfg["n_params"] = sum(p.numel() for p in model.parameters())
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = int(np.ceil(len(train_ds) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch,
                                                pct_start=0.15, div_factor=10, final_div_factor=50)
    rng = np.random.default_rng(cfg["seed"])
    history, best, t0 = [], -1.0, time.time()
    for epoch in range(cfg["epochs"]):
        model.train(); criterion.train()
        losses = []
        for batch in iterate(train_ds, cfg["batch_size"], True, rng):
            batch = {k: (v.to(cfg["device"]) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["image"])
            loss, logs = criterion(out, batch)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            losses.append(logs)
        rec = {"epoch": epoch + 1, "seconds": round(time.time() - t0, 1),
               **{k: float(np.mean([l[k] for l in losses])) for k in losses[0]}}
        if (epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"]:
            agg, _ = evaluate_dataset(model, val_ds, cfg["head"], device=cfg["device"], limit=cfg["val_limit"])
            rec.update({f"val_{k}": v for k, v in agg.items() if isinstance(v, (int, float))})
            if agg["map"] > best:
                best = agg["map"]
                torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch + 1,
                            "val": agg}, out_dir / "model.pt")
        history.append(rec)
        if verbose:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if time.time() - t0 > cfg["time_limit"]:
            if verbose:
                print(f"limite de tempo ({cfg['time_limit']}s) atingido na época {epoch + 1}")
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    summary = {"best_val_map": best, "epochs_run": len(history), "train_seconds": round(time.time() - t0, 1),
               "n_params": cfg["n_params"], "out": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        elif isinstance(v, tuple):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: tuple(int(x) for x in s.split(",")), default=v)
        elif k in ("val_limit",):
            ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=v)
        elif k in ("alpha_max",):
            ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if v is not None else str, default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    print(json.dumps(run(cfg), indent=1))


if __name__ == "__main__":
    main()
