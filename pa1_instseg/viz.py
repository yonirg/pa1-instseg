"""Figuras — todas reproduzíveis a partir dos JSONs em ``runs/``."""
from __future__ import annotations

import sys

import matplotlib
if "ipykernel" not in sys.modules:   # scripts salvam PNG sem display; no Jupyter mantém o backend inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def color_instances(inst: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(inst.max())
    lut = np.vstack([[0, 0, 0], rng.uniform(0.25, 1.0, size=(max(n, 1), 3))])
    rgb = lut[inst]
    rgb[find_boundaries(inst, mode="inner")] *= 0.55
    return rgb


def to_display(image: np.ndarray) -> np.ndarray:
    im = image.transpose(1, 2, 0) if image.ndim == 3 else image
    return im[..., 0] if (im.ndim == 3 and im.shape[2] == 1) else im


def panel(image, gt, pred, maps=None, head="boundary", title="", path=None):
    cols = 3 + (2 if (maps is not None and head == "boundary") else (1 if maps is not None else 0))
    fig, ax = plt.subplots(1, cols, figsize=(2.6 * cols, 2.8))
    ax[0].imshow(to_display(image), cmap="gray", vmin=0, vmax=1); ax[0].set_title("imagem")
    ax[1].imshow(color_instances(gt)); ax[1].set_title(f"GT ({int(gt.max())})")
    ax[2].imshow(color_instances(pred, 1)); ax[2].set_title(f"pred ({int(pred.max())})")
    if maps is not None:
        if head == "boundary":
            ax[3].imshow(maps[2], cmap="magma", vmin=0, vmax=1); ax[3].set_title("p(fronteira)")
            ax[4].imshow(maps[3], cmap="viridis", vmin=0, vmax=1); ax[4].set_title("distância")
        else:
            ax[3].imshow(maps[1], cmap="magma", vmin=0, vmax=1); ax[3].set_title("p(objeto)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return fig


def map_vs_density(per_image: list[dict], path, key="n_gt", metric="map", label=None, bins=None):
    x = np.array([r[key] for r in per_image]); y = np.array([r[metric] for r in per_image])
    if bins is None:
        bins = np.linspace(x.min(), x.max() + 1e-6, 6)
    idx = np.digitize(x, bins) - 1
    centers, means, stds = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        centers.append((bins[b] + bins[b + 1]) / 2); means.append(y[m].mean()); stds.append(y[m].std())
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(x, y, s=8, alpha=0.25, color=PALETTE[0], label="por imagem")
    ax.errorbar(centers, means, yerr=stds, marker="o", color=PALETTE[3], label="média por faixa")
    ax.set_xlabel("nº de objetos na imagem (densidade)"); ax.set_ylabel(metric)
    ax.set_title(label or f"{metric} cai com a densidade")
    ax.legend(); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def ablation_bars(table: dict[str, tuple[float, float]], path, title, ylabel="mAP"):
    names = list(table); m = [table[n][0] for n in names]; s = [table[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(names)), 3.2))
    ax.bar(names, m, yerr=s, capsize=4, color=PALETTE[: len(names)])
    for i, v in enumerate(m):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, max(1.0, max(m) + 0.1))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def rf_histogram(diams: np.ndarray, rf_lines: dict[str, int], path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(diams, bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
    for i, (name, v) in enumerate(rf_lines.items()):
        ax.axvline(v, color=PALETTE[(i + 1) % 6], ls="--", lw=1.5, label=f"{name}: {v}px")
    ax.set_xlabel("diâmetro equivalente do objeto (px)"); ax.set_ylabel("nº de objetos")
    ax.set_title("Campo receptivo teórico vs. tamanho dos objetos"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def curve(xs, series: dict[str, list[float]], path, xlabel, ylabel="mAP", title=""):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(xs, ys, marker="o", color=PALETTE[i % 6], label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
