"""Fábrica de datasets. Splits:

* synthetic — train/val/test são faixas de seeds disjuntas (0, 1e6, 2e6);
* dsb       — split estratificado por modalidade (``dsb2018.stratified_split``).
"""
from __future__ import annotations

import numpy as np
import torch


def get_datasets(cfg: dict):
    if cfg["dataset"] == "synthetic":
        from .synthetic import SyntheticEllipses
        kw = dict(size=cfg.get("size", 128), boundary_thickness=cfg.get("boundary_thickness", 2),
                  boundary_mode=cfg.get("boundary_mode", "touching"),
                  n_min=cfg.get("n_min", 5), n_max=cfg.get("n_max", 20))
        return {
            "train": SyntheticEllipses(cfg.get("n_train", 400), seed=0, **kw),
            "val": SyntheticEllipses(cfg.get("n_val", 64), seed=1_000_000, **kw),
            "test": SyntheticEllipses(cfg.get("n_test", 128), seed=2_000_000, **kw),
        }, 1
    if cfg["dataset"] == "dsb":
        from .dsb2018 import DSB2018, index_dataset, stratified_split
        items = index_dataset(cfg["data_root"])
        split = stratified_split(items, seed=cfg.get("split_seed", 0),
                                 holdout_modality=cfg.get("holdout_modality"))
        kw = dict(boundary_thickness=cfg.get("boundary_thickness", 2),
                  boundary_mode=cfg.get("boundary_mode", "touching"))
        crop = cfg.get("crop", 256)
        return {
            "train": DSB2018(split["train"], crop=crop, augment=True, **kw),
            "val": DSB2018(split["val"], crop=crop, augment=False, **kw),
            "test": DSB2018(split["test"], crop=None, augment=False, **kw),   # imagem inteira
        }, 3
    raise ValueError(cfg["dataset"])


class CachedDataset:
    """Materializa um dataset em memória (a geração sintética custa ~15 ms/amostra,
    o que domina o passo de treino em CPU). ``augment`` aplica rot90/flip na hora."""

    def __init__(self, ds, augment: bool = False, seed: int = 0):
        self.items = [ds[i] for i in range(len(ds))]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        if not self.augment:
            return s
        k = int(self.rng.integers(4)); flip = self.rng.random() < 0.5
        out = {}
        for key, v in s.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                a = np.rot90(v, k, axes=(-2, -1))
                a = a[..., ::-1] if flip else a
                out[key] = np.ascontiguousarray(a)
            else:
                out[key] = v
        return out


def collate(samples: list[dict]) -> dict:
    out = {}
    for k in samples[0]:
        v = [s[k] for s in samples]
        if k in ("index", "inst"):
            out[k] = v  # inst: tamanhos podem diferir (DSB teste); fica como lista
        else:
            out[k] = torch.from_numpy(np.stack(v))
    return out


def iterate(ds, batch_size: int, shuffle: bool, rng: np.random.Generator):
    idx = np.arange(len(ds))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield collate([ds[int(j)] for j in idx[i:i + batch_size]])
