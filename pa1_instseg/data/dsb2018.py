"""Opção A — DSB2018 / BBBC038v1 (``stage1_train``).

Layout esperado::

    <root>/stage1_train/<id>/images/<id>.png
    <root>/stage1_train/<id>/masks/*.png        (um PNG por núcleo)

Split estratificado por *modalidade*. O DSB não rotula modalidade, então usamos
uma heurística de fotometria (justificada na apresentação):

* ``histology``  — imagem colorida (variância entre canais alta): H&E / roxo;
* ``fluorescence`` — cinza com fundo escuro (mediana < 0.5);
* ``brightfield``  — cinza com fundo claro (mediana ≥ 0.5).

As três se comportam de forma bem diferente para a rede; estratificar garante
que teste e validação tenham as mesmas proporções. ``holdout_modality`` permite
o teste de estresse da Parte 6 (treinar sem uma modalidade e avaliar só nela).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .targets import instance_to_targets


def load_image(path: Path) -> np.ndarray:
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return im  # (H,W,3)


def load_instances(mask_dir: Path) -> np.ndarray:
    inst = None
    for k, p in enumerate(sorted(mask_dir.glob("*.png")), start=1):
        m = np.asarray(Image.open(p).convert("L")) > 127
        if inst is None:
            inst = np.zeros(m.shape, dtype=np.int32)
        inst[m & (inst == 0)] = k
    return inst


def modality_of(im: np.ndarray) -> str:
    chan_var = float(np.mean(np.var(im, axis=2)))
    if chan_var > 1e-3:
        return "histology"
    return "fluorescence" if np.median(im) < 0.5 else "brightfield"


def index_dataset(root: str | os.PathLike, cache: bool = True) -> list[dict]:
    root = Path(root) / "stage1_train"
    cache_file = root.parent / "index_stage1_train.json"
    if cache and cache_file.exists():
        return json.loads(cache_file.read_text())
    items = []
    for d in sorted(root.iterdir()):
        img = next((d / "images").glob("*.png"), None)
        if img is None:
            continue
        im = load_image(img)
        items.append({
            "id": d.name, "image": str(img), "masks": str(d / "masks"),
            "modality": modality_of(im), "height": im.shape[0], "width": im.shape[1],
        })
    if cache:
        cache_file.write_text(json.dumps(items, indent=1))
    return items


def stratified_split(items: list[dict], seed: int = 0, val: float = 0.15, test: float = 0.15,
                     holdout_modality: str | None = None) -> dict[str, list[dict]]:
    if holdout_modality:
        test_items = [it for it in items if it["modality"] == holdout_modality]
        rest = [it for it in items if it["modality"] != holdout_modality]
        tr, va = train_test_split(rest, test_size=val, random_state=seed,
                                  stratify=[it["modality"] for it in rest])
        return {"train": tr, "val": va, "test": test_items}
    strat = [it["modality"] for it in items]
    tr, tmp = train_test_split(items, test_size=val + test, random_state=seed, stratify=strat)
    va, te = train_test_split(tmp, test_size=test / (val + test), random_state=seed,
                              stratify=[it["modality"] for it in tmp])
    return {"train": tr, "val": va, "test": te}


class DSB2018:
    def __init__(self, items: list[dict], crop: int | None = 256, augment: bool = False,
                 boundary_thickness: int = 2, boundary_mode: str = "touching", seed: int = 0):
        self.items, self.crop, self.augment = items, crop, augment
        self.boundary_thickness, self.boundary_mode = boundary_thickness, boundary_mode
        self.seed = seed
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self.items)

    def raw(self, i: int):
        if i not in self._cache:
            it = self.items[i]
            self._cache[i] = (load_image(Path(it["image"])), load_instances(Path(it["masks"])))
        return self._cache[i]

    def __getitem__(self, i: int):
        img, inst = self.raw(i)
        rng = np.random.default_rng(None if self.augment else self.seed + i)
        if self.crop:
            img, inst = _random_crop(img, inst, self.crop, rng)
        if self.augment:
            k = int(rng.integers(4))
            img, inst = np.rot90(img, k), np.rot90(inst, k)
            if rng.random() < 0.5:
                img, inst = img[:, ::-1], inst[:, ::-1]
            img = np.clip((img - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1), 0, 1)
            img, inst = np.ascontiguousarray(img), np.ascontiguousarray(inst)
        t = instance_to_targets(inst, thickness=self.boundary_thickness, mode=self.boundary_mode)
        return {"image": img.transpose(2, 0, 1).astype(np.float32), "inst": inst.astype(np.int64),
                "sem2": t["sem2"], "sem3": t["sem3"], "dist": t["dist"], "index": i}


def _random_crop(img, inst, size, rng):
    H, W = inst.shape
    if H < size or W < size:
        ph, pw = max(0, size - H), max(0, size - W)
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        inst = np.pad(inst, ((0, ph), (0, pw)), mode="constant")
        H, W = inst.shape
    y = int(rng.integers(0, H - size + 1))
    x = int(rng.integers(0, W - size + 1))
    return img[y:y + size, x:x + size], inst[y:y + size, x:x + size]
