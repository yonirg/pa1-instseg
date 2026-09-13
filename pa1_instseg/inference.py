"""Inferência de uma imagem (qualquer tamanho) → mapas ativados → instâncias."""
from __future__ import annotations

import numpy as np
import torch

from .postproc import naive_instances, watershed_instances


@torch.no_grad()
def predict_maps(model: torch.nn.Module, image: np.ndarray, head: str, device="cpu",
                 multiple: int = 16) -> np.ndarray:
    """``image``: (C,H,W) float32 em [0,1]. Retorna mapas ativados (C_out,H,W):
    binary → [p_bg, p_fg]; boundary → [p_bg, p_int, p_bnd, dist]."""
    model.eval()
    C, H, W = image.shape
    ph, pw = (-H) % multiple, (-W) % multiple
    x = np.pad(image, ((0, 0), (0, ph), (0, pw)), mode="reflect") if (ph or pw) else image
    x = torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
    out = model(x)[0]
    n_cls = 2 if head == "binary" else 3
    probs = torch.softmax(out[:n_cls], 0)
    maps = [probs]
    if out.shape[0] > n_cls:
        maps.append(torch.sigmoid(out[n_cls:n_cls + 1]))
    maps = torch.cat(maps, 0).cpu().numpy()
    return maps[:, :H, :W]


def decode_maps(maps: np.ndarray, head: str, naive: bool = False, **kw) -> np.ndarray:
    if head == "binary":
        return naive_instances(maps[1], **kw)
    if naive:  # mesma rede da Parte 2, pós-processamento da Parte 1
        return naive_instances(maps[1] + maps[2], **kw)
    dist = maps[3] if maps.shape[0] > 3 else None
    return watershed_instances(maps[:3], dist, **kw)


def foreground_of(maps: np.ndarray, head: str) -> np.ndarray:
    return maps[1] > 0.5 if head == "binary" else (maps[1] + maps[2]) > 0.5


def make_mosaic(dataset, idxs, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cola ``grid[0] x grid[1]`` amostras numa imagem grande; ids re-rotulados."""
    rows, cols = grid
    assert len(idxs) >= rows * cols
    imgs, insts = [], []
    offset = 0
    k = 0
    for r in range(rows):
        row_i, row_m = [], []
        for c in range(cols):
            s = dataset[idxs[k]]; k += 1
            inst = s["inst"].copy()
            inst[inst > 0] += offset
            offset = int(inst.max())
            row_i.append(s["image"]); row_m.append(inst)
        imgs.append(np.concatenate(row_i, axis=2)); insts.append(np.concatenate(row_m, axis=1))
    return np.concatenate(imgs, axis=1), np.concatenate(insts, axis=0)
