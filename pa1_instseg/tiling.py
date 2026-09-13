"""Parte 4 — inferência em mosaico.

Três estratégias sobre a mesma rede:

1. ``maps_tiled``            — a prática do slide 83 para *semântica*: tiles com
   sobreposição, só o miolo (core) de cada tile conta, média onde há mais de um.
2. ``instances_tiled_naive`` — decodifica cada tile e cola só o miolo. Um objeto
   que cruza a borda entre dois miolos vira **dois ids** (ou perde um pedaço se o
   watershed de um tile não o reconheceu com o contexto cortado). É a falha.
3. Correções:
   a) ``instances_tiled_fused`` — decodifica o tile inteiro, e na **faixa de
      sobreposição** casa instâncias de tiles vizinhos por IoU (union-find);
      fragmentos casados recebem o mesmo id.
   b) ``instances_from_fused_maps`` — o que a representação da Trilha A permite:
      costurar os mapas (probabilidades + distância) como no item 1 e rodar o
      watershed **uma vez** sobre a imagem inteira.
"""
from __future__ import annotations

import numpy as np

from .metrics import iou_matrix


def tile_positions(H: int, W: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    stride = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    return [(y, x) for y in ys for x in xs]


def core_slices(y0, x0, H, W, tile, overlap):
    m = overlap // 2
    y1, y2 = (0 if y0 == 0 else y0 + m), (H if y0 + tile >= H else y0 + tile - m)
    x1, x2 = (0 if x0 == 0 else x0 + m), (W if x0 + tile >= W else x0 + tile - m)
    return slice(y1, y2), slice(x1, x2)


def maps_tiled(predict_fn, image: np.ndarray, tile: int, overlap: int) -> np.ndarray:
    _, H, W = image.shape
    acc, wsum = None, np.zeros((H, W), dtype=np.float32)
    for y0, x0 in tile_positions(H, W, tile, overlap):
        m = predict_fn(image[:, y0:y0 + tile, x0:x0 + tile])
        if acc is None:
            acc = np.zeros((m.shape[0], H, W), dtype=np.float32)
        ys, xs = core_slices(y0, x0, H, W, tile, overlap)
        w = np.zeros((tile, tile), dtype=np.float32)
        w[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = 1.0
        acc[:, y0:y0 + tile, x0:x0 + tile] += m * w
        wsum[y0:y0 + tile, x0:x0 + tile] += w
    return acc / np.maximum(wsum, 1e-6)


def instances_tiled_naive(predict_fn, decode_fn, image, tile, overlap) -> np.ndarray:
    _, H, W = image.shape
    canvas = np.zeros((H, W), dtype=np.int32)
    offset = 0
    for y0, x0 in tile_positions(H, W, tile, overlap):
        inst = decode_fn(predict_fn(image[:, y0:y0 + tile, x0:x0 + tile]))
        ys, xs = core_slices(y0, x0, H, W, tile, overlap)
        core = inst[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        core = np.where(core > 0, core + offset, 0)
        canvas[ys, xs] = core
        offset = max(offset, int(core.max()))
    return _relabel(canvas)


class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def instances_tiled_fused(predict_fn, decode_fn, image, tile, overlap, iou_thr: float = 0.25):
    _, H, W = image.shape
    pos = tile_positions(H, W, tile, overlap)
    tiles = []  # (y0, x0, inst)
    for y0, x0 in pos:
        tiles.append((y0, x0, decode_fn(predict_fn(image[:, y0:y0 + tile, x0:x0 + tile]))))
    uf = _UnionFind()
    # casa instâncias de cada par de tiles que se sobrepõem, usando só a faixa comum
    for a in range(len(tiles)):
        ya, xa, ia = tiles[a]
        for b in range(a + 1, len(tiles)):
            yb, xb, ib = tiles[b]
            oy1, oy2 = max(ya, yb), min(ya, yb) + tile
            ox1, ox2 = max(xa, xb), min(xa, xb) + tile
            if oy2 <= oy1 or ox2 <= ox1:
                continue
            ra = ia[oy1 - ya:oy2 - ya, ox1 - xa:ox2 - xa]
            rb = ib[oy1 - yb:oy2 - yb, ox1 - xb:ox2 - xb]
            iou = iou_matrix(ra, rb)
            for i, j in zip(*np.nonzero(iou >= iou_thr)):
                uf.union((a, int(i) + 1), (b, int(j) + 1))
    # cola só o miolo, mas com ids unificados
    canvas = np.zeros((H, W), dtype=np.int32)
    root_id = {}
    for t, (y0, x0, inst) in enumerate(tiles):
        ys, xs = core_slices(y0, x0, H, W, tile, overlap)
        core = inst[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        out = np.zeros_like(core)
        for lid in np.unique(core):
            if lid == 0:
                continue
            r = uf.find((t, int(lid)))
            root_id.setdefault(r, len(root_id) + 1)
            out[core == lid] = root_id[r]
        canvas[ys, xs] = out
    return _relabel(canvas)


def instances_from_fused_maps(predict_fn, decode_fn, image, tile, overlap) -> np.ndarray:
    return decode_fn(maps_tiled(predict_fn, image, tile, overlap))


def _relabel(inst):
    from .data.targets import relabel_sequential
    return relabel_sequential(inst)
