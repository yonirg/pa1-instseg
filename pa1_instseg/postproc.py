"""Decodificação: da previsão da rede para um mapa de instâncias.

* ``naive_instances``     — Parte 1: limiar + componentes conexos (8-conexo).
* ``watershed_instances`` — Parte 2, Trilha A: os *interiores* previstos viram
  marcadores; a máscara é o foreground (interior ∪ fronteira); a elevação é
  −distância (se a rede prevê distância) ou a prob. de fronteira. Regiões de
  foreground que ficaram sem marcador (objeto sem interior visível) viram
  instâncias próprias, para não perder objetos pequenos.

Os dois retornam ``int32`` com ids 1..N sequenciais.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .data.targets import relabel_sequential

STRUCT8 = np.ones((3, 3), dtype=bool)


def _remove_small(inst: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return inst
    areas = np.bincount(inst.ravel())
    small = np.where(areas < min_area)[0]
    small = small[small > 0]
    if len(small):
        inst[np.isin(inst, small)] = 0
    return inst


def naive_instances(fg_prob: np.ndarray, thr: float = 0.5, min_area: int = 4) -> np.ndarray:
    fg = fg_prob > thr
    inst, _ = ndi.label(fg, structure=STRUCT8)
    inst = _remove_small(inst.astype(np.int32), min_area)
    return relabel_sequential(inst)


def watershed_instances(probs: np.ndarray, dist: np.ndarray | None = None,
                        thr_fg: float = 0.5, thr_int: float = 0.5,
                        min_marker: int = 3, min_area: int = 4, marker_opening: int = 0,
                        marker_source: str = "interior", thr_dist: float = 0.5) -> np.ndarray:
    """``probs``: (3,H,W) softmax [fundo, interior, fronteira].

    ``marker_source``: ``"interior"`` (classe interior > thr_int) ou ``"dist"``
    (picos da distância prevista: dist > thr_dist). Como a distância é
    normalizada por instância, todo objeto tem um pico, mesmo os pequenos cuja
    classe interior desaparece sob a fronteira (Parte 5, correção que funcionou).

    ``marker_opening`` (Parte 5, correção): abertura morfológica dos marcadores
    com ``k`` iterações — corta pontes de até ``2k`` px entre interiores de
    objetos encostados. O watershed recresce o marcador até a fronteira, então
    o objeto não perde área; objetos pequenos que sumirem viram órfãos."""
    fg = (probs[1] + probs[2]) > thr_fg
    if marker_source == "dist":
        assert dist is not None, "marker_source='dist' exige a cabeça de distância"
        interior = dist > thr_dist
    else:
        interior = probs[1] > thr_int
    if marker_opening > 0:
        interior = ndi.binary_opening(interior, structure=STRUCT8, iterations=marker_opening)
    markers, _ = ndi.label(interior & fg, structure=STRUCT8)
    markers = _remove_small(markers.astype(np.int32), min_marker)
    elevation = -dist if dist is not None else probs[2]
    inst = watershed(elevation, markers, mask=fg, connectivity=2).astype(np.int32)
    # foreground órfão (sem marcador) → instâncias novas
    orphan = fg & (inst == 0)
    if orphan.any():
        extra, n = ndi.label(orphan, structure=STRUCT8)
        extra = _remove_small(extra.astype(np.int32), min_area)
        inst[extra > 0] = extra[extra > 0] + inst.max()
    inst = _remove_small(inst, min_area)
    return relabel_sequential(inst)


def decode(out: np.ndarray, head: str, **kw) -> np.ndarray:
    """``out``: logits (C,H,W) numpy. Escolhe o decodificador pelo tipo de cabeça."""
    if head == "binary":
        probs = _softmax(out[:2])
        return naive_instances(probs[1], **kw)
    probs = _softmax(out[:3])
    dist = _sigmoid(out[3]) if out.shape[0] > 3 else None
    return watershed_instances(probs, dist, **kw)


def naive_from_boundary_head(out: np.ndarray, **kw) -> np.ndarray:
    """Mesma rede da Parte 2 decodificada ingenuamente (fg = interior ∪ fronteira)
    — isola o efeito do pós-processamento do efeito da representação."""
    probs = _softmax(out[:3])
    return naive_instances(probs[1] + probs[2], **kw)


def _softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
