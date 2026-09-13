"""Geração de rótulos a partir das máscaras de instância.

Decisões de projeto (Trilha A):

* ``sem2``  — binário fundo/objeto (Parte 1).
* ``sem3``  — 0 fundo, 1 interior, 2 fronteira. A fronteira é gerada
  diretamente do mapa de instâncias:

  - ``mode="touching"`` (padrão): pixel de objeto cuja vizinhança
    ``(2*thickness+1)^2`` contém **outro** id de instância. É exatamente o
    pixel que a segmentação semântica não consegue separar; fundo não conta.
    Implementado com max/min filter sobre o mapa de ids (O(N), sem loop).
  - ``mode="all"``: fronteira interna de cada instância (contra fundo e contra
    vizinhos), dilatada até ``thickness``. Mais pixels positivos, mais fácil de
    aprender, mas "come" a borda de todo objeto.

* ``dist`` — distância euclidiana ao fundo **por instância**, normalizada pelo
  máximo da instância (∈ [0,1]). Como é por instância, dois núcleos encostados
  têm dois picos separados por um vale — é a elevação (negada) do watershed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


def touching_boundary(inst: np.ndarray, thickness: int = 2) -> np.ndarray:
    fg = inst > 0
    k = 2 * thickness + 1
    big = np.iinfo(np.int32).max
    mx = ndi.maximum_filter(inst, size=k)
    mn = ndi.minimum_filter(np.where(fg, inst, big).astype(np.int32), size=k)
    return fg & (mx != mn)


def all_boundary(inst: np.ndarray, thickness: int = 2) -> np.ndarray:
    b = find_boundaries(inst, mode="inner")
    if thickness > 1:
        b = ndi.binary_dilation(b, iterations=thickness - 1) & (inst > 0)
    return b


def instance_distance(inst: np.ndarray) -> np.ndarray:
    dist = np.zeros(inst.shape, dtype=np.float32)
    for k, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        sub = inst[sl] == k
        # pad para a EDT ver o fundo na borda do recorte
        e = ndi.distance_transform_edt(np.pad(sub, 1))[1:-1, 1:-1]
        m = e.max()
        if m > 0:
            dist[sl][sub] = (e[sub] / m).astype(np.float32)
    return dist


def instance_to_targets(inst: np.ndarray, thickness: int = 2, mode: str = "touching") -> dict:
    fg = inst > 0
    if mode == "touching":
        b = touching_boundary(inst, thickness)
    elif mode == "all":
        b = all_boundary(inst, thickness)
    else:
        raise ValueError(mode)
    sem3 = np.zeros(inst.shape, dtype=np.int64)
    sem3[fg] = 1
    sem3[b] = 2
    return {
        "sem2": fg.astype(np.int64),
        "sem3": sem3,
        "dist": instance_distance(inst),
    }


def relabel_sequential(inst: np.ndarray) -> np.ndarray:
    ids = np.unique(inst)
    ids = ids[ids > 0]
    out = np.zeros_like(inst)
    for new, old in enumerate(ids, start=1):
        out[inst == old] = new
    return out
