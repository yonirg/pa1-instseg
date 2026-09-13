"""Parte 0 — dataset sintético de elipses.

Gera imagens ``size x size`` com ``n_min..n_max`` elipses de raios variados,
muitas se tocando (``touch_prob``), com ruído gaussiano e contraste variáveis.
As máscaras de instância vêm de graça: cada elipse recebe um id inteiro.

Não há sobreposição: a primeira elipse pintada "ganha" o pixel (como no DSB2018,
onde as máscaras são disjuntas). Se uma elipse nova ficar com menos de
``min_visible`` px visíveis ela é descartada.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import ellipse as draw_ellipse

from .targets import instance_to_targets


def make_synthetic_sample(
    rng: np.random.Generator,
    size: int = 128,
    n_min: int = 5,
    n_max: int = 20,
    r_min: float = 5.0,
    r_max: float = 16.0,
    touch_prob: float = 0.6,
    min_visible: int = 20,
):
    """Retorna (image float32 [0,1] HxW, inst int32 HxW)."""
    inst = np.zeros((size, size), dtype=np.int32)
    n = int(rng.integers(n_min, n_max + 1))
    params = []  # (r, c, ra, rb, theta)
    label = 0
    tries = 0
    while len(params) < n and tries < n * 20:
        tries += 1
        ra = rng.uniform(r_min, r_max)
        rb = rng.uniform(r_min, r_max)
        theta = rng.uniform(0, np.pi)
        if params and rng.random() < touch_prob:
            # coloca encostada em uma elipse existente
            pr, pc, pa, pb, _ = params[int(rng.integers(len(params)))]
            ang = rng.uniform(0, 2 * np.pi)
            d = (pa + pb) / 2 + (ra + rb) / 2
            d *= rng.uniform(0.80, 0.98)  # ligeiramente sobreposta → encosta
            r = pr + d * np.sin(ang)
            c = pc + d * np.cos(ang)
        else:
            r = rng.uniform(r_max, size - r_max)
            c = rng.uniform(r_max, size - r_max)
        if not (0 <= r < size and 0 <= c < size):
            continue
        rr, cc = draw_ellipse(r, c, ra, rb, shape=inst.shape, rotation=theta)
        free = inst[rr, cc] == 0
        if free.sum() < min_visible:
            continue
        label += 1
        inst[rr[free], cc[free]] = label
        params.append((r, c, ra, rb, theta))

    # garante conectividade: um objeto pode ter sido "cortado" em 2 por outro
    inst = _keep_largest_component(inst)

    # --- fotometria ---
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size] / size
    bg_level = rng.uniform(0.05, 0.30)
    grad = rng.uniform(-0.15, 0.15) * xx + rng.uniform(-0.15, 0.15) * yy
    img += bg_level + grad
    for k in range(1, inst.max() + 1):
        m = inst == k
        level = rng.uniform(0.45, 0.95)
        # textura interna leve + queda de intensidade nas bordas (parece célula)
        edt = ndi.distance_transform_edt(m)
        shade = 0.75 + 0.25 * (edt / max(edt.max(), 1.0))
        img[m] = level * shade[m]
    img = ndi.gaussian_filter(img, sigma=rng.uniform(0.3, 1.2))
    noise_sigma = rng.uniform(0.02, 0.15)
    img += rng.normal(0, noise_sigma, img.shape).astype(np.float32)
    contrast = rng.uniform(0.6, 1.2)
    img = (img - img.mean()) * contrast + img.mean()
    img = np.clip(img, 0, 1).astype(np.float32)
    return img, inst


def _keep_largest_component(inst: np.ndarray) -> np.ndarray:
    out = np.zeros_like(inst)
    new = 0
    for k in range(1, inst.max() + 1):
        lab, n = ndi.label(inst == k)
        if n == 0:
            continue
        if n > 1:
            sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            m = lab == keep
        else:
            m = lab == 1
        new += 1
        out[m] = new
    return out


class SyntheticEllipses:
    """Dataset determinístico: a amostra ``i`` é sempre gerada com seed ``seed+i``."""

    def __init__(self, n: int, size: int = 128, seed: int = 0, boundary_thickness: int = 2,
                 boundary_mode: str = "touching", augment: bool = False, **gen_kwargs):
        self.n, self.size, self.seed = n, size, seed
        self.gen_kwargs = gen_kwargs
        self.boundary_thickness = boundary_thickness
        self.boundary_mode = boundary_mode
        self.augment = augment

    def __len__(self):
        return self.n

    def raw(self, i: int):
        rng = np.random.default_rng(self.seed + i)
        return make_synthetic_sample(rng, size=self.size, **self.gen_kwargs)

    def __getitem__(self, i: int):
        img, inst = self.raw(i)
        if self.augment:
            rng = np.random.default_rng(10_000_003 * (self.seed + i) + 7)
            img, inst = _flip_rot(img, inst, rng)
        t = instance_to_targets(inst, thickness=self.boundary_thickness, mode=self.boundary_mode)
        return {
            "image": img[None].astype(np.float32),  # (1,H,W)
            "inst": inst.astype(np.int64),
            "sem2": t["sem2"], "sem3": t["sem3"], "dist": t["dist"],
            "index": i,
        }


def _flip_rot(img, inst, rng):
    k = int(rng.integers(4))
    img, inst = np.rot90(img, k), np.rot90(inst, k)
    if rng.random() < 0.5:
        img, inst = img[:, ::-1], inst[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(inst)
