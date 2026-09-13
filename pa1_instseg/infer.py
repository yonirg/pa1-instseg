"""Inferência em uma imagem qualquer (usado por ``inferencia.ipynb``).

    python -m pa1_instseg.infer --run runs/p2_unet_boundary --image foto.png [--out saida.png]

Aceita PNG/JPG/TIF em cinza ou RGB e qualquer tamanho: converte para o nº de
canais do checkpoint, normaliza para [0,1], faz padding até múltiplo de 16 e,
se a imagem for maior que ``--tile``, roda em tiles com costura de mapas
(correção B da Parte 4). Devolve o mapa de instâncias, a contagem e a figura.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluate import load_run
from .inference import decode_maps, predict_maps
from .tiling import maps_tiled
from .viz import color_instances


def load_any_image(path: str | Path, in_ch: int) -> np.ndarray:
    im = Image.open(path)
    im = im.convert("L") if in_ch == 1 else im.convert("RGB")
    x = np.asarray(im).astype(np.float32)
    if x.max() > 1.0:
        x = x / (65535.0 if x.max() > 255 else 255.0)
    x = x[None] if x.ndim == 2 else x.transpose(2, 0, 1)
    return np.ascontiguousarray(x)


def infer_image(run: str | Path, image_path: str | Path, tile: int = 256, overlap: int = 64,
                decode_kw: dict | None = None):
    model, cfg = load_run(run)
    img = load_any_image(image_path, cfg["in_ch"])
    predict = lambda im: predict_maps(model, im, cfg["head"])
    _, H, W = img.shape
    maps = predict(img) if max(H, W) <= tile else maps_tiled(predict, img, tile, overlap)
    inst = decode_maps(maps, cfg["head"], **(decode_kw or {}))
    return {"instances": inst, "count": int(inst.max()), "maps": maps, "image": img,
            "colored": (color_instances(inst) * 255).astype(np.uint8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p2_unet_boundary")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--marker-source", default="interior", choices=["interior", "dist"])
    args = ap.parse_args()
    r = infer_image(args.run, args.image, decode_kw={"marker_source": args.marker_source})
    out = args.out or str(Path(args.image).with_suffix("")) + "_instancias.png"
    Image.fromarray(r["colored"]).save(out)
    print(f"{r['count']} instâncias → {out}")


if __name__ == "__main__":
    main()
