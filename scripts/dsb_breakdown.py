"""DSB2018 — mAP por faixa de densidade (Parte 1, item 5) e por modalidade, lado a lado.

    python scripts/dsb_breakdown.py        # lê os eval_test*.json e escreve runs/dsb_breakdown.{json,md,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa1_instseg.data.dsb2018 import index_dataset, stratified_split  # noqa: E402
from pa1_instseg.viz import curve  # noqa: E402

RUNS = {
    "Parte 1: U-Net binária + limiar/CC": "runs/dsb_p1_unet_binary/eval_test.json",
    "Parte 2: mesma rede, decodificação ingênua (CC)": "runs/dsb_p2_unet_boundary/eval_test_naive.json",
    "Parte 2: fronteira+distância + watershed": "runs/dsb_p2_unet_boundary/eval_test.json",
    "Parte 2: watershed com marcadores da distância": "runs/dsb_p2_unet_boundary/eval_test_distmarkers.json",
    "Parte 2 com α automático (antes da Parte 5)": "runs/dsb_p2_unet_boundary_autoalpha/eval_test.json",
}
BINS = [0, 10, 25, 50, 100, 10_000]
MODS = ("fluorescence", "histology", "brightfield")


def main():
    test = stratified_split(index_dataset("data"))["test"]
    labels = [f"{BINS[i]}–{BINS[i + 1] - 1}" if BINS[i + 1] < 10_000 else f"≥{BINS[i]}" for i in range(len(BINS) - 1)]
    out, lines = {}, []
    head = "| modelo | mAP | AP50 | erro contagem | " + " | ".join(f"mAP {l} núcleos" for l in labels) + " | " + " | ".join(MODS) + " |"
    lines += [head, "|" + "---|" * (4 + len(labels) + len(MODS))]
    for name, p in RUNS.items():
        if not Path(p).exists():
            continue
        d = json.loads(Path(p).read_text()); per = d["per_image"]; a = d["aggregate"]
        ng = np.array([r["n_gt"] for r in per]); mp = np.array([r["map"] for r in per])
        mod = np.array([test[r["index"]]["modality"] for r in per])
        dens = [float(mp[(ng >= BINS[i]) & (ng < BINS[i + 1])].mean()) for i in range(len(BINS) - 1)]
        mods = [float(mp[mod == m].mean()) for m in MODS]
        out[name] = {"map": a["map"], "ap50": a["ap50"], "count_err": a["count_err"], "por_densidade": dict(zip(labels, dens)),
                     "por_modalidade": dict(zip(MODS, mods))}
        lines.append(f"| {name} | {a['map']:.3f} | {a['ap50']:.3f} | {a['count_err']:.1f} | " + " | ".join(f"{v:.2f}" for v in dens)
                     + " | " + " | ".join(f"{v:.2f}" for v in mods) + " |")
    n_bin = [int(((ng >= BINS[i]) & (ng < BINS[i + 1])).sum()) for i in range(len(BINS) - 1)]
    n_mod = [int((mod == m).sum()) for m in MODS]
    lines.append(f"\nImagens por faixa: {dict(zip(labels, n_bin))}; por modalidade: {dict(zip(MODS, n_mod))}.")
    Path("runs/dsb_breakdown.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    Path("runs/dsb_breakdown.md").write_text("\n".join(lines) + "\n")
    keep = [k for k in out if "α automático" not in k and "marcadores" not in k]
    curve(list(range(len(labels))), {k.split(": ", 1)[1]: list(out[k]["por_densidade"].values()) for k in keep},
          "runs/dsb_breakdown_density.png", xlabel="nº de núcleos na imagem (faixas: " + ", ".join(labels) + ")",
          title="DSB2018 (teste): mAP × densidade")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
