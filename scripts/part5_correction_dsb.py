"""Parte 5 (DSB2018) — resume o antes/depois da correção escolhida a partir da galeria.

    python scripts/part5_failures.py --run runs/dsb_p2_unet_boundary_autoalpha --fixed runs/dsb_p2_unet_boundary
    python scripts/part5_correction_dsb.py

Diagnóstico (galeria de ``dsb_p2_unet_boundary_autoalpha``): p(fronteira) alta no núcleo
inteiro e o interior quase some → sem marcadores, o watershed devolve fragmentos e
fantasmas. Causa: o α automático (frequência inversa normalizada para média 1) nos recortes do DSB,
onde a fronteira é só 0,4% dos pixels, dá peso 0,017 ao fundo, 0,11 ao interior e 2,87 à fronteira.
Mudança sugerida: α fixo fundo/interior/fronteira = 1/1/3 (mesma rede, mesmo orçamento).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BEFORE, AFTER = Path("runs/dsb_p2_unet_boundary_autoalpha"), Path("runs/dsb_p2_unet_boundary")


def alpha_auto_dsb() -> list[float]:
    """Pesos que o α automático produz com as frequências de classe do treino do DSB."""
    import sys
    sys.path.insert(0, ".")
    from pa1_instseg.data import get_datasets
    from pa1_instseg.train import DEFAULTS
    ds = get_datasets({**DEFAULTS, "dataset": "dsb", "data_root": "data"})[0]["train"]
    cnt = np.zeros(3)
    for i in range(len(ds)):
        cnt += np.bincount(ds[i]["sem3"].ravel(), minlength=3)
    freq = cnt / cnt.sum()
    inv = 1.0 / (freq + 1e-3)
    inv = inv / inv.sum() * 3
    return freq.round(4).tolist(), inv.round(3).tolist()


def main():
    ba = json.loads((BEFORE / "part5_before_after.json").read_text())
    freq, alpha = alpha_auto_dsb()
    out = {
        "diagnostico": "α automático zera o peso do fundo; a rede chama o núcleo inteiro de fronteira",
        "frequencia_classes_treino": freq, "alpha_automatico": alpha,
        "mudanca": f"frequência das classes no treino (fundo/interior/fronteira) = {freq}; α automático = {alpha} "
                   "→ α fixo = [1, 1, 3], mesma U-Net, mesmo orçamento (`--alpha 1,1,3`).",
        "antes": ba["antes"], "depois": ba["depois"],
    }
    a, d = ba["antes"], ba["depois"]
    worked = d["map"] > a["map"]
    out["conclusao"] = (
        f"mAP {a['map']:.3f} → {d['map']:.3f}; fantasmas {a['fantasma']} → {d['fantasma']}, fragmentos {a['fragmento']} → {d['fragmento']}, "
        f"fusões {a['fusao']} → {d['fusao']}. "
        + ("A correção funcionou: o diagnóstico (peso do fundo) estava certo. "
           if worked else "A correção NÃO melhorou o mAP: o peso do fundo não era a causa principal. ")
        + "Figuras: `runs/dsb_p2_unet_boundary_autoalpha/part5_failure1..5.png` (antes) e `part5_failure1_after.png`, `part5_failure2_after.png` (depois)."
    )
    (AFTER / "part5_correction.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
