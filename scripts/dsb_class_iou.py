"""IoU por classe (fundo/interior/fronteira) no teste do DSB2018 para os dois modelos da Parte 2.

    python scripts/dsb_class_iou.py      # → runs/dsb_class_iou_final.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa1_instseg.ablation import class_iou  # noqa: E402
from pa1_instseg.data import get_datasets  # noqa: E402
from pa1_instseg.evaluate import load_run  # noqa: E402

out = {}
for rd in ("runs/dsb_p2_unet_boundary", "runs/dsb_p2_unet_boundary_autoalpha"):
    model, cfg = load_run(rd)
    out[rd] = class_iou(model, get_datasets(cfg)[0]["test"], "boundary", 3)
    print(rd, out[rd])
Path("runs/dsb_class_iou_final.json").write_text(json.dumps(out, indent=1))
