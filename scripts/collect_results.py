"""Gera RESULTS.md a partir dos JSONs em runs/ — toda tabela da apresentação sai daqui.

    python scripts/collect_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("runs")


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(a, keys=("map", "ap50", "ap75", "count_err", "iou", "dice")):
    return " | ".join(f"{a[k]:.3f}" for k in keys)


def main():
    out = ["# RESULTS.md — números reproduzíveis (1 núcleo de CPU)\n",
           "Todos gerados pelos comandos do README; JSONs de origem indicados em cada seção.\n"]

    # Parte 0 / 1 / 2
    out += ["## Partes 0, 1 e 2 — baseline vs. cabeça de instâncias (teste sintético, 128 imagens)\n",
            "| modelo | decodificação | matching | mAP | AP50 | AP75 | erro contagem | IoU fg | Dice |",
            "|---|---|---|---|---|---|---|---|---|"]
    rows = [("U-Net binário (Parte 1)", "limiar + CC", "greedy", "p1_unet_binary/eval_test.json"),
            ("U-Net binário (Parte 1)", "limiar + CC", "hungarian", "p1_unet_binary/eval_test_hungarian.json"),
            ("U-Net fronteira+dist (Parte 2)", "limiar + CC (ingênuo)", "greedy", "p2_unet_boundary/eval_test_naive.json"),
            ("U-Net fronteira+dist (Parte 2)", "watershed (marcador = interior)", "greedy", "p2_unet_boundary/eval_test.json"),
            ("U-Net fronteira+dist (Parte 2)", "watershed (marcador = interior)", "hungarian", "p2_unet_boundary/eval_test_hungarian.json"),
            ("U-Net fronteira+dist (Parte 2)", "watershed (marcador = distância, Parte 5)", "greedy", "p2_unet_boundary/eval_test_distmarkers.json")]
    for name, dec, rule, p in rows:
        d = j(R / p)
        if d:
            out.append(f"| {name} | {dec} | {rule} | {fmt(d['aggregate'])} |")
    for run in ("p2_unet_boundary", "p1_unet_binary"):
        s = j(R / run / "summary.json"); h = j(R / run / "history.json")
        if s:
            out.append(f"\n`{run}`: {s['epochs_run']} épocas em {s['train_seconds']} s de CPU, "
                       f"{s['n_params']/1e3:.0f}k parâmetros, melhor val mAP {s['best_val_map']:.3f} "
                       f"(Parte 0: treina em < 5 min).")
    out.append("\nFigura mAP × densidade: `runs/p1_unet_binary/eval_test_density.png` e `runs/p2_unet_boundary/eval_test_density.png`.\n")

    # Parte 3
    for ax in (1, 2):
        t = R / "ablation" / f"axis{ax}_table.md"
        if t.exists():
            title = {1: "Eixo 1 — como recuperar resolução (mesmo encoder base=8, depth=3)",
                     2: "Eixo 2 — função de perda (U-Net)"}[ax]
            out += [f"## Parte 3 — {title}\n",
                    "Configuração: `--epochs 12 --time-limit 100 --base 8 --n-train 300 --lr 2e-3 --alpha-max 3`, 2 seeds, média ± desvio, teste em 96 imagens.\n",
                    t.read_text(), f"Figura: `runs/ablation/axis{ax}_map.png`" + (" e `axis2_boundary_iou.png`" if ax == 2 else "") + "\n"]

    # Parte 4
    d = j(R / "p2_unet_boundary" / "part4_mosaic.json")
    if d:
        a = d["args"]
        out += [f"## Parte 4 — inferência em mosaico (cena {a['grid']}×{a['grid']} tiles de {a['tile']} px, sobreposição {a['overlap']}, {a['n_mosaics']} cenas)\n",
                "| estratégia | mAP | AP50 | erro contagem | objetos na borda quebrados |", "|---|---|---|---|---|"]
        for k, v in d["summary"].items():
            out.append(f"| {k} | {v['map']:.3f} | {v['ap50']:.3f} | {v['count_err']:.1f} | {v['objetos_na_borda_quebrados']} |")
        out.append("\nFiguras: `part4_border_object.png`, `part4_mosaic_full.png`.\n")

    # Parte 5
    rf = j(R / "p2_unet_boundary" / "part5_receptive_field.json")
    f5 = j(R / "p2_unet_boundary" / "part5_failures.json")
    ba = j(R / "p2_unet_boundary" / "part5_before_after.json")
    if rf and f5:
        out += ["## Parte 5 — galeria de falhas, campo receptivo e correção\n",
                f"Campo receptivo teórico do encoder ({rf['arch']}): **{rf['rf']} px** (jump {rf['jump']}). "
                f"Objetos: mediana {rf['diam_median']:.1f} px, p90 {rf['diam_p90']:.1f}, máx {rf['diam_max']:.1f} → "
                f"{100*rf['frac_objetos_maiores_que_rf']:.0f}% maiores que o RF. "
                f"DeepLab OS8: **{rf['deeplab_os8_com_atrous']} px com atrous vs {rf['deeplab_os8_sem_atrous']} px sem**, mesma resolução de saída. "
                "Figura: `part5_rf_vs_objects.png`.\n",
                f"Taxonomia no teste inteiro: {f5['taxonomia_teste']}. Galeria: `part5_failure1..5.png`.\n",
                "| falha | idx | mAP | fusões | fragmentos | perdidos | fantasmas | contraste fg/bg | ruído | p(fronteira) no contato GT |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for k, g in enumerate(f5["galeria"], 1):
            out.append(f"| {k} | {g['index']} | {g['map']:.2f} | {g['fusao']} | {g['fragmento']} | {g['perdido']} | {g['fantasma']} | "
                       f"{g['contraste_fg_bg']:.2f} | {g['ruido_bg']:.2f} | {g['p_fronteira_media_na_fronteira_gt']} |")
        if ba:
            out += ["\n**Correção 1 (fronteira 3 px no rótulo, retreino mesmo orçamento):**",
                    f"antes mAP {ba['antes']['map']:.3f} / fusões {ba['antes']['fusao']} → depois mAP {ba['depois']['map']:.3f} / fusões {ba['depois']['fusao']} — **não funcionou**; "
                    "o diagnóstico 'fronteira fina' estava errado (`part5_before_after.json`)."]
        sw = j(R / "p2_unet_boundary" / "part5_postproc_sweep.json")
        if sw:
            b = sw["best"]
            out.append(f"\n**Correção 2 (limiar do interior / tamanho do marcador):** melhor {b} — fusões não caem; subir o limiar perde marcadores e cria órfãos fundidos (`part5_postproc_sweep.png`).")
        dm = j(R / "p2_unet_boundary" / "part5_dist_markers.json")
        if dm:
            best = max(dm["rows"], key=lambda r: r["map"])
            out.append(f"\n**Correção 3 (marcadores pelos picos da distância):** objetos sem interior no rótulo = {dm['sem_interior']}/{dm['objetos_gt']}; "
                       f"melhor {best} — fantasmas caem, fusões (~130) persistem em todas as variantes → contatos sem evidência fotométrica.\n")

    # Parte 6
    d = j(R / "p2_unet_boundary" / "part6_stress.json")
    if d:
        out += ["## Parte 6 — teste de estresse (96 imagens de teste)\n",
                "| modelo | limpo | blur σ=1/2/3 | ruído σ=.05/.10/.20 | contraste ×.7/.5/.35 (+brilho) | escala 0.5× / 2× |", "|---|---|---|---|---|---|"]
        for n, r in d.items():
            c = r["corrupcoes"]
            out.append(f"| {n} | {r['limpo']:.3f} | " + " / ".join(f"{v:.2f}" for v in c["blur"]) + " | "
                       + " / ".join(f"{v:.2f}" for v in c["ruido"]) + " | " + " / ".join(f"{v:.2f}" for v in c["brilho_contraste"])
                       + f" | {r['escala']['0.5']:.2f} / {r['escala']['2.0']:.2f} |")
        out.append("\nFiguras: `part6_blur.png`, `part6_ruido.png`, `part6_brilho_contraste.png`, `part6_escala.png`.\n")

    Path("RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
