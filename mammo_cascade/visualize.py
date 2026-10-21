"""Overlay and patch images for a cascade result."""
import cv2
import numpy as np

_BOX_COLOR = {"detector": (0, 200, 255), "gradcam": (255, 0, 255)}   # BGR: orange / magenta


def draw_overlay(result, art):
    """Mammogram with the lesion box, the tumour outline (red) and the scores. Returns a BGR image."""
    g = art["gray"]
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    th = max(2, int(0.003 * max(g.shape)))
    if art.get("mask") is not None and art["mask"].any():
        red = np.zeros_like(vis); red[art["mask"] > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 1.0, red, 0.35, 0)
        cnts, _ = cv2.findContours(art["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), th)
    if art.get("box") is not None:
        src = result.get("localisation", {}).get("source")
        x0, y0, x1, y1 = art["box"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), _BOX_COLOR.get(src, (0, 255, 0)), th)

    a = result["stage_a"]
    lines = [f"Stage A  P(abnormal)={a['p_abnormal']:.3f}  (thr {a['threshold']})"]
    if "stage_b" in result:
        b = result["stage_b"]
        src = b.get("decision_from", "patch" if "p_malignant_patch" in b else "whole")
        lines.append(f"Stage B  P(malignant, {src})={b[f'p_malignant_{src}']:.3f}  (thr {b['threshold']})")
    lines.append(f"DECISION: {result['decision'].upper()}   [research use only]")
    fs = max(0.8, g.shape[1] / 1400)
    y = int(40 * fs)
    for ln in lines:
        cv2.putText(vis, ln, (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), int(5 * fs), cv2.LINE_AA)
        cv2.putText(vis, ln, (15, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), int(2 * fs), cv2.LINE_AA)
        y += int(40 * fs)
    return vis


def save_outputs(result, art, stem):
    """Write <stem>_overlay.png (and <stem>_patch.png, <stem>_mask.png when a lesion was processed)."""
    cv2.imwrite(f"{stem}_overlay.png", draw_overlay(result, art))
    g = art["gray"]
    if art.get("mask") is not None and art["mask"].any():
        cv2.imwrite(f"{stem}_mask.png", art["mask"] * 255)
    if art.get("crop_box") is not None:
        x0, y0, x1, y1 = art["crop_box"]
        cv2.imwrite(f"{stem}_patch.png", g[y0:y1, x0:x1])
