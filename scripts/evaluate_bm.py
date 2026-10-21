#!/usr/bin/env python
"""
Evaluate segment-then-classify (benign vs malignant) on a labelled set.

Input CSV columns:
  image    path to the image (required)
  label    benign / malignant (or 0 / 1) (required)
  mask     path to a ground-truth lesion mask (optional; enables segmentation Dice)
  patient  patient ID (optional; bootstrap CIs resample patients instead of images)
  group    e.g. source dataset or vendor (optional; per-group AUCs)

For every image the full chain runs (Stage A recorded but not used as a stop, localise, MedSAM,
crop, Stage B on the patch AND on the whole image). Writes per_image.csv (resumable) and metrics.json.

  python scripts/evaluate_bm.py --csv test.csv --out eval_out/ --device cuda:0
"""
import os
import sys
import csv
import json
import argparse
import traceback
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mammo_cascade import Cascade  # noqa: E402
from mammo_cascade.models import cap_size  # noqa: E402

FIELDS = ["image", "label", "patient", "group", "p_abnormal", "stage_a_passed", "box_source", "box_score",
          "lesion_area_px", "p_mal_patch", "p_mal_whole", "dice", "error"]


def _label(v):
    v = str(v).strip().lower()
    if v in ("1", "malignant", "cancer"):
        return 1
    if v in ("0", "benign"):
        return 0
    raise ValueError(f"unknown label '{v}'")


def _dice(pred, gt_path, shape):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    gt, _ = cap_size(gt)
    if gt.shape != shape:
        gt = cv2.resize(gt, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    g = gt > 127
    if not g.any():
        return None
    p = pred > 0 if pred is not None else np.zeros_like(g)
    return float(2 * (p & g).sum() / (p.sum() + g.sum()))


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def _boot(fn, units, n=1000, seed=0):
    """Percentile 95% CI by resampling units (patients or images). fn(index_array) -> float or None."""
    rng = np.random.default_rng(seed)
    keys = list(units)
    vals = []
    for _ in range(n):
        idx = np.concatenate([units[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        v = fn(idx)
        if v is not None:
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def _sens_spec(y, s, thr):
    y, p = np.asarray(y), np.asarray(s) >= thr
    return {"sensitivity": round(float(p[y == 1].mean()), 4), "specificity": round(float((~p[y == 0]).mean()), 4)}


def metrics(rows, thr_b):
    ok = [r for r in rows if not r["error"]]
    y = np.array([int(r["label"]) for r in ok])
    pp = np.array([float(r["p_mal_patch"]) for r in ok])
    pw = np.array([float(r["p_mal_whole"]) for r in ok])
    passed = np.array([r["stage_a_passed"] in ("True", True) for r in ok])
    units = {}
    for i, r in enumerate(ok):
        units.setdefault(r["patient"] or r["image"], []).append(i)
    units = {k: np.array(v) for k, v in units.items()}

    m = {"n_images": len(ok), "n_errors": len(rows) - len(ok),
         "n_benign": int((y == 0).sum()), "n_malignant": int((y == 1).sum()), "n_bootstrap_units": len(units),
         "stage_b_threshold": thr_b}
    if len(set(y)) < 2:
        raise SystemExit("metrics need both benign and malignant images in the CSV")
    m["auc_patch"] = round(_auc(y, pp), 4)
    m["auc_patch_ci95"] = _boot(lambda i: _auc(y[i], pp[i]), units)
    m["auc_whole"] = round(_auc(y, pw), 4)
    m["auc_whole_ci95"] = _boot(lambda i: _auc(y[i], pw[i]), units)
    m["auc_diff_patch_minus_whole"] = round(m["auc_patch"] - m["auc_whole"], 4)
    m["auc_diff_ci95"] = _boot(lambda i: (lambda a, b: None if a is None or b is None else a - b)(
        _auc(y[i], pp[i]), _auc(y[i], pw[i])), units)
    m["at_threshold_patch"] = _sens_spec(y, pp, thr_b)
    m["at_threshold_whole"] = _sens_spec(y, pw, thr_b)
    m["stage_a_pass_rate"] = {"benign": round(float(passed[y == 0].mean()), 4),
                              "malignant": round(float(passed[y == 1].mean()), 4)}
    # end to end: a cancer is caught only if Stage A passes it AND Stage B (patch) calls it malignant
    called = passed & (pp >= thr_b)
    m["end_to_end"] = {"malignant_called_malignant": round(float(called[y == 1].mean()), 4),
                       "benign_called_malignant": round(float(called[y == 0].mean()), 4)}
    src = [r["box_source"] or "none" for r in ok]
    m["box_source_counts"] = {s: src.count(s) for s in sorted(set(src))}
    d = np.array([float(r["dice"]) for r in ok if r["dice"] not in ("", None)])
    if len(d):
        m["dice"] = {"n": int(len(d)), "mean": round(float(d.mean()), 4), "median": round(float(np.median(d)), 4),
                     "frac_ge_0.5": round(float((d >= 0.5).mean()), 4)}
    groups = sorted({r["group"] for r in ok if r["group"]})
    if groups:
        m["per_group"] = {}
        for g in groups:
            idx = np.array([i for i, r in enumerate(ok) if r["group"] == g])
            m["per_group"][g] = {"n_benign": int((y[idx] == 0).sum()), "n_malignant": int((y[idx] == 1).sum()),
                                 "auc_patch": _auc(y[idx], pp[idx]), "auc_whole": _auc(y[idx], pw[idx])}
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--medsam-finetuned", default=None)
    a = ap.parse_args()

    items = list(csv.DictReader(open(a.csv)))
    os.makedirs(a.out, exist_ok=True)
    per = os.path.join(a.out, "per_image.csv")
    done = {r["image"] for r in csv.DictReader(open(per))} if os.path.exists(per) else set()
    todo = [it for it in items if it["image"] not in done]
    print(f"{len(items)} images, {len(done)} already done, {len(todo)} to run", flush=True)

    casc = Cascade(device=a.device, use_detector=not a.no_detector, medsam_finetuned=a.medsam_finetuned)
    new = not os.path.exists(per)
    with open(per, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, it in enumerate(todo, 1):
            row = {k2: "" for k2 in FIELDS}
            row.update(image=it["image"], patient=it.get("patient", ""), group=it.get("group", ""))
            try:
                row["label"] = _label(it["label"])
                res, art = casc.run(it["image"], stage_b_input="both", always_classify=True)
                loc, sb = res.get("localisation", {}), res["stage_b"]
                row.update(p_abnormal=res["stage_a"]["p_abnormal"], stage_a_passed=res["stage_a"]["passed"],
                           box_source=loc.get("source") or "", box_score=loc.get("score") or "",
                           lesion_area_px=res.get("segmentation", {}).get("area_px", ""),
                           p_mal_patch=sb["p_malignant_patch"], p_mal_whole=sb["p_malignant_whole"])
                if it.get("mask"):
                    dsc = _dice(art["mask"], it["mask"], art["gray"].shape)
                    row["dice"] = "" if dsc is None else round(dsc, 4)
            except Exception as e:
                traceback.print_exc()
                row["error"] = repr(e)
            w.writerow(row); f.flush()
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}", flush=True)

    rows = list(csv.DictReader(open(per)))
    m = metrics(rows, casc.thr_b)
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
