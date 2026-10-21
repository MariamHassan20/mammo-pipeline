#!/usr/bin/env python
"""
Run the mammography cascade on one image or a folder of images.

  python run_cascade.py --input view.png --out outputs/
  python run_cascade.py --input folder_of_pngs/ --out outputs/ --device cuda:0
  python run_cascade.py --input folder/ --labels labels.csv --out outputs/   # + metrics vs ground truth

Output folder:
  segmented_tumors/   <name>_patch.png (tumour crop) and <name>_mask.png (tumour mask)
  overlays/           <name>_overlay.png (box, tumour outline, scores, decision)
  json/               <name>.json (everything for one image)
  diagnosis.xlsx      one row per image (decision, scores, files; true label if --labels)
  metrics.xlsx        only with --labels: accuracy, AUC, F1, sensitivity, specificity, confusion table,
                      and segmentation Dice/IoU when the labels file has true masks
  examples/           only with true masks: 4 cases comparing our segmentation with the true tumour

labels.csv columns: image, label (normal/benign/malignant, or 0/1 for benign/malignant), optional mask
(path to the true tumour mask). With --labels, Stage B is also scored for images Stage A calls normal (so an
AUC can be computed); the decision still follows Stage A.

Research use only. Not a medical device; not for clinical decisions.
"""
import os
import sys
import csv
import glob
import argparse
import traceback
from collections import Counter

from mammo_cascade import Cascade
from mammo_cascade import config as C
from mammo_cascade import report as R

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images(path, out_dir):
    if os.path.isfile(path):
        return [path]
    out_abs = os.path.abspath(out_dir) + os.sep
    files = [f for f in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))
             if f.lower().endswith(EXTS) and not f.lower().endswith(("_overlay.png", "_patch.png", "_mask.png"))
             and not os.path.abspath(f).startswith(out_abs)]          # never read our own outputs
    if not files:
        sys.exit(f"no images found under {path}")
    return files


def unique_names(images):
    seen, out = Counter(), []
    for img in images:
        base = os.path.splitext(os.path.basename(img))[0]
        seen[base] += 1
        out.append(base if seen[base] == 1 else f"{base}__{seen[base]}")
    return out


def pick_examples(rows, k=4):
    """k cases with a ground-truth mask spread over the Dice range (best, upper-middle, lower-middle, worst)."""
    cand = sorted([r for r in rows if r.get("dice") is not None], key=lambda r: r["dice"], reverse=True)
    if len(cand) <= k:
        return cand
    idx = sorted({round(q * (len(cand) - 1)) for q in (0.05, 0.35, 0.65, 0.95)})
    return [cand[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or folder (searched recursively)")
    ap.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    ap.add_argument("--labels", default=None, help="optional CSV with ground truth: image,label[,mask]")
    ap.add_argument("--anonymize", action="store_true",
                    help="name outputs case_0001, case_0002, ... instead of the image file names; the table "
                         "linking case IDs to images is written NEXT TO the output folder (<out>_case_map.csv)")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu ... (default: cuda if available)")
    ap.add_argument("--no-detector", action="store_true", help="skip the detector; localise with GradCAM only")
    ap.add_argument("--stage-b-input", choices=["patch", "whole", "both"], default="patch",
                    help="what Stage B classifies (default: patch). 'both' reports both scores; decision uses the patch")
    ap.add_argument("--pixel-spacing", type=float, default=None,
                    help="mm per pixel of the original image (DICOM PixelSpacing) to report lesion size in mm")
    ap.add_argument("--stage-a-threshold", type=float, default=C.STAGE_A_THRESHOLD,
                    help=f"P(abnormal) threshold (default {C.STAGE_A_THRESHOLD})")
    ap.add_argument("--stage-b-threshold", type=float, default=None,
                    help="P(malignant) threshold (default: value stored in the Stage B checkpoint)")
    ap.add_argument("--medsam-finetuned", default=C.MEDSAM_FINETUNED,
                    help="optional SamModel state_dict with a fine-tuned decoder (default: original MedSAM)")
    a = ap.parse_args()

    images = list_images(a.input, a.out)
    if a.anonymize:
        names = [f"case_{i:04d}" for i in range(1, len(images) + 1)]
        map_path = os.path.abspath(a.out.rstrip("/\\")) + "_case_map.csv"
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        with open(map_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["case_id", "image"]); w.writerows(zip(names, images))
        print(f"case map (keep private): {map_path}")
    else:
        names = unique_names(images)
    labels = R.load_labels(a.labels) if a.labels else None
    sb_input = "both" if labels else a.stage_b_input
    R.make_dirs(a.out)
    casc = Cascade(device=a.device, use_detector=not a.no_detector,
                   stage_a_threshold=a.stage_a_threshold, stage_b_threshold=a.stage_b_threshold,
                   medsam_finetuned=a.medsam_finetuned)
    print(f"models loaded on {casc.device} in {casc.load_seconds}s; {len(images)} image(s)", flush=True)

    rows = []
    for i, (img, name) in enumerate(zip(images, names), 1):
        truth = R.lookup(labels, img) if labels else None
        try:
            res, art = casc.run(img, pixel_spacing_mm=a.pixel_spacing, stage_b_input=sb_input,
                                always_classify=bool(labels),
                                decide_on="whole" if a.stage_b_input == "whole" else "patch")
            if a.anonymize:
                res["image"] = name
            paths = R.save_image_outputs(res, art, a.out, name)
            dice = iou = None
            if truth and truth["mask"]:
                gt = R.gt_mask(truth["mask"], art["gray"].shape)
                if gt is not None:
                    dice, iou = R.seg_scores(art["mask"], gt)
            row = R.diagnosis_row(name if a.anonymize else img, res, paths,
                                  truth["label"] if truth else None, dice, iou)
            print(f"[{i}/{len(images)}] {name}: {res['decision']} ({res['seconds']}s)", flush=True)
        except Exception as e:      # one bad image must not stop a batch
            traceback.print_exc()
            err = repr(e)
            if a.anonymize:         # keep file names out of the outputs
                err = err.replace(img, name).replace(os.path.basename(img), name)
            row = R.diagnosis_row(name if a.anonymize else img, None, {},
                                  truth["label"] if truth else None, error=err)
            print(f"[{i}/{len(images)}] {name}: ERROR {e!r}", flush=True)
        row["_name"], row["_path"] = name, img
        rows.append(row)

    R.write_diagnosis(rows, os.path.join(a.out, "diagnosis.xlsx"))
    print(f"diagnosis: {os.path.join(a.out, 'diagnosis.xlsx')}")
    if labels:
        settings = {"stage_a_threshold": casc.thr_a, "stage_b_threshold": round(casc.thr_b, 4),
                    "stage_b_input_for_decision": "patch" if a.stage_b_input != "whole" else "whole",
                    "detector": not a.no_detector, "medsam": casc.medsam.weights, "labels_file": os.path.basename(a.labels),
                    "n_images": len(images)}
        R.write_metrics(rows, os.path.join(a.out, "metrics.xlsx"), settings)
        print(f"metrics:   {os.path.join(a.out, 'metrics.xlsx')}")
        ex = pick_examples(rows)
        if ex:
            os.makedirs(os.path.join(a.out, "examples"), exist_ok=True)
            tags = ["high_dice", "upper_mid_dice", "lower_mid_dice", "low_dice"] if len(ex) == 4 else ["dice"] * len(ex)
            for k, r in enumerate(ex, 1):
                res, art = casc.run(r["_path"], stage_b_input=sb_input, always_classify=True,
                                    decide_on="whole" if a.stage_b_input == "whole" else "patch")
                gt = R.gt_mask(R.lookup(labels, r["_path"])["mask"], art["gray"].shape)
                R.save_example(art["gray"], art["mask"], gt, art["crop_box"], r,
                               os.path.join(a.out, "examples", f"example_{k}_{tags[k-1]}_{r['_name']}.png"))
            print(f"examples:  {os.path.join(a.out, 'examples')} ({len(ex)})")


if __name__ == "__main__":
    main()
