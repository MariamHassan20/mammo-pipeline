#!/usr/bin/env python
"""
Extract frozen Mammo-FM features (2048-d per image) so a dataset can be studied without moving the images.
Features cannot be turned back into images; only this file needs to be shared.

  python scripts/extract_features.py --input path/to/images/   --out features.npz
  python scripts/extract_features.py --csv   labels.csv         --out features.npz   # columns: image[,label][,source]
  python scripts/extract_features.py --input path/ --out f.npz --anonymize           # file names replaced by case IDs

Needs the Mammo-FM checkpoint in weights/ (python scripts/download_weights.py). Runs on a 10 GB GPU
(about 3 GB used) or on CPU. Images are preprocessed exactly as in the pipeline (CLAHE, 1520x912).

The output .npz holds: X (N x 2048, float16), image (file name or case ID), label, source, shape_hw,
and the preprocessing settings. With --anonymize the mapping case ID -> file is written next to the
output (<out>_case_map.csv); keep that file private.
"""
import os
import sys
import csv
import glob
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mammo_cascade.models import StageA, read_gray, cap_size, prep_stage_a  # noqa: E402
from mammo_cascade import config as C  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class _DS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        try:
            g, _ = cap_size(read_gray(self.paths[i]))
            return prep_stage_a(g)[0], i, True, g.shape[0], g.shape[1]
        except Exception:
            return torch.zeros(3, 1520, 912), i, False, 0, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="folder of images (searched recursively)")
    ap.add_argument("--csv", help="CSV with column image (path) and optional label, source")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anonymize", action="store_true", help="store case IDs instead of file names")
    a = ap.parse_args()
    if bool(a.input) == bool(a.csv):
        sys.exit("give exactly one of --input or --csv")
    if a.csv:
        with open(a.csv, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if (r.get("image") or "").strip()]
        base = os.path.dirname(os.path.abspath(a.csv))
        paths = [r["image"] if os.path.isabs(r["image"]) else os.path.join(base, r["image"]) for r in rows]
        labels = [(r.get("label") or "").strip() for r in rows]
        sources = [(r.get("source") or "").strip() for r in rows]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(a.input, "**", "*"), recursive=True) if p.lower().endswith(EXTS))
        labels, sources = [""] * len(paths), [""] * len(paths)
    if not paths:
        sys.exit("no images found")
    names = [f"case_{i:05d}" for i in range(1, len(paths) + 1)] if a.anonymize else [os.path.basename(p) for p in paths]
    if a.anonymize:
        cm = os.path.splitext(os.path.abspath(a.out))[0] + "_case_map.csv"
        with open(cm, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["case_id", "image"]); w.writerows(zip(names, paths))
        print(f"case map (keep private): {cm}")

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = StageA(device).encoder
    X = np.zeros((len(paths), 2048), np.float16); ok = np.zeros(len(paths), bool); hw = np.zeros((len(paths), 2), int)
    dl = DataLoader(_DS(paths), batch_size=a.batch, num_workers=a.workers)
    with torch.no_grad():
        for k, (x, idx, good, h, w) in enumerate(dl):
            f = enc(x.to(device)).float().cpu().numpy()
            X[idx.numpy()] = f.astype(np.float16); ok[idx.numpy()] = good.numpy(); hw[idx.numpy()] = np.stack([h.numpy(), w.numpy()], 1)
            if k % 100 == 0:
                print(f"  {min((k + 1) * a.batch, len(paths))}/{len(paths)}", flush=True)
    np.savez_compressed(a.out, X=X, ok=ok, image=np.array(names), label=np.array(labels), source=np.array(sources), shape_hw=hw,
                        settings=json.dumps({"backbone": "Mammo-FM EfficientNet-B5 image encoder", "input_hw": [1520, 912],
                                             "preprocessing": "CLAHE(2.0, 8x8), resize, mean/std normalise", "max_side": C.MAX_SIDE}))
    print(f"saved {a.out}: {ok.sum()} images ({(~ok).sum()} unreadable), {os.path.getsize(a.out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
