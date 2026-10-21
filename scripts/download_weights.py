#!/usr/bin/env python
"""
Download the public weights into weights/ (Mammo-FM) and the Hugging Face cache (MedSAM).

Mammo-FM (batmanLab/Mammo-FM) is released for non-commercial academic research only.
Our own checkpoints (Stage A head, Stage B, detector) are NOT downloaded here: get them from the links in
README.md and put them in weights/. The script lists which of them are still missing.
"""
import os
import shutil
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.environ.get("CASCADE_WEIGHTS_DIR", os.path.join(ROOT, "weights"))


def main():
    os.makedirs(WEIGHTS, exist_ok=True)
    dst = os.path.join(WEIGHTS, "Mammo-FM_BatmanlabTrained_CLIP.tar")
    if os.path.exists(dst):
        print(f"Mammo-FM already present: {dst}")
    else:
        p = hf_hub_download("batmanLab/Mammo-FM", "Mammo-FM_BatmanlabTrained_CLIP.tar")
        shutil.copy(p, dst)
        print(f"Mammo-FM -> {dst}")
    snapshot_download("wanglab/medsam-vit-base")
    print("MedSAM (wanglab/medsam-vit-base) cached in the Hugging Face cache")
    missing = [f for f in ("mammo_fm_head.pt", "stage_b_finetuned.pt", "lesion_detector.pt")
               if not os.path.exists(os.path.join(WEIGHTS, f))]
    if missing:
        print(f"\nStill missing in {WEIGHTS}: {', '.join(missing)}\n"
              "Download them from the links in README.md (section 'Weights', step 2).")
    else:
        print("\nAll weights are in place.")


if __name__ == "__main__":
    main()
