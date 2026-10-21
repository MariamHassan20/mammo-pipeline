#!/usr/bin/env python
"""
Download the public weights into weights/ (Mammo-FM) and the Hugging Face cache (MedSAM).

Mammo-FM (batmanLab/Mammo-FM) is released for non-commercial academic research only and
may be gated: log in first (`huggingface-cli login`) and accept its terms on the model page.
Our own checkpoints (Stage A head, Stage B, detector) are NOT downloaded here; see weights/README.md.
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


if __name__ == "__main__":
    main()
