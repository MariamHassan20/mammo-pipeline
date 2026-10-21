#!/usr/bin/env python
"""
Download all weights:
  - Mammo-FM (batmanLab/Mammo-FM, Hugging Face)  -> weights/
  - MedSAM   (wanglab/medsam-vit-base)           -> Hugging Face cache
  - our trained checkpoints (GitHub release v1.0) -> weights/   (Stage A head, Stage B, detector)
Each of our checkpoints is checked against its SHA-256 (also listed in weights/README.md).

Mammo-FM is released for non-commercial academic research only.
"""
import os
import shutil
import hashlib
import urllib.request
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.environ.get("CASCADE_WEIGHTS_DIR", os.path.join(ROOT, "weights"))
RELEASE = "https://github.com/MariamHassan20/mammo-pipeline/releases/download/v1.0"
OURS = {
    "mammo_fm_head.pt": "8242ab1e5554946989435de747ac5577b90fa11bd01bdf34064893babd069377",
    "stage_b_finetuned.pt": "ec29d8db7aa974617968b77c3022a62f72661efc12c1168e6e1279d0aabbaa77",
    "lesion_detector.pt": "da1f3a9ad24ff22df93ad4914c2c2030548550ad2ff3b6066e162822d0474ecb",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    for name, sha in OURS.items():
        dst = os.path.join(WEIGHTS, name)
        if os.path.exists(dst) and _sha256(dst) == sha:
            print(f"{name} already present")
            continue
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(f"{RELEASE}/{name}", dst + ".part")
        if _sha256(dst + ".part") != sha:
            os.remove(dst + ".part")
            raise SystemExit(f"{name}: checksum mismatch; download it by hand from {RELEASE}/{name}")
        os.replace(dst + ".part", dst)
        print(f"{name} -> {dst}")
    print("\nAll weights are in place.")


if __name__ == "__main__":
    main()
