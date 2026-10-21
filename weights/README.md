# Weights

Weights are **not** stored in git (file sizes). Download them with the links below, put them in this folder, or
point to them with the environment variables shown.

| File (default name) | Size | What it is | Env var to override | How to get it |
|---|---|---|---|---|
| `Mammo-FM_BatmanlabTrained_CLIP.tar` | 2.1 GB | Mammo-FM base checkpoint (batmanLab). Used by Stage A. | `MAMMOFM_CKPT` | `python scripts/download_weights.py` (Hugging Face `batmanLab/Mammo-FM`; log in and accept its terms first). **Non-commercial academic licence; do not re-upload.** |
| `mammo_fm_head.pt` | 63 KB | Stage A head: linear 3-class probe (normal / benign / malignant) on frozen Mammo-FM features, with temperature T = 0.961 | `CASCADE_STAGE_A_HEAD` | [download](LINK_STAGE_A_HEAD) |
| `stage_b_finetuned.pt` | 110 MB | Stage B: fully fine-tuned Mammo-FM EfficientNet-B5 + Linear(2048, 2), benign (0) / malignant (1). Stores its own threshold (0.058) and input resolution (760). **Contains fine-tuned Mammo-FM weights, so the Mammo-FM licence applies.** | `CASCADE_STAGE_B_CKPT` | [download](LINK_STAGE_B) |
| `lesion_detector.pt` | 159 MB | Faster R-CNN ResNet50-FPN, 2 classes (background, lesion) | `CASCADE_DETECTOR_CKPT` | [download](LINK_DETECTOR) (or run with `--no-detector`) |
| MedSAM | ~360 MB | `wanglab/medsam-vit-base` (Apache-2.0), original weights | `CASCADE_MEDSAM_ID` | downloaded automatically by `transformers` (or `scripts/download_weights.py`) |
| *(optional)* `medsam_finetuned.pt` | 358 MB | full SamModel state_dict with our fine-tuned mask decoder | `CASCADE_MEDSAM_FINETUNED` | from the authors; off by default |

All paths can also be changed at once with `CASCADE_WEIGHTS_DIR=/some/folder`.

SHA-256 checksums:
```
58ae959de52d563acc1d43f50571498eddcd4fb7ca599e706540fdd319e3599b  Mammo-FM_BatmanlabTrained_CLIP.tar
8242ab1e5554946989435de747ac5577b90fa11bd01bdf34064893babd069377  mammo_fm_head.pt
0b2e62ea5af0a29a8f52d992d65ed4d999a4a7fb71cc283907191da9fe3bfac2  stage_b_finetuned.pt
da1f3a9ad24ff22df93ad4914c2c2030548550ad2ff3b6066e162822d0474ecb  lesion_detector.pt
1cc236a14943a20a672aa3783ee2eb4cfca3f4cec547de93f337f7c625882f08  medsam_finetuned.pt   (optional)
```
Check with `sha256sum weights/*`.
