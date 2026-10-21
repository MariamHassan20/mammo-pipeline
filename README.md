# Mammography cascade: normal/abnormal → lesion segmentation → benign/malignant

> **Research use only.** Not a medical device. Not validated for clinical use. Do not use it for diagnosis or
> any clinical decision.

One mammogram view in, one decision out:

```
mammogram (PNG/JPG, one view)
  │
  ├─ 1. Stage A  Mammo-FM (frozen) + linear head ─► P(normal), P(benign), P(malignant)
  │              P(abnormal) = P(benign) + P(malignant)
  │              P(abnormal) < 0.762  ──────────────────────────────►  "normal"  (stop)
  │
  ├─ 2. Localise Faster R-CNN lesion detector ─► box
  │              (no detection → GradCAM of the Stage B model ─► box)
  ├─ 3. Segment  MedSAM, prompted with the box ─► lesion mask (largest component)
  ├─ 4. Crop     patch = mask bounding box + 10% margin
  │
  └─ 5. Stage B  fine-tuned Mammo-FM classifier on the patch ─► P(malignant)
                 P(malignant) ≥ 0.058 ─► "malignant", else "benign"
```

## Quick start

```bash
# 1. environment (tested: Python 3.10, torch 2.3.0 + CUDA 12.1, transformers 4.37.2)
conda env create -f environment.yml && conda activate mammo-cascade
#    or with pip in an existing Python 3.10 env:
#    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 2. weights (see weights/README.md)
huggingface-cli login                # Mammo-FM may be gated: accept its terms on the HF page
python scripts/download_weights.py   # Mammo-FM -> weights/, MedSAM -> HF cache
#    then copy mammo_fm_head.pt, stage_b_finetuned.pt, lesion_detector.pt into weights/

# 3. run
python run_cascade.py --input path/to/view.png --out outputs/
python run_cascade.py --input path/to/folder/  --out outputs/ --device cuda:0
```

**Hardware:** one GPU with at least 6 GB of memory is recommended (all four models stay loaded). Models load
once at start-up (about 1 minute); after that each image takes about a second on a recent GPU. CPU also works,
but is much slower.

### Options
| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | image file, or folder (searched recursively for .png/.jpg/.tif) |
| `--out` | `outputs` | output folder (created if missing) |
| `--labels` | none | optional ground-truth CSV (`image,label[,mask]`) → adds `metrics.xlsx` and `examples/` |
| `--anonymize` | off | name all outputs `case_0001`, `case_0002`, … instead of the image file names; the table linking case IDs to images is written next to (not inside) the output folder as `<out>_case_map.csv` |
| `--device` | `cuda` if available | e.g. `cuda:0`, `cpu` |
| `--no-detector` | off | skip the detector; localise with GradCAM only |
| `--stage-b-input` | `patch` | `patch`, `whole` (whole image) or `both` (both scores are reported; the decision uses the patch) |
| `--pixel-spacing` | none | mm per pixel of the original image (DICOM `PixelSpacing`), to report lesion size in mm |
| `--stage-a-threshold` | `0.762` | P(abnormal) threshold |
| `--stage-b-threshold` | from checkpoint (`0.058`) | P(malignant) threshold |
| `--medsam-finetuned` | none | path to our fine-tuned MedSAM state_dict (default: original MedSAM) |

### Output folder
```
<out>/
├── segmented_tumors/   <name>_patch.png  tumour crop that Stage B classified
│                       <name>_mask.png   tumour mask (white = tumour)
├── overlays/           <name>_overlay.png  mammogram with box (orange = detector, magenta = GradCAM),
│                                           tumour outline (red), scores and decision
├── json/               <name>.json       everything for one image (format below)
├── diagnosis.xlsx      one row per image: decision, all probabilities, box, tumour size, file paths,
│                       warnings/errors (+ true label and Dice/IoU with --labels)
├── metrics.xlsx        only with --labels: accuracy, AUC (with 95% CI), F1, precision, sensitivity,
│                       specificity, confusion table, Stage A pass rates, segmentation Dice/IoU
└── examples/           only with --labels and true masks: 4 cases across the Dice range, each showing
                        the tumour region (green = true tumour, red = our segmentation), our tumour patch
                        and the true tumour patch
```
Normal images get a JSON, an overlay and a row in `diagnosis.xlsx` (no patch or mask). A bad image never stops
a batch; its error is recorded in `diagnosis.xlsx`.

**Labels file** (`--labels`): CSV with columns `image` (same path as given to `--input`, or just the file
name), `label` (`normal`, `benign`, `malignant`, or `0`/`1` for benign/malignant) and optional `mask` (path to
the true tumour mask). With labels, Stage B is also scored for images Stage A calls normal, so an AUC can be
computed; the decision itself still follows Stage A.
```bash
python run_cascade.py --input test_images/ --labels labels.csv --out outputs/test_run/
```

```json
{
  "image": "view.png",
  "image_size_px": [W, H],
  "working_scale": 1.0,
  "stage_a": {"probs": {"normal": p, "benign": p, "malignant": p},
              "p_abnormal": p, "threshold": 0.762, "passed": true},
  "localisation": {"box_xyxy": [x0, y0, x1, y1], "source": "detector | gradcam", "score": s},
  "segmentation": {"area_px": n, "equiv_diameter_px": d, "medsam_weights": "wanglab/medsam-vit-base",
                   "area_mm2": "(only with --pixel-spacing)", "equiv_diameter_mm": "(only with --pixel-spacing)"},
  "patch": {"box_xyxy": [x0, y0, x1, y1], "from": "mask | box"},
  "stage_b": {"threshold": 0.058, "input": "patch", "p_malignant_patch": p, "decision": "benign | malignant"},
  "decision": "normal | benign | malignant",
  "warnings": [],
  "seconds": t
}
```
If Stage A says "normal", only `stage_a` and `decision` are filled. All coordinates are in the *working* image:
the original, downscaled so its longer side is at most 4000 px. `working_scale` is the factor applied
(1.0 = unchanged).

### Python API
```python
from mammo_cascade import Cascade
from mammo_cascade.visualize import save_outputs

casc = Cascade(device="cuda")                    # loads all models once
result, art = casc.run("view.png", pixel_spacing_mm=0.07, stage_b_input="patch")
save_outputs(result, art, "outputs/view")        # writes the PNGs
```

### Evaluation script (CSV output, resumable)
For large evaluations that may be interrupted, `scripts/evaluate_bm.py` runs the full chain on a labelled CSV (`image`, `label` = benign/malignant, optional
`mask`, `patient`, `group`) and writes `per_image.csv` (resumable) and `metrics.json`. The metrics cover AUC for
patch and whole image with patient-bootstrap CIs, sensitivity/specificity at the threshold, Stage A pass rates,
box sources, Dice when lesion masks are given, and per-group AUCs.
```bash
python scripts/evaluate_bm.py --csv labelled.csv --out eval_out/ --device cuda:0
```

## The models

| Step | Model | Trained on | Input preprocessing | Output |
|---|---|---|---|---|
| Stage A | Mammo-FM image encoder (EfficientNet-B5, frozen) + `Linear(2048, 3)`, temperature-calibrated | Mammo-Bench (public, 7 sources), patient-level split | CLAHE (clip 2.0, 8×8) → resize 1520×912 (H×W) → normalise (mean 0.3089, std 0.2505) → 3 channels | P(normal / benign / malignant) |
| Detector | Faster R-CNN ResNet50-FPN (COCO init), 2 classes (background, lesion) | private multi-centre mass dataset, patient-level split | resize long side to 800 → CLAHE → /255 → 3 channels | top-scoring lesion box |
| Segment | MedSAM `wanglab/medsam-vit-base` (original weights; optional fine-tuned decoder) | MedSAM's public training data | gray → RGB, longest side 1024 (SamProcessor), box prompt in image pixels | binary mask (logit > 0), largest component kept |
| Stage B | Mammo-FM EfficientNet-B5 fully fine-tuned + `Linear(2048, 2)` | private benign/malignant dataset (whole images) | CLAHE → resize 760×456 (H×W) → normalise → 3 channels | P(malignant) |

Thresholds were chosen on validation data during development:
- **Stage A, 0.762:** maximises balanced accuracy for normal vs abnormal.
- **Stage B, 0.058:** maximises malignant F2, so it favours sensitivity. It is stored in the Stage B checkpoint.

Both can be changed with the CLI flags above.

## Usage notes

1. **Stage B was trained on whole images.** With the default `patch` input it classifies a lesion crop, which
   differs from what it saw in training. Use `--stage-b-input both` to get the patch and whole-image scores side by side.
2. **Thresholds are not universal.** Re-tune both on validation data from your own site (`--stage-a-threshold`,
   `--stage-b-threshold`).
3. **The detector was trained on masses only.** Calcifications and architectural distortions are not targeted.
4. **GradCAM fallback** only runs when the detector finds no box. A warning is written to the JSON when it is
   used.
5. **MedSAM weights:** original `wanglab/medsam-vit-base` by default. Our fine-tuned decoder can be enabled with
   `--medsam-finetuned path/to/medsam_finetuned.pt`.
6. **Input** is an 8-bit PNG/JPG of one view. The code does no DICOM reading, no laterality or view handling,
   and no CC/MLO fusion. Images are read in grayscale; 16-bit PNGs are reduced to 8-bit. Images whose longer
   side exceeds 4000 px are downscaled first.
7. **Lesion size in mm** needs `--pixel-spacing` (mm/px of the original image). The correction for the working
   scale is applied automatically.

## Licences
- This code: add your licence.
- `mammo_cascade/mfm/efficientnet_custom.py`, `efficient_net_custom_utils.py`, `projection.py`: from the
  Mammo-FM repository (batmanLab, Apache-2.0).
- **Mammo-FM weights**: custom academic licence (non-commercial research only; no redistribution, no hosting
  as a service). `stage_b_finetuned.pt` and `mammo_fm_head.pt` are derived from Mammo-FM.
- MedSAM: Apache-2.0. torchvision Faster R-CNN: BSD-3-Clause code.
