"""
Default paths and thresholds for the cascade.

Every weight path can be overridden with an environment variable (MedSAM's fine-tuned decoder also with
--medsam-finetuned). Default file names are looked up in the repo's weights/ folder (or CASCADE_WEIGHTS_DIR);
a relative path given in an environment variable is resolved against the current working directory.

Thresholds were chosen on VALIDATION data during development (never on test):
  - STAGE_A_THRESHOLD 0.762: P(abnormal) = p_benign + p_malignant from the 3-class
    Mammo-FM probe; maximises balanced accuracy (sens+spec)/2 on the Mammo-Bench
    validation split.
  - Stage B threshold: stored inside the Stage B checkpoint ("threshold"), chosen to
    maximise malignant F2 (recall-weighted) on validation. For the shipped checkpoint
    it is 0.058.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.environ.get("CASCADE_WEIGHTS_DIR", os.path.join(REPO_ROOT, "weights"))


def _w(env, filename):
    return os.environ.get(env, os.path.join(WEIGHTS_DIR, filename))


# --- weights -----------------------------------------------------------------
# Mammo-FM base checkpoint (batmanLab/Mammo-FM on Hugging Face). Not redistributable:
# each user downloads it under its own licence (scripts/download_weights.py).
MAMMOFM_CKPT = _w("MAMMOFM_CKPT", "Mammo-FM_BatmanlabTrained_CLIP.tar")
# Stage A: linear 3-class head (normal/benign/malignant) on frozen Mammo-FM features.
STAGE_A_HEAD = _w("CASCADE_STAGE_A_HEAD", "mammo_fm_head.pt")
# Stage B: fully fine-tuned Mammo-FM EfficientNet-B5 + Linear(2048, 2) (benign/malignant).
STAGE_B_CKPT = _w("CASCADE_STAGE_B_CKPT", "stage_b_finetuned.pt")
# Lesion detector: torchvision Faster R-CNN ResNet50-FPN, 2 classes (background, lesion).
DETECTOR_CKPT = _w("CASCADE_DETECTOR_CKPT", "lesion_detector.pt")
# MedSAM: original weights from Hugging Face by default.
MEDSAM_HF_ID = os.environ.get("CASCADE_MEDSAM_ID", "wanglab/medsam-vit-base")
# Optional: full SamModel state_dict with our fine-tuned mask decoder (None = original MedSAM).
MEDSAM_FINETUNED = os.environ.get("CASCADE_MEDSAM_FINETUNED") or None

# --- thresholds / settings -----------------------------------------------------
STAGE_A_THRESHOLD = 0.762
DETECTOR_MIN_SCORE = 0.05          # torchvision's own default box_score_thresh
DETECTOR_LONG_SIDE = 800           # detector was trained with the long side resized to 800
STAGE_B_RES = 760                  # Stage B input: 760 x 456 (H x W)
MAX_SIDE = 4000                    # images with a longer side are downscaled first
MIN_SIDE = 256                     # smaller images are rejected (not a full mammogram view)
MIN_BOX_SIDE = 8                   # detector boxes thinner than this (px) are ignored
CROP_PAD_FRAC = 0.10               # crop = mask bbox + 10% of its longer side + 5 px
GRADCAM_BOX_CAP = 0.22             # GradCAM fallback box capped at 22% of W and H
