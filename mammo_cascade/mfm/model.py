"""
mfm/model.py — minimal Mammo-FM image encoder + classifier head.

Only needs: torch, numpy, opencv (cv2), omegaconf. NO timm / transformers /
torchvision / huggingface_hub — the Mammo-FM checkpoint uses the bundled custom
EfficientNet-B5, and classification uses the image branch only.

Loads:
  - Mammo-FM base checkpoint (batmanLab) -> 2048-d image features
  - trained linear head (normal/benign/malignant) with calibrated temperature and
    a val-tuned malignant threshold

DISCLAIMER: research/education only. Not for clinical use.
"""
import os
import numpy as np
import cv2
import torch
from torch import nn

from .efficientnet_custom import EfficientNet
from .projection import LinearProjectionHead

# Mammo-FM training preprocessing constants
MF_MEAN, MF_STD = 0.3089279, 0.25053555408335154
MF_H, MF_W = 1520, 912
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MammoFM(nn.Module):
    """Image branch of Mammo-FM. encode_image() -> (B, 2048) features."""
    def __init__(self):
        super().__init__()
        # from_name (NOT from_pretrained) -> no ImageNet download; weights come from
        # the Mammo-FM checkpoint. num_classes=1 matches the checkpoint's _fc shape.
        self.image_encoder = EfficientNet.from_name("efficientnet-b5", num_classes=1)
        self.image_projection = LinearProjectionHead(2048, 512)   # loads image_projection.*
        self.text_projection = LinearProjectionHead(768, 512)     # loads (unused) text_projection.*
        self.logit_scale = nn.Parameter(torch.zeros([]))

    @torch.no_grad()
    def encode_image(self, x):
        return self.image_encoder(x)


def _resolve_ckpt(ckpt_path=None):
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path
    env = os.environ.get("MAMMOFM_CKPT")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(here, "weights", "Mammo-FM_BatmanlabTrained_CLIP.tar")
    if os.path.exists(local):
        return local
    raise FileNotFoundError(
        "Mammo-FM checkpoint not found. Put Mammo-FM_BatmanlabTrained_CLIP.tar in "
        "./weights/ or set MAMMOFM_CKPT=/path/to/it.")


def load_mammofm(ckpt_path=None):
    """Build the image encoder and load the Mammo-FM weights (offline-friendly)."""
    ck = torch.load(_resolve_ckpt(ckpt_path), map_location="cpu", weights_only=False)
    model = MammoFM()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    # image_encoder.* / image_projection.* / text_projection.* / logit_scale load;
    # text_encoder.* keys are 'unexpected' (ignored) — we never use text.
    img_missing = [k for k in missing if k.startswith(("image_encoder", "image_projection"))]
    if img_missing:
        raise RuntimeError(f"Image weights failed to load: {img_missing[:5]} ...")
    return model.to(DEVICE).eval()


def preprocess(image_path):
    """Grayscale -> CLAHE -> resize 912x1520 -> normalize -> [1,3,H,W] tensor."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    mx = max(img.shape[:2])
    if mx > 4000:                       # cap huge images (memory safety)
        s = 4000.0 / mx
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    img = cv2.resize(img, (MF_W, MF_H), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - MF_MEAN) / MF_STD
    return torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1).to(DEVICE)


def load_head(head_path):
    """Load the trained classifier head (dict with W,b,mean,std,temperature,classes,
    malignant_threshold)."""
    return torch.load(head_path, map_location="cpu", weights_only=False)


def classify(model, head, image_path):
    """Return {probs:{class:p}, prediction, malignant_prob, malignant_flag}."""
    feat = model.encode_image(preprocess(image_path)).float().cpu().numpy()  # (1,2048)
    z = (feat - head["mean"]) / head["std"]
    logits = (z @ head["W"].T + head["b"]) / float(head.get("temperature", 1.0))
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = (e / e.sum(1, keepdims=True))[0]
    classes = head["classes"]
    probs = {c: round(float(p[i]), 4) for i, c in enumerate(classes)}
    out = {"probs": probs, "prediction": classes[int(p.argmax())]}
    if "malignant" in classes:
        thr = head.get("malignant_threshold") or 0.5
        mp = probs["malignant"]
        out["malignant_prob"] = mp
        out["malignant_threshold"] = round(float(thr), 3)
        out["malignant_flag"] = bool(mp >= thr)
    return out
