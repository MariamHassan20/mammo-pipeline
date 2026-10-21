"""
Model loaders and preprocessing for the four networks used by the cascade.

  Stage A   Mammo-FM image encoder (frozen) + linear 3-class head
  Detector  Faster R-CNN ResNet50-FPN (lesion box)
  MedSAM    SamModel (box-prompted segmentation)
  Stage B   fine-tuned Mammo-FM EfficientNet-B5 + Linear(2048, 2)
"""
import os
import numpy as np
import cv2
import torch
from torch import nn

from .mfm.model import MammoFM, MF_MEAN, MF_STD, MF_H, MF_W
from .mfm.efficientnet_custom import EfficientNet
from . import config as C


def _require(path, what, hint):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found at '{path}'. {hint}")


# ------------------------------------------------------------------ preprocessing
def read_gray(path):
    """Read an image as 8-bit grayscale (HxW uint8). 16-bit images (e.g. 12-bit mammograms) are min-max
    scaled to 0-255; reading them as plain 8-bit would keep only the high byte and give a near-black image."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if g is None:
        raise ValueError(f"cannot read image: {os.path.basename(path)}")
    if g.dtype != np.uint8:
        g = g.astype(np.float32)
        lo, hi = float(g.min()), float(g.max())
        g = ((g - lo) * (255.0 / max(hi - lo, 1e-6))).round().astype(np.uint8)
    if min(g.shape) < C.MIN_SIDE:
        raise ValueError(f"image too small ({g.shape[1]}x{g.shape[0]} px); expected a full mammogram view")
    if float(g.std()) < 2.0:
        raise ValueError("image is blank (almost uniform)")
    return g


def cap_size(gray, max_side=C.MAX_SIDE):
    """Downscale so the longer side is <= max_side. Returns (image, scale)."""
    mx = max(gray.shape)
    if mx <= max_side:
        return gray, 1.0
    s = max_side / mx
    out = cv2.resize(gray, (int(gray.shape[1] * s), int(gray.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return out, s


def _clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def prep_stage_a(gray):
    """Mammo-FM input: CLAHE -> resize 1520x912 (HxW) -> normalise -> [1,3,H,W]."""
    g = cv2.resize(_clahe(gray), (MF_W, MF_H), interpolation=cv2.INTER_AREA)
    x = (g.astype(np.float32) / 255.0 - MF_MEAN) / MF_STD
    return torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1)


def prep_stage_b(gray, res=C.STAGE_B_RES):
    """Stage B input: CLAHE -> resize res x 0.6*res (HxW) -> normalise -> [1,3,H,W]."""
    g = cv2.resize(_clahe(gray), (int(res * 0.6), res), interpolation=cv2.INTER_AREA)
    x = (g.astype(np.float32) / 255.0 - MF_MEAN) / MF_STD
    return torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1)


# ------------------------------------------------------------------ Stage A
class StageA:
    """Frozen Mammo-FM encoder + calibrated linear head (normal / benign / malignant)."""

    def __init__(self, device, mammofm_ckpt=C.MAMMOFM_CKPT, head_path=C.STAGE_A_HEAD):
        _require(mammofm_ckpt, "Mammo-FM checkpoint", "Run scripts/download_weights.py or set MAMMOFM_CKPT.")
        _require(head_path, "Stage A head", "See weights/README.md.")
        self.device = device
        ck = torch.load(mammofm_ckpt, map_location="cpu", weights_only=False)
        model = MammoFM()
        missing, _ = model.load_state_dict(ck["model"], strict=False)
        bad = [k for k in missing if k.startswith(("image_encoder", "image_projection"))]
        if bad:
            raise RuntimeError(f"Mammo-FM image weights failed to load: {bad[:5]}")
        self.encoder = model.image_encoder.to(device).eval()
        self.head = torch.load(head_path, map_location="cpu", weights_only=False)
        self.classes = list(self.head["classes"])          # ["normal", "benign", "malignant"]

    @torch.no_grad()
    def __call__(self, gray):
        feat = self.encoder(prep_stage_a(gray).to(self.device)).float().cpu().numpy()   # (1, 2048)
        h = self.head
        z = (feat - h["mean"]) / h["std"]
        logits = (z @ h["W"].T + h["b"]) / float(h.get("temperature", 1.0))
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = (e / e.sum(1, keepdims=True))[0]
        return {c: float(p[i]) for i, c in enumerate(self.classes)}


# ------------------------------------------------------------------ Stage B
class _ClassifierB(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b5", num_classes=1)
        self.head = nn.Linear(2048, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


class StageB:
    """Fine-tuned Mammo-FM EfficientNet-B5 + Linear(2048,2): benign (0) / malignant (1)."""

    def __init__(self, device, ckpt=C.STAGE_B_CKPT):
        _require(ckpt, "Stage B checkpoint", "See weights/README.md.")
        self.device = device
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = _ClassifierB()
        self.model.load_state_dict(ck["state_dict"])
        self.model.backbone.set_swish(memory_efficient=False)   # needed for GradCAM backward
        self.model.to(device).eval()
        self.threshold = float(ck.get("threshold", 0.5))
        self.res = int(ck.get("res", C.STAGE_B_RES))

    @torch.no_grad()
    def __call__(self, gray):
        x = prep_stage_b(gray, self.res).to(self.device)
        return float(torch.softmax(self.model(x), 1)[0, 1])     # P(malignant)

    def gradcam(self, gray):
        """GradCAM of the malignant logit on the last conv block, resized to the image."""
        x = prep_stage_b(gray, self.res).to(self.device)
        bb = self.model.backbone
        fm = bb.extract_features(x)
        fm.retain_grad()
        logit = self.model.head(torch.nn.functional.adaptive_avg_pool2d(fm, 1).flatten(1))[0, 1]
        self.model.zero_grad()
        logit.backward()
        alpha = fm.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((alpha * fm).sum(1))[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (gray.shape[1], gray.shape[0]))


# ------------------------------------------------------------------ Detector
class LesionDetector:
    """Faster R-CNN ResNet50-FPN with 2 classes. Returns the top box in image pixels."""

    def __init__(self, device, ckpt=C.DETECTOR_CKPT, min_score=C.DETECTOR_MIN_SCORE):
        _require(ckpt, "Lesion detector checkpoint", "See weights/README.md (or run with --no-detector).")
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = m.to(device).eval()
        self.device = device
        self.min_score = min_score

    @torch.no_grad()
    def __call__(self, gray):
        # same order as training (detector/train_detector.py): resize long side to 800, THEN CLAHE
        sc = C.DETECTOR_LONG_SIDE / max(gray.shape)
        g = _clahe(cv2.resize(gray, (int(gray.shape[1] * sc), int(gray.shape[0] * sc)), interpolation=cv2.INTER_AREA))
        x = torch.from_numpy(g.astype(np.float32) / 255.0)[None].repeat(3, 1, 1).to(self.device)
        out = self.model([x])[0]
        H, W = gray.shape
        for b, s in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()):   # sorted by score
            if float(s) < self.min_score:
                break
            x0, y0, x1, y1 = (b / sc).round().astype(int).tolist()
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 >= C.MIN_BOX_SIDE and y1 - y0 >= C.MIN_BOX_SIDE:            # skip degenerate boxes
                return [x0, y0, x1, y1], float(s)
        return None, 0.0


# ------------------------------------------------------------------ MedSAM
class MedSAM:
    """Box-prompted MedSAM. Original HF weights unless a fine-tuned state_dict is given."""

    def __init__(self, device, hf_id=C.MEDSAM_HF_ID, finetuned=C.MEDSAM_FINETUNED):
        from transformers import SamModel, SamProcessor
        self.proc = SamProcessor.from_pretrained(hf_id)
        m = SamModel.from_pretrained(hf_id)
        self.weights = hf_id
        if finetuned:
            _require(finetuned, "Fine-tuned MedSAM state_dict", "Unset CASCADE_MEDSAM_FINETUNED to use original MedSAM.")
            m.load_state_dict(torch.load(finetuned, map_location="cpu"))
            self.weights = f"{hf_id} + fine-tuned decoder ({os.path.basename(finetuned)})"
        self.model = m.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, gray, box):
        """box = [x0, y0, x1, y1] in image pixels. Returns a uint8 {0,1} mask (largest component)."""
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        inp = self.proc(rgb, input_boxes=[[box]], return_tensors="pt").to(self.device)
        out = self.model(**inp, multimask_output=False)
        mask = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu()
        )[0][0][0].numpy().astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n > 2:   # keep the largest connected component
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            mask = (lab == k).astype(np.uint8)
        return mask
