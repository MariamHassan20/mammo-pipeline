"""
The cascade:

  1. Stage A  - normal vs abnormal on the whole mammogram
                P(abnormal) = p_benign + p_malignant from the 3-class Mammo-FM probe.
                P(abnormal) < threshold  ->  "normal", stop.
  2. Localise - lesion box from the Faster R-CNN detector;
                if the detector finds nothing, GradCAM of the Stage B model.
  3. Segment  - MedSAM prompted with that box -> lesion mask (largest component).
  4. Crop     - patch = mask bounding box + margin (falls back to the box itself if the mask is empty).
  5. Stage B  - benign vs malignant on the patch.

Research use only. Not a medical device; not for clinical decisions.
"""
import time
import numpy as np
import cv2
import torch

from . import config as C
from .models import read_gray, cap_size, StageA, StageB, LesionDetector, MedSAM


def _gradcam_box(cam, cap=C.GRADCAM_BOX_CAP):
    """Tight box around the CAM peak: cam >= max(0.75, q99.9), largest component, 1.5% pad, capped size."""
    H, W = cam.shape
    peak = (cam >= max(0.75, float(np.quantile(cam, 0.999)))).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(peak, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x0, y0, w, h = (int(st[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    pad = int(0.015 * max(H, W))
    box = [max(0, x0 - pad), max(0, y0 - pad), min(W, x0 + w + pad), min(H, y0 + h + pad)]
    bw, bh, mw, mh = box[2] - box[0], box[3] - box[1], int(cap * W), int(cap * H)
    if bw > mw or bh > mh:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        bw, bh = min(bw, mw), min(bh, mh)
        box = [max(0, cx - bw // 2), max(0, cy - bh // 2), min(W, cx + bw // 2), min(H, cy + bh // 2)]
    return box


def _crop_box(mask, box, shape, pad_frac=C.CROP_PAD_FRAC):
    """Patch around the lesion: mask bbox (or the prompt box) + pad_frac * longer side + 5 px."""
    ys, xs = np.where(mask > 0) if mask is not None else ([], [])
    if len(xs) > 20:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        source = "mask"
    else:
        x0, y0, x1, y1 = box
        source = "box"
    pad = int(pad_frac * max(y1 - y0, x1 - x0) + 5)
    H, W = shape
    return [max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)], source


class Cascade:
    def __init__(self, device=None, use_detector=True, stage_a_threshold=C.STAGE_A_THRESHOLD,
                 stage_b_threshold=None, medsam_finetuned=C.MEDSAM_FINETUNED):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t = time.time()
        self.stage_a = StageA(self.device)
        self.stage_b = StageB(self.device)
        self.detector = LesionDetector(self.device) if use_detector else None
        self.medsam = MedSAM(self.device, finetuned=medsam_finetuned)
        self.thr_a = float(stage_a_threshold)
        self.thr_b = float(stage_b_threshold) if stage_b_threshold is not None else self.stage_b.threshold
        self.load_seconds = round(time.time() - t, 1)

    def run(self, image_path, pixel_spacing_mm=None, stage_b_input="patch", always_classify=False,
            decide_on=None):
        """
        image_path        8-bit grayscale (or RGB) mammogram PNG/JPG, one view.
        pixel_spacing_mm  optional mm per pixel of the ORIGINAL image (from DICOM PixelSpacing),
                          used only to report lesion size in mm.
        stage_b_input     "patch" (default), "whole" or "both".
        always_classify   evaluation only: continue to segmentation and Stage B even when Stage A
                          says "normal". The decision stays "normal" and those later results are put
                          under "evaluation_only" instead of the normal keys.
        decide_on         "patch" or "whole": which Stage B score sets the decision when both are
                          computed (default: patch if available).
        Returns (result_dict, artefacts) where artefacts holds arrays for visualisation.
        """
        t0 = time.time()
        orig = read_gray(image_path)
        gray, scale = cap_size(orig)
        res = {"image": image_path, "image_size_px": [int(orig.shape[1]), int(orig.shape[0])],
               "working_scale": round(scale, 4), "warnings": []}
        art = {"gray": gray, "box": None, "mask": None, "crop_box": None}

        # 1. Stage A ----------------------------------------------------------------
        probs = self.stage_a(gray)
        p_abn = probs["benign"] + probs["malignant"]
        res["stage_a"] = {"probs": {k: round(v, 4) for k, v in probs.items()},
                          "p_abnormal": round(p_abn, 4), "threshold": self.thr_a,
                          "passed": bool(p_abn >= self.thr_a)}
        if p_abn < self.thr_a and not always_classify:
            res["decision"] = "normal"
            res["seconds"] = round(time.time() - t0, 2)
            return res, art

        # 2. Localise ---------------------------------------------------------------
        n_warn = len(res["warnings"])
        box, source, score = None, None, None
        if self.detector is not None:
            box, score = self.detector(gray)
            source = "detector" if box is not None else None
        if box is None:
            box = _gradcam_box(self.stage_b.gradcam(gray))
            source, score = ("gradcam", None) if box is not None else (None, None)
            if self.detector is not None:
                res["warnings"].append("detector found no lesion; used GradCAM fallback")
        res["localisation"] = {"box_xyxy": box, "source": source,
                               "score": round(score, 4) if score is not None else None}
        art["box"] = box

        # 3. Segment ----------------------------------------------------------------
        mask = None
        if box is not None:
            mask = self.medsam(gray, box)
            area = int(mask.sum())
            seg = {"area_px": area, "equiv_diameter_px": round(float(np.sqrt(4 * area / np.pi)), 1),
                   "medsam_weights": self.medsam.weights}
            if pixel_spacing_mm:
                sp = float(pixel_spacing_mm) / scale            # mm per pixel at the working resolution
                seg["area_mm2"] = round(area * sp * sp, 1)
                seg["equiv_diameter_mm"] = round(float(np.sqrt(4 * area * sp * sp / np.pi)), 1)
            if area == 0:
                res["warnings"].append("empty mask; patch taken from the prompt box")
            res["segmentation"] = seg
            art["mask"] = mask
        else:
            res["warnings"].append("no lesion box found; Stage B runs on the whole image")

        # 4. Crop -------------------------------------------------------------------
        if box is not None:
            cb, csrc = _crop_box(mask, box, gray.shape)
            patch = gray[cb[1]:cb[3], cb[0]:cb[2]]
            res["patch"] = {"box_xyxy": cb, "from": csrc}
            art["crop_box"] = cb
        else:
            patch = gray

        # 5. Stage B ----------------------------------------------------------------
        sb = {"threshold": round(self.thr_b, 4), "input": stage_b_input}
        if stage_b_input in ("patch", "both"):
            sb["p_malignant_patch"] = round(self.stage_b(patch), 4)
        if stage_b_input in ("whole", "both"):
            sb["p_malignant_whole"] = round(self.stage_b(gray), 4)
        if decide_on == "whole" and "p_malignant_whole" in sb:
            src = "whole"
        else:
            src = "patch" if "p_malignant_patch" in sb else "whole"
        p_dec = sb[f"p_malignant_{src}"]
        sb["decision_from"] = src
        sb["decision"] = "malignant" if p_dec >= self.thr_b else "benign"
        res["stage_b"] = sb
        res["decision"] = sb["decision"] if res["stage_a"]["passed"] else "normal"
        if not res["stage_a"]["passed"]:
            # always_classify run: the image stopped at Stage A, so the later steps are not part of its
            # result; keep them apart (used only for evaluation metrics such as AUC)
            res["evaluation_only"] = {k: res.pop(k) for k in ("localisation", "segmentation", "patch", "stage_b")
                                      if k in res}
            res["evaluation_only"]["warnings"] = res["warnings"][n_warn:]
            res["warnings"] = res["warnings"][:n_warn]
        res["seconds"] = round(time.time() - t0, 2)
        return res, art
