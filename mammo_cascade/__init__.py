"""Mammography cascade: normal/abnormal -> lesion localisation -> MedSAM segmentation -> benign/malignant on the patch.

Research use only. Not a medical device; not for clinical decisions.
"""
from .pipeline import Cascade

__all__ = ["Cascade"]
