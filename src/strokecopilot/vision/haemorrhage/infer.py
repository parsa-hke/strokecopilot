"""Scan -> the contract's haemorrhage field, {present, confidence} (plan §6.1, §8.2, DEC-15).

confidence is the calibrated probability of a bleed; present compares it with the threshold chosen
on validation for high sensitivity, which may be below 0.5.
"""

import numpy as np
import torch

from strokecopilot.vision.haemorrhage import model as hmodel
from strokecopilot.vision.haemorrhage.aggregate import Calibrator


def scan_score(slice_probs, method, k=None):
    probs = np.sort(np.asarray(slice_probs, dtype=float))
    if probs.size == 0:
        raise ValueError("A scan needs at least one slice.")
    return float(probs[-1]) if method == "max" else float(probs[-k:].mean())


def haemorrhage_output(slice_probs, calibrator):
    """Slice probabilities of one scan -> {"present": bool, "confidence": float in [0, 1]}."""
    confidence = float(calibrator.apply(np.array([scan_score(slice_probs, calibrator.method, calibrator.k)]))[0])
    return {"present": bool(confidence >= calibrator.threshold), "confidence": round(confidence, 4)}


@torch.no_grad()
def slice_probs(net, images, device, batch_size=64):
    """uint8 S x H x W x 3 windowed slices (rsna.window_hu) -> S probabilities."""
    net.eval()
    images = torch.as_tensor(np.ascontiguousarray(images))
    out = [torch.sigmoid(net(hmodel.normalise(images[i : i + batch_size], device)).float().squeeze(1)).cpu() for i in range(0, len(images), batch_size)]
    return torch.cat(out).numpy()


def infer_scan(net, images, calibrator: Calibrator, device):
    return haemorrhage_output(slice_probs(net, images, device), calibrator)
