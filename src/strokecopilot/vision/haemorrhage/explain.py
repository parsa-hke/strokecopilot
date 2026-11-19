"""Why does Vision B flag presumed bleed-free ISLES'24 scans? (Item 2, minimal: no paid GPU, no API, no re-run.)

Three pieces, all on the uint8 brain / subdural / bone image both datasets share (data/prep_rsna.WINDOWS, 320 px):

- grad_cam(): Grad-CAM on the classifier's last convolutional block (ResNet-50's layer4) for its single logit.
- tissue_masks() and heat_shares(): where the heat falls, as shares of the map inside brain-density tissue, bone,
  other head tissue (scalp, muscle, sinuses) and outside the head.
- slice_measures(): per slice, the brain fraction (brain-density tissue as a share of the head's area) and the
  height in the head (0 at the lowest slice with head, 1 at the highest), for the candidate rules in slab_rules.py.

The masks are density rules, not a segmentation: brain-density means HU 0-80 (the brain window's open range) away from
the head's edge and from bone. They are measured the same way on RSNA and ISLES'24, which is what the comparison needs;
they are not a brain mask to report volumes from. Thresholds below are in HU, converted to the bone channel's pixel
values from its window (centre 600, width 2800).
"""

import numpy as np

from strokecopilot.data.prep_rsna import WINDOWS

BRAIN, SUBDURAL, BONE = 0, 1, 2  # channel order of the windowed image
AIR_BELOW_HU = -500.0  # head = denser than this
BONE_ABOVE_HU = 300.0  # cortical bone
EDGE_PX = 8  # brain tissue must lie this far inside the head's outline (about 6 mm at 320 px over a 250 mm field)
BONE_MARGIN_PX = 2
MIN_HEAD_SHARE = 0.10  # a slice counts as head for the height when its head area is this share of the scan's largest


def _hu_to_channel(hu, channel):
    centre, width = WINDOWS[channel]
    return float(np.clip((hu - (centre - width / 2)) / width, 0, 1) * 255)


def tissue_masks(image):
    """uint8 H x W x 3 windowed slice -> {"head", "bone", "brain", "other"} boolean H x W masks. other = head tissue
    that is neither bone nor brain-density (scalp, muscle, sinus walls, fat)."""
    from scipy import ndimage

    image = np.asarray(image)
    bone_ch, brain_ch = image[..., BONE].astype(float), image[..., BRAIN]
    head = ndimage.binary_fill_holes(bone_ch > _hu_to_channel(AIR_BELOW_HU, BONE))
    bone = bone_ch > _hu_to_channel(BONE_ABOVE_HU, BONE)
    inner = ndimage.binary_erosion(head, iterations=EDGE_PX) if head.any() else head
    near_bone = ndimage.binary_dilation(bone, iterations=BONE_MARGIN_PX) if bone.any() else bone
    brain = inner & ~near_bone & (brain_ch > 0) & (brain_ch < 255)
    return {"head": head, "bone": bone, "brain": brain, "other": head & ~bone & ~brain}


def heat_shares(cam, masks):
    """Shares of a non-negative heat map in brain, bone, other head tissue and outside the head; they sum to 1. All
    None for an empty map."""
    cam = np.asarray(cam, dtype=float)
    total = cam.sum()
    keys = ("brain", "bone", "other", "outside")
    if total <= 0:
        return dict.fromkeys(keys)
    outside = ~masks["head"]
    return {k: float((cam * (outside if k == "outside" else masks[k])).sum() / total) for k in keys}


def slice_measures(images):
    """uint8 S x H x W x 3 slices of one scan, in order along the body -> per slice {"brain_fraction", "height",
    "head_px"}. brain_fraction = brain-density pixels / head pixels (0 with no head). height is the slice's place
    between the lowest and highest slices whose head area is at least MIN_HEAD_SHARE of the scan's largest (0 to 1;
    outside that range it is clipped, and a scan with one such slice has height 0.5 there)."""
    rows = []
    for image in images:
        m = tissue_masks(image)
        head = int(m["head"].sum())
        rows.append({"head_px": head, "brain_fraction": float(m["brain"].sum() / head) if head else 0.0})
    heads = np.array([r["head_px"] for r in rows], dtype=float)
    present = np.flatnonzero(heads >= MIN_HEAD_SHARE * heads.max()) if heads.max() > 0 else np.array([], dtype=int)
    lo, hi = (int(present[0]), int(present[-1])) if present.size else (0, 0)
    for i, r in enumerate(rows):
        r["height"] = 0.5 if hi == lo else float(np.clip((i - lo) / (hi - lo), 0, 1))
    return rows


def target_layer(net):
    """The last convolutional block: ResNet's layer4, EfficientNet's conv_head (the two models of plan §8.2)."""
    for name in ("layer4", "conv_head"):
        if hasattr(net, name):
            return getattr(net, name)
    raise ValueError("No known last convolutional block (layer4 or conv_head) on this model.")


def grad_cam(net, images, device="cpu", layer=None):
    """uint8 N x H x W x 3 windowed slices -> (N x H x W Grad-CAM maps in [0, 1], N probabilities).

    For the single "any haemorrhage" logit: the block's activations weighted by the mean gradient of the logit per
    channel, summed, ReLU, upsampled to the image size and scaled by each map's maximum (an all-zero map stays zero).
    """
    import torch
    import torch.nn.functional as F

    from strokecopilot.vision.haemorrhage import model as hmodel

    layer = layer or target_layer(net)
    saved = {}

    def keep(_, __, output):
        saved["act"] = output
        output.register_hook(lambda grad: saved.__setitem__("grad", grad))

    handle = layer.register_forward_hook(keep)
    net.eval()
    try:
        x = hmodel.normalise(torch.as_tensor(np.ascontiguousarray(images)), device)
        with torch.enable_grad():
            logits = net(x).float().squeeze(1)
            net.zero_grad(set_to_none=True)
            logits.sum().backward()
        act, grad = saved["act"].detach().float(), saved["grad"].detach().float()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12).view(-1, 1, 1)
        cam = torch.where(cam.flatten(1).max(dim=1).values.view(-1, 1, 1) > 0, cam / peak, cam)
        return cam.cpu().numpy(), torch.sigmoid(logits.detach()).cpu().numpy()
    finally:
        handle.remove()


def explain_slices(net, images, device="cpu", batch_size=16):
    """Grad-CAM and its heat shares for each slice: [{"prob", "brain", "bone", "other", "outside"}] and the maps."""
    rows, maps = [], []
    for i in range(0, len(images), batch_size):
        cams, probs = grad_cam(net, images[i : i + batch_size], device)
        for image, cam, prob in zip(images[i : i + batch_size], cams, probs):
            rows.append({"prob": float(prob), **heat_shares(cam, tissue_masks(image))})
            maps.append(cam)
    return rows, np.stack(maps) if maps else np.empty((0,))
