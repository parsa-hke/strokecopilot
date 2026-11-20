"""Phase 5 task D (DEC-33): BiomedCLIP against EXP-6's ResNet-50, and per-subtype probes on EXP-6's frozen features.

Fixed before any score:
- **Inputs:** the same uint8 three-window slices EXP-6 reads (brain, subdural, bone as the three colour channels),
  resized to 224 px (bicubic) and normalised with BiomedCLIP's own constants, on the GPU (`to_input`).
- **BiomedCLIP, linear probe:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` image embeddings (normalised,
  512-d) of a seeded sample of TRAIN_SAMPLE training slices; scikit-learn logistic regression (C 1.0, lbfgs,
  1,000 iterations) on "any haemorrhage".
- **BiomedCLIP, zero-shot:** the two fixed prompts in PROMPTS; slice probability = softmax of the image-text logits.
- **Scans:** EXP-6's aggregation for every model (mean of the top 2 slice probabilities); Platt scaling fitted on RSNA
  validation for the probe and zero-shot, as EXP-6's own calibration was.
- **Read-outs on RSNA validation:** scan AUC; sensitivity and specificity at the threshold reaching 96% sensitivity on
  validation (DEC-28's target; in-sample for every model alike); per-patient Brier score of the calibrated scan
  probability, paired with EXP-6, for DEC-36's T9.
- **Per-subtype probes:** EXP-6's frozen 2,048-d pooled features of the same training sample; one logistic regression per
  subtype; slice and scan AUC per subtype on validation, beside EXP-6's own "any" score as a detector of that subtype.
- **ISLES'24 (Mac, `isles_main`):** the probe and zero-shot calls on the 129 development scans (EXP-50's slabs) at their
  RSNA thresholds, against EXP-6's 68 flags (McNemar).
Only aggregates leave Kaggle, plus the probe's 513 numbers and the thresholds (`probe.json`), which the ISLES'24 half
reads. The test split is never read.
"""

import json
from pathlib import Path

import numpy as np

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
PROMPTS = ("a CT scan of the head showing intracranial hemorrhage", "a CT scan of the head with no intracranial hemorrhage")
TRAIN_SAMPLE = 60000
SUBTYPES = ("epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural")
SEED = 20261120
TARGET_SENSITIVITY = 0.96


def load_biomedclip(device):
    import open_clip
    model, preprocess = open_clip.create_model_from_pretrained(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    return model.to(device).eval(), preprocess, tokenizer


def _normaliser(preprocess):
    """(size, mean, std) from BiomedCLIP's own torchvision preprocessing."""
    size, mean, std = 224, None, None
    for tr in getattr(preprocess, "transforms", []):
        if type(tr).__name__ == "Normalize":
            mean, std = tuple(tr.mean), tuple(tr.std)
        if type(tr).__name__ in ("Resize", "CenterCrop"):
            s = tr.size
            size = s if isinstance(s, int) else s[0]
    if mean is None:
        raise ValueError("BiomedCLIP's preprocessing has no Normalize step")
    return size, mean, std


def to_input(images, preprocess, device):
    """uint8 N x H x W x 3 (array or tensor) -> BiomedCLIP's normalised input, resized on the device (bicubic)."""
    import torch
    import torch.nn.functional as F
    size, mean, std = _normaliser(preprocess)
    x = torch.as_tensor(np.ascontiguousarray(images)).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


def embed_images(model, preprocess, images, device, batch=128):
    """uint8 N x H x W x 3 -> normalised N x 512 float32 embeddings."""
    import torch
    out = []
    for i in range(0, len(images), batch):
        with torch.no_grad():
            f = model.encode_image(to_input(images[i:i + batch], preprocess, device)).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def biomedclip_pass(model, preprocess, dataset, device, batch=128, workers=4):
    """Embeddings for every item of a `train.SliceDataset`, in its order (Kaggle)."""
    import torch

    from strokecopilot.vision.haemorrhage import train
    out = []
    for images, _ in train.loader(dataset, batch_size=batch, workers=workers):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.encode_image(to_input(images, preprocess, device)).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def exp6_pass(net, dataset, device, batch=128, workers=4):
    """EXP-6's slice probabilities and frozen pooled features (2,048-d) in one pass (Kaggle)."""
    import torch

    from strokecopilot.vision.haemorrhage import model as hmodel
    from strokecopilot.vision.haemorrhage import train
    net.eval()
    probs, feats = [], []
    for images, _ in train.loader(dataset, batch_size=batch, workers=workers):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = net.forward_head(net.forward_features(hmodel.normalise(images, device)), pre_logits=True).float()
            logit = net.get_classifier()(f).float().squeeze(1)
        feats.append(f.cpu().numpy())
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs), np.concatenate(feats)


def _tokenize(tokenizer, texts):
    """open_clip's tokenizer, or, where transformers 5 removed the call it uses, the same padding done directly."""
    try:
        return tokenizer(texts)
    except AttributeError:
        out = tokenizer.tokenizer(texts, padding="max_length", truncation=True, max_length=tokenizer.context_length,
                                  return_tensors="pt")
        return out["input_ids"]


def zero_shot(model, tokenizer, embeddings, device):
    """P(bleed) per slice from the two prompts."""
    import torch
    with torch.no_grad():
        t = model.encode_text(_tokenize(tokenizer, list(PROMPTS)).to(device)).float()
        t = torch.nn.functional.normalize(t, dim=-1)
        logits = model.logit_scale.exp().float() * torch.from_numpy(embeddings).to(device) @ t.T
    return torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()


def fit_probe(x, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(x, y)


def probe_json(clf):
    return {"coef": clf.coef_.ravel().tolist(), "intercept": float(clf.intercept_[0])}


def probe_predict(params, x):
    z = np.asarray(x) @ np.asarray(params["coef"]) + params["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def scan_table(slices, prob, k=2):
    """One row per scan: patient, label, subtype labels, top-k mean score. slices: series_id, patient_id, any, subtypes."""
    df = slices.assign(_p=prob)
    g = df.groupby("series_id")
    out = g.agg(patient_id=("patient_id", "first"), label=("any", "max"),
                **{s: (s, "max") for s in SUBTYPES if s in df})
    out["score"] = g["_p"].apply(lambda p: float(np.sort(p.to_numpy())[-k:].mean()))
    return out.reset_index(drop=True)


def platt(score, label):
    """(a, b) of p = sigmoid(a * logit(score) + b), fitted on these scans."""
    from sklearn.linear_model import LogisticRegression
    s = np.clip(np.asarray(score, float), 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(s / (1 - s)).reshape(-1, 1), np.asarray(label))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(ab, score):
    s = np.clip(np.asarray(score, float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(ab[0] * np.log(s / (1 - s)) + ab[1])))


def operating_point(prob, label, target=TARGET_SENSITIVITY):
    """The highest threshold reaching `target` sensitivity, and sensitivity and specificity there."""
    prob, label = np.asarray(prob, float), np.asarray(label).astype(bool)
    pos = np.sort(prob[label])[::-1]
    t = float(pos[int(np.ceil(target * len(pos))) - 1])
    return {"threshold": t, "sensitivity": float((prob[label] >= t).mean()), "specificity": float((prob[~label] < t).mean())}


def evaluate_scans(scans, prob_col):
    from strokecopilot.vision.haemorrhage.aggregate import roc_auc
    return {"scan_auc": roc_auc(scans["label"], scans[prob_col]), **operating_point(scans[prob_col], scans["label"])}


def per_patient_brier(scans, prob_col):
    """Mean Brier score per patient (scans averaged), in patient order, as a plain list (no ID)."""
    b = (scans[prob_col] - scans["label"]) ** 2
    return scans.assign(_b=b).groupby("patient_id")["_b"].mean().sort_index().tolist()


def subtype_report(val_slices, probs_by_subtype, any_prob):
    """Slice and scan AUC per subtype for the subtype probes and for EXP-6's 'any' score as that subtype's detector."""
    from strokecopilot.vision.haemorrhage.aggregate import roc_auc
    out = {}
    for s in SUBTYPES:
        if s not in val_slices or val_slices[s].sum() == 0:
            continue
        y = val_slices[s].to_numpy().astype(bool)
        probe_scans = scan_table(val_slices, probs_by_subtype[s])
        any_scans = scan_table(val_slices, any_prob)
        out[s] = {"slices_positive": int(y.sum()), "scans_positive": int(probe_scans[s].sum()),
                  "probe_slice_auc": roc_auc(y, probs_by_subtype[s]), "exp6_any_slice_auc": roc_auc(y, any_prob),
                  "probe_scan_auc": roc_auc(probe_scans[s], probe_scans["score"]),
                  "exp6_any_scan_auc": roc_auc(any_scans[s], any_scans["score"])}
    return out


def isles_main(argv=None):
    """The ISLES'24 half on the Mac (EXP-83): BiomedCLIP probe and zero-shot flags on the 129 development scans."""
    import argparse
    import hashlib

    import torch

    from strokecopilot.data import isles24, splits
    from strokecopilot.vision.haemorrhage import isles_check as ic
    from strokecopilot.vision.haemorrhage import isles_false_alarms as fa

    ap = argparse.ArgumentParser(description="Task D, ISLES'24 half")
    ap.add_argument("--probe", required=True, help="probe.json from the Kaggle run (EXP-82)")
    ap.add_argument("--out", default="runs/EXP-83_biomedclip_isles")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    probe = json.loads(Path(args.probe).read_text())
    if hashlib.sha256(splits.FOLDS_PATH.read_bytes()).hexdigest() != fa.FOLDS_SHA256:
        raise ValueError("The installed split is not the fixed one. Stop and tell Parsa.")
    folds = splits.load_folds()
    root = Path(__file__).resolve().parents[3] / "data" / "local" / isles24.SLIM_NAME
    cases = isles24.list_cases(root, folds)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, preprocess, tokenizer = load_biomedclip(device)
    exp6 = {r["case"]: r["none_called"] in ("True", "true", "1") for r in
            __import__("csv").DictReader(open(Path(fa.DRIVE) / "runs" / "EXP-50_visionB_isles_false_alarms" / "per_case.csv"))}
    flags = {"probe": [], "zero_shot": [], "exp6": []}
    for case in cases:
        images, _ = ic.prepare(isles24.load_native_ncct(case, root, folds))
        e = embed_images(model, preprocess, images, device)
        for name, p in (("probe", probe_predict(probe["probe"]["probe"], e)), ("zero_shot", zero_shot(model, tokenizer, e, device))):
            score = float(np.sort(p)[-2:].mean())
            prob = float(apply_platt(probe[name]["platt"], [score])[0])
            flags[name].append(prob >= probe[name]["threshold"])
        flags["exp6"].append(exp6[case])
    from strokecopilot.vision.haemorrhage.isles_false_alarms import mcnemar
    result = {"experiment": "EXP-83", "scans": len(cases), "flagged": {k: int(sum(v)) for k, v in flags.items()},
              "vs_exp6": {k: mcnemar(np.array(flags["exp6"]), np.array(flags[k])) for k in ("probe", "zero_shot")}}
    text = json.dumps(result, indent=1, default=str)
    if ic.ids_in(text, cases + list(folds["heldout"])):
        raise ValueError("A case ID reached metrics.json. Nothing was saved.")
    (out / "metrics.json").write_text(text)
    print(text)


if __name__ == "__main__":
    isles_main()
