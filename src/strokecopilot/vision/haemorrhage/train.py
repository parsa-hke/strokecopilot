"""Vision B training on RSNA shards: balanced epochs, augmentation, checkpoints and resume (plan §8.2).

An epoch is every training slice with a bleed plus as many slices without one, taken in turn from
a fixed shuffle of the negatives, so all negatives are used across epochs. Only training is
balanced; validation stays at the natural rate. Augmentation is a horizontal flip, rotation up to
10 degrees, scaling by up to 10% and brightness and contrast by up to 10%.

Checkpoints hold weights and optimiser state only, no IDs or images. `last.pt` resumes a run and
`best.pt` keeps the weights with the best validation slice AUC. Logged lines give numbers only.
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from strokecopilot.vision.haemorrhage import model as hmodel
from strokecopilot.vision.haemorrhage.aggregate import roc_auc

ROTATE_DEG = 10.0
SCALE = 0.10
BRIGHTNESS = 0.10


@dataclass
class TrainConfig:
    model: str = "resnet50"
    pretrained: bool = True
    epochs: int = 8
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    negatives_per_positive: float = 1.0
    workers: int = 4
    amp: bool = True
    seed: int = 20261103
    max_steps_per_epoch: int | None = None  # dummy runs only
    log_every: int = 200


def augment(image, rng):
    """Flip, rotate, scale, then brightness and contrast, on a uint8 H x W x 3 slice."""
    if rng.random() < 0.5:
        image = image[:, ::-1]
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-ROTATE_DEG, ROTATE_DEG), 1 + rng.uniform(-SCALE, SCALE))
    image = cv2.warpAffine(np.ascontiguousarray(image), matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    gain = 1 + rng.uniform(-BRIGHTNESS, BRIGHTNESS)
    shift = 255 * rng.uniform(-BRIGHTNESS, BRIGHTNESS) / 2
    return np.clip(image.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)


class SliceDataset(Dataset):
    """Slices from shard readers. items needs shard, image_id and any. Yields (uint8 image, label)."""

    def __init__(self, items, readers, train=False):
        self.shards = items["shard"].to_numpy()
        self.ids = items["image_id"].to_numpy()
        self.labels = items["any"].to_numpy().astype(np.float32)
        self.readers = readers
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image = self.readers[int(self.shards[i])].read(self.ids[i])
        if self.train:
            image = augment(image, np.random.default_rng([torch.initial_seed() % 2 ** 32, i]))
        return torch.from_numpy(np.ascontiguousarray(image)), torch.tensor(self.labels[i])


def balanced_epoch(labels, epoch, seed, negatives_per_positive=1.0):
    """Indices for one epoch: every positive and the next block of a fixed shuffle of negatives."""
    labels = np.asarray(labels).astype(bool)
    positives, negatives = np.flatnonzero(labels), np.flatnonzero(~labels)
    order = np.random.default_rng(seed).permutation(negatives)
    n = min(len(negatives), int(round(negatives_per_positive * len(positives))))
    taken = np.take(order, np.arange(epoch * n, epoch * n + n), mode="wrap") if n else order[:0]
    return np.random.default_rng([seed, epoch]).permutation(np.concatenate([positives, taken]))


def _init_worker(_):
    cv2.setNumThreads(1)  # one decoding thread per worker; the workers are the parallelism
    torch.set_num_threads(1)


def loader(dataset, indices=None, batch_size=64, workers=4):
    sampler = list(indices) if indices is not None else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=indices is not None, prefetch_factor=4 if workers else None, worker_init_fn=_init_worker if workers else None)


def _schedule(optimizer, warmup, total):
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda")


@torch.no_grad()
def predict(net, dataset, device, batch_size=128, workers=4, amp=True):
    """Slice probabilities in dataset order."""
    net.eval()
    out = []
    for images, _ in loader(dataset, batch_size=batch_size, workers=workers):
        with _autocast(device, amp):
            out.append(torch.sigmoid(net(hmodel.normalise(images, device)).float().squeeze(1)).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)


def _loss(probs, labels):
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit(config, train_items, val_items, readers, out_dir, device, log=print, time_budget_s=None):
    """Train, validating each epoch, and resume from out_dir/last.pt if it is there. Returns the history.

    With time_budget_s, no epoch starts unless the slowest one so far would still finish in time, so a
    Kaggle session ends with a checkpoint instead of being cut off. Rerun with last.pt to go on.
    """
    session_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    train_set = SliceDataset(train_items, readers, train=True)
    val_set = SliceDataset(val_items, readers)
    steps_per_epoch = len(balanced_epoch(train_set.labels, 0, config.seed, config.negatives_per_positive)) // config.batch_size
    if config.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, config.max_steps_per_epoch)

    net = hmodel.build(config.model, config.pretrained).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = _schedule(optimizer, config.warmup_steps, config.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    criterion = torch.nn.BCEWithLogitsLoss()

    last, best = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    history, start = [], 0
    if os.path.exists(last):
        state = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, start = state["history"], state["epoch"] + 1
        log(f"Resuming after epoch {start - 1} from last.pt.")

    for epoch in range(start, config.epochs):
        slowest = max((h["train_seconds"] + h["val_seconds"] for h in history), default=0)
        if time_budget_s and epoch > start and time.perf_counter() - session_start + slowest > time_budget_s:
            log(f"Stopping before epoch {epoch}: the time budget would run out. Rerun to resume.")
            break
        net.train()
        indices = balanced_epoch(train_set.labels, epoch, config.seed, config.negatives_per_positive)[: steps_per_epoch * config.batch_size]
        started, seen, running, step = time.perf_counter(), 0, torch.zeros((), device=device), 0
        for step, (images, labels) in enumerate(loader(train_set, indices, config.batch_size, config.workers), 1):
            with _autocast(device, config.amp):
                loss = criterion(net(hmodel.normalise(images, device)).squeeze(1).float(), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            seen += len(labels)
            running += loss.detach()  # summed on the device, so no step waits for the GPU
            if config.log_every and step % config.log_every == 0:
                log(f"epoch {epoch} step {step}/{steps_per_epoch} loss {float(running) / step:.4f} {seen / (time.perf_counter() - started):.0f} slices/s")
        train_seconds = time.perf_counter() - started
        probs = predict(net, val_set, device, config.batch_size * 2, config.workers, config.amp)
        row = {"epoch": epoch, "train_loss": round(float(running) / max(step, 1), 4), "train_slices": seen,
               "train_seconds": round(train_seconds, 1), "train_slices_per_second": round(seen / max(train_seconds, 1e-6), 1),
               "val_seconds": round(time.perf_counter() - started - train_seconds, 1),
               "val_slice_auc": roc_auc(val_set.labels, probs), "val_slice_log_loss": round(_loss(probs, val_set.labels), 4),
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if row["val_slice_auc"] is not None and row["val_slice_auc"] >= max(h["val_slice_auc"] or 0 for h in history):
            torch.save({"model": net.state_dict(), "config": asdict(config), "epoch": epoch}, best)
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": history, "config": asdict(config)}, last)
        log(json.dumps(row))
    return history


def load_best(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    net = hmodel.build(state["config"]["model"], pretrained=False)
    net.load_state_dict(state["model"])
    return net.to(device).to(memory_format=torch.channels_last).eval(), state


def loader_speed(dataset, batch_size=64, workers=4, batches=50):
    """Slices per second through the data loader alone, decoding and augmenting."""
    indices = np.random.default_rng(0).permutation(len(dataset))[: batch_size * batches]
    started, seen = time.perf_counter(), 0
    for images, _ in loader(dataset, indices, batch_size, workers):
        seen += len(images)
    return {"workers": workers, "slices": seen, "slices_per_second": round(seen / max(time.perf_counter() - started, 1e-6), 1)}


def gpu_speed(name, device, batch_size=64, size=320, steps=60, warmup=10, amp=True):
    """Training slices per second on random images, so the GPU alone is measured. Weights are random."""
    net = hmodel.build(name, pretrained=False).to(device).to(memory_format=torch.channels_last).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    images = torch.randint(0, 256, (batch_size, size, size, 3), dtype=torch.uint8)
    labels = torch.randint(0, 2, (batch_size,), device=device).float()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + steps):
        if step == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
        with _autocast(device, amp):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(hmodel.normalise(images, device)).squeeze(1).float(), labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None
    return {"model": name, "params_m": hmodel.parameters_millions(net), "batch_size": batch_size,
            "slices_per_second": round(steps * batch_size / seconds, 1), "peak_gib": round(peak, 2) if peak else None}
