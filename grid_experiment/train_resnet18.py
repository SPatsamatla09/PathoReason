"""Sourced ResNet-18 baseline for MHIST (Kiran's request; replaces the unsourced ResNet/ViT numbers).

ResNet-18 written in plain PyTorch (torchvision is not installed) and trained FROM SCRATCH: no pretrained
weights are downloaded. Train set: MHIST train split minus the competence dev set (runs/competence/splits.json).
Epochs are chosen on dev (best dev AUC); 3 seeds; the test split is evaluated exactly once per seed, at the
chosen epoch. Resumable: a checkpoint and log line are written after every epoch.

    python3 train_resnet18.py --seed 0 [--epochs 30]   ->  runs/resnet18/seed{S}/{log.jsonl, best.pt, test.json}
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "..", "images")
OUT = os.path.join(ROOT, "runs", "resnet18")
# Checkpoints live OUTSIDE ~/Documents: iCloud "Optimize Mac Storage" evicted/renamed checkpoints that were
# rewritten every epoch (seed 0 lost best.pt -> 'best 9.pt'). Logs and test.json stay in runs/.
CKPT = os.path.expanduser("~/Library/Caches/mhist_resnet18")


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class ResNet18(nn.Module):
    """Standard ResNet-18 (He et al. 2016): 7x7 stem, 4 stages of 2 basic blocks, 64-128-256-512."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [Block(cin, cout, stride), Block(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def load_split():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "..", "annotations.csv"))))
    lab = {r["Image Name"]: int(r["Majority Vote Label"] == "SSA") for r in rows}
    part = {r["Image Name"]: r["Partition"] for r in rows}
    dev = set(json.load(open(os.path.join(ROOT, "runs", "competence", "splits.json")))["dev"])
    train = sorted(n for n in lab if part[n] == "train" and n not in dev)
    test = sorted(n for n in lab if part[n] == "test")
    return lab, train, sorted(dev), test


def load_images(names):
    return np.stack([np.asarray(Image.open(os.path.join(IMG, n)).convert("RGB"), dtype=np.uint8) for n in names])


def to_tensor(x, mean, std):
    return (torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0 - mean) / std


def augment(batch, rng):
    """Histology is orientation-free: random 90-degree rotations and flips."""
    out = []
    for im in batch:
        im = np.rot90(im, rng.randrange(4))
        if rng.random() < 0.5:
            im = im[:, ::-1]
        out.append(np.ascontiguousarray(im))
    return np.stack(out)


def auc(scores, y):
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(r for r, t in zip(ranks, y) if t)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def evaluate(model, x, y, mean, std, bs=64):
    model.eval()
    p = []
    for i in range(0, len(x), bs):
        p += torch.softmax(model(to_tensor(x[i:i + bs], mean, std)), 1)[:, 1].tolist()
    pred = [int(v >= 0.5) for v in p]
    acc = sum(a == b for a, b in zip(pred, y)) / len(y)
    r1 = sum(a == 1 and b == 1 for a, b in zip(pred, y)) / max(sum(y), 1)
    r0 = sum(a == 0 and b == 0 for a, b in zip(pred, y)) / max(len(y) - sum(y), 1)
    return {"accuracy": round(acc, 4), "auc": round(auc(p, y), 4), "balanced_accuracy": round((r0 + r1) / 2, 4),
            "recall_HP": round(r0, 4), "recall_SSA": round(r1, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    d = os.path.join(OUT, f"seed{args.seed}")
    cd = os.path.join(CKPT, f"seed{args.seed}")
    os.makedirs(d, exist_ok=True)
    os.makedirs(cd, exist_ok=True)
    if os.path.exists(os.path.join(d, "test.json")):
        print("done already:", json.load(open(os.path.join(d, "test.json"))))
        return
    lab, train, dev, test = load_split()
    xtr, xdv = load_images(train), load_images(dev)
    ytr, ydv = [lab[n] for n in train], [lab[n] for n in dev]
    mean = torch.tensor(xtr.reshape(-1, 3).mean(0) / 255.0).view(1, 3, 1, 1).float()
    std = torch.tensor(xtr.reshape(-1, 3).std(0) / 255.0).view(1, 3, 1, 1).float()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = ResNet18()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * steps_per_epoch)
    # class-weighted loss: the train split is ~72% HP
    w = torch.tensor([1.0 / (len(ytr) - sum(ytr)), 1.0 / sum(ytr)])
    w = w / w.sum() * 2
    logp = os.path.join(d, "log.jsonl")
    start, best = 0, {"auc": -1}
    ck = os.path.join(cd, "last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, best = s["epoch"] + 1, s["best"]
        rng.setstate(s["rng"]); torch.set_rng_state(s["torch_rng"])
    for ep in range(start, args.epochs):
        model.train()
        t0, order, tot = time.time(), list(range(len(train))), 0.0
        rng.shuffle(order)
        for i in range(0, len(order), args.bs):
            idx = order[i:i + args.bs]
            xb = to_tensor(augment(xtr[idx], rng), mean, std)
            loss = F.cross_entropy(model(xb), torch.tensor([ytr[j] for j in idx]), weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        m = evaluate(model, xdv, ydv, mean, std)
        rec = {"epoch": ep, "train_loss": round(tot / len(train), 4), "dev": m, "seconds": round(time.time() - t0, 1)}
        open(logp, "a").write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if m["auc"] > best["auc"]:
            best = {**m, "epoch": ep}
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "epoch": ep}, os.path.join(cd, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "rng": rng.getstate(), "torch_rng": torch.get_rng_state()}, ck)
    # the one test evaluation for this seed, at the dev-chosen epoch
    b = torch.load(os.path.join(cd, "best.pt"), weights_only=False)
    model.load_state_dict(b["model"])
    xte = load_images(test)
    res = {"seed": args.seed, "chosen_epoch": b["epoch"], "dev_at_chosen_epoch": best,
           "test": evaluate(model, xte, [lab[n] for n in test], b["mean"], b["std"]), "n_test": len(test),
           "n_train": len(train), "n_dev": len(dev), "pretrained": False}
    json.dump(res, open(os.path.join(d, "test.json"), "w"), indent=2)
    print("TEST", json.dumps(res))


if __name__ == "__main__":
    main()
