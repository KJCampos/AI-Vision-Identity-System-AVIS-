"""
train_identity.py - train AVIS's identity network on your GPU.

What it learns: to tell enrolled people apart (and from "Unknown", if you give it
examples of people who aren't enrolled). At runtime it must AGREE with the
similarity match before AVIS calls someone by name, which cuts false accepts.

Training data (all consented):
  * faces_db/<name>.npy                every enrolled person (grows as AVIS learns)
  * eval/unknown/*.mp4|jpg             clips of people who are NOT enrolled (record_clip.py unknown)
  * eval/<name>/*.mp4|jpg              extra clips of enrolled people (record_clip.py <name>)

    python train_identity.py                 # trains on GPU if available
    python train_identity.py --epochs 300

Output: models/identity/identity_mlp.pt, runs/identity_training.png
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avis.learning import FaceDB, build_mlp  # noqa: E402


def load_data(faces_db, eval_dir, step):
    db = FaceDB(faces_db)
    X, y, classes = [], [], sorted(db.people)
    for i, name in enumerate(classes):
        X.append(db.people[name]); y += [i] * len(db.people[name])
    sources = {c: len(db.people[c]) for c in classes}

    extra = {}
    from avis.facepipe import FacePipe, iter_media, iter_photos
    pipe = FacePipe(liveness=False)

    # face photos AVIS saved per run: dataset/<name>/<date time run>/*.jpg
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(faces_db)), "dataset")
    for name in classes:
        folder = os.path.join(dataset_dir, name)
        if not os.path.isdir(folder):
            continue
        have = db.people[name]
        new = []
        for _, img in iter_photos(folder):
            e = pipe.embed_aligned(img)
            if e is None:
                continue
            if float(np.max(have @ e)) > 0.995 or (new and float(np.max(np.stack(new) @ e)) > 0.995):
                continue                       # already in the profile / duplicate
            new.append(e)
        if new:
            extra.setdefault(name, []).extend(new)
            print(f"  + {len(new)} photos of {name.replace('_', ' ')} from dataset/{name}/")

    if os.path.isdir(eval_dir):
        for folder in sorted(os.listdir(eval_dir)):
            path = os.path.join(eval_dir, folder)
            if not os.path.isdir(path) or folder == "spoof":
                continue
            label = "Unknown" if folder.lower() == "unknown" else folder
            if label != "Unknown" and label not in classes:
                print(f"  skip eval/{folder}: '{folder}' is not enrolled")
                continue
            embs = [r[0] for _, fr in iter_media(path, step) for r in [pipe.largest_face(fr)] if r is not None]
            if embs:
                extra.setdefault(label, []).extend(embs)
    neg_file = os.path.join(faces_db, "_negatives.npy")
    if os.path.exists(neg_file):
        extra.setdefault("Unknown", []).extend(list(np.load(neg_file)))

    if "Unknown" in extra:
        classes.append("Unknown")
    for label, embs in extra.items():
        idx = classes.index(label)
        X.append(np.stack(embs).astype(np.float32)); y += [idx] * len(embs)
        sources[label] = sources.get(label, 0) + len(embs)
    if not X:
        return None, None, classes, sources
    return np.concatenate(X).astype(np.float32), np.array(y), classes, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces-db", default=os.path.join(HERE, "faces_db"))
    ap.add_argument("--eval-dir", default=os.path.join(HERE, "eval"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "identity", "identity_mlp.pt"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--step", type=int, default=3, help="use every Nth video frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0) + ' (CUDA ' + torch.version.cuda + ')' if dev == 'cuda' else 'CPU'}")

    X, y, classes, sources = load_data(a.faces_db, a.eval_dir, a.step)
    print("Training data:", ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in sources.items()) or "none")
    if X is None or len(classes) < 2:
        sys.exit("\nNeed at least 2 classes to learn from. Either enroll a second person, or record a clip of\n"
                 "someone who is NOT enrolled (with their OK):  Run Task > 'AVIS: Record Test Clip' > unknown\n")

    # stratified 80/20 split
    tr, va = [], []
    for c in range(len(classes)):
        idx = np.random.permutation(np.where(y == c)[0])
        k = max(1, int(round(len(idx) * 0.2))) if len(idx) > 1 else 0
        va += list(idx[:k]); tr += list(idx[k:])
    Xtr, ytr = torch.tensor(X[tr], device=dev), torch.tensor(y[tr], device=dev)
    Xva, yva = torch.tensor(X[va], device=dev), torch.tensor(y[va], device=dev)

    counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), device=dev)
    net = build_mlp(X.shape[1], len(classes), a.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    hist = {"loss": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb = Xtr[b] + torch.randn_like(Xtr[b]) * 0.02          # augmentation: small embedding noise
            xb = xb / xb.norm(dim=1, keepdim=True)
            if len(b) < 2:
                continue
            loss = loss_fn(net(xb), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(Xva).argmax(1) == yva).float().mean()) if len(Xva) else 1.0
        hist["loss"].append(loss.item()); hist["val_acc"].append(acc)
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    secs = time.time() - t0
    net.load_state_dict(best_state); net.eval()

    with torch.no_grad():
        pred = net(Xva).argmax(1).cpu().numpy() if len(Xva) else np.array([])
    yv = yva.cpu().numpy()
    print(f"\nTrained {a.epochs} epochs in {secs:.1f}s on {dev.upper()} | {len(tr)} train / {len(va)} validation samples")
    print(f"Validation accuracy: {100 * best_acc:.1f}%")
    if len(yv):
        print("Confusion (rows = true, cols = predicted):")
        print("  " + " ".join(f"{c[:10]:>10}" for c in classes))
        for i, c in enumerate(classes):
            row = [int(((yv == i) & (pred == j)).sum()) for j in range(len(classes))]
            print(f"  {c[:10]:>10} " + " ".join(f"{v:>10}" for v in row))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "in_dim": int(X.shape[1]), "hidden": a.hidden,
                "val_acc": float(best_acc), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "samples": {k: int(v) for k, v in sources.items()}}, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)          # atomic: AVIS never loads a half-written model
    print(f"Saved model -> {a.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"], color="#08519c"); ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot(np.array(hist["val_acc"]) * 100, color="#31a354"); ax[1].set(title="Validation accuracy (%)", xlabel="epoch", ylim=(0, 101))
    fig.suptitle(f"AVIS identity model - {', '.join(c.replace('_', ' ') for c in classes)} ({dev.upper()}, {secs:.1f}s)")
    fig.tight_layout()
    png = os.path.join(HERE, "runs", "identity_training.png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=110)
    print(f"Training graph -> {png}")


if __name__ == "__main__":
    main()
