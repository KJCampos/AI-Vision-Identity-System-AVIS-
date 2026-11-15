"""
Face profiles that learn over time + an optional GPU-trained identity model.

FaceDB (faces_db/<name>.npy = N x 128 SFace embeddings):
  * match()        best cosine similarity against every stored sample
  * maybe_learn()  adds a new sample of an already-recognized person when it is
                   very confident (high similarity + passed the spoof check), not a
                   near-duplicate of what's stored, and not too often. Over time the
                   profile covers new lighting, angles, glasses, haircuts, etc.
                   Enrollment samples are never overwritten; once a profile is full,
                   a new learned sample replaces the most redundant learned one.
  * forget()       deletes a person completely ("forget me").

IdentityModel: small PyTorch network trained by train_identity.py on these profiles
(on your GPU if available). At runtime it has to AGREE with the similarity match
before a face is called a known person, which cuts down false accepts.
"""
import json
import logging
import os
import time

import numpy as np


class FaceDB:
    def __init__(self, folder, max_samples=300):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.max_samples = int(max_samples)
        self.people = {}
        self.meta = {}
        self._dirty = set()
        self._last_learn = {}
        self.learned_this_session = {}
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".npy") and not f.startswith("_"):
                name = f[:-4]
                try:
                    arr = np.load(os.path.join(folder, f)).astype(np.float32)
                    arr = arr[None, :] if arr.ndim == 1 else arr
                    self.people[name] = arr
                    self.meta[name] = self._load_meta(name, len(arr))
                except Exception as e:
                    logging.warning("[FACEDB] could not load %s: %s", f, e)

    def _meta_path(self, name):
        return os.path.join(self.folder, f"{name}.json")

    def _load_meta(self, name, n):
        try:
            with open(self._meta_path(name), encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            m = {}
        m.setdefault("enrolled", n)     # older profiles: everything counts as enrolled
        m.setdefault("learned", 0)
        return m

    def _save(self, name):
        np.save(os.path.join(self.folder, f"{name}.npy"), self.people[name])
        with open(self._meta_path(name), "w", encoding="utf-8") as f:
            json.dump(self.meta[name], f)

    def add(self, name, embs):
        arr = np.stack(embs).astype(np.float32)
        if name in self.people:
            self.people[name] = np.concatenate([self.people[name], arr])
            self.meta[name]["enrolled"] += len(arr)
        else:
            self.people[name] = arr
            self.meta[name] = {"enrolled": len(arr), "learned": 0}
        self._save(name)

    def match(self, emb, thresh):
        """-> (name if above threshold else None, best similarity, closest name)."""
        best_name, best_sim = None, -1.0
        for name, arr in self.people.items():
            sim = float(np.max(arr @ emb))
            if sim > best_sim:
                best_name, best_sim = name, sim
        return (best_name if best_sim >= thresh else None), best_sim, best_name

    def maybe_learn(self, name, emb, sim, min_sim=0.55, min_gap_s=2.0, max_redundancy=0.93, now=None):
        now = time.time() if now is None else now
        if name not in self.people or sim < min_sim:
            return False
        if now - self._last_learn.get(name, 0) < min_gap_s:
            return False
        arr = self.people[name]
        sims = arr @ emb
        if float(sims.max()) > max_redundancy:
            return False                              # already have this look
        self._last_learn[name] = now
        protected = int(self.meta[name]["enrolled"])
        if len(arr) < self.max_samples or protected >= len(arr):
            self.people[name] = np.concatenate([arr, emb[None, :].astype(np.float32)])
        else:
            i = protected + int(np.argmax(sims[protected:]))   # most redundant learned sample
            arr[i] = emb
        self.meta[name]["learned"] += 1
        self.learned_this_session[name] = self.learned_this_session.get(name, 0) + 1
        self._dirty.add(name)
        return True

    def save(self):
        for name in list(self._dirty):
            self._save(name)
        self._dirty.clear()

    def forget(self, name):
        self.people.pop(name, None)
        self.meta.pop(name, None)
        for ext in (".npy", ".json"):
            p = os.path.join(self.folder, name + ext)
            if os.path.exists(p):
                os.remove(p)


# ---------------- identity model (PyTorch) ----------------
def build_mlp(in_dim, n_classes, hidden=256):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(hidden // 2, n_classes),
    )


class IdentityModel:
    UNKNOWN = "Unknown"

    def __init__(self, path):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.classes = list(ckpt["classes"])
        self.net = build_mlp(int(ckpt["in_dim"]), len(self.classes), int(ckpt["hidden"])).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        logging.info("[ML] identity model loaded (%s, classes=%s, val acc %.1f%%)",
                     self.device, self.classes, 100 * float(ckpt.get("val_acc", 0)))

    def predict(self, emb):
        with self.torch.no_grad():
            x = self.torch.from_numpy(emb.astype(np.float32))[None].to(self.device)
            p = self.torch.softmax(self.net(x), dim=1)[0].cpu().numpy()
        k = int(np.argmax(p))
        return self.classes[k], float(p[k])
