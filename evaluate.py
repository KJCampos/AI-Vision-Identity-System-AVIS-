"""
evaluate.py - measure how accurate AVIS really is, on clips you labeled yourself.

Folder layout (record with Run Task > "AVIS: Record Test Clip", or record_clip.py):
    eval/<EnrolledName>/   clips of that enrolled person       -> should be recognized
    eval/unknown/          clips of people NOT enrolled        -> should be "Unknown"
    eval/spoof/            photos/phone screens of an enrolled -> should be blocked as spoof
                           person held up to the camera

Reports:
  False Accept Rate (FAR)  - strangers wrongly called an enrolled person   (security risk)
  False Reject Rate (FRR)  - enrolled people wrongly called Unknown        (annoyance)
  EER                      - where FAR = FRR (single-number summary)
  Spoof: APCER (spoof let through as live) / BPCER (real face blocked)
and recommends thresholds from YOUR data.  --apply writes them into config.yaml.

    python evaluate.py
    python evaluate.py --target-far 0.001 --apply
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SAFE_MIN_MATCH = 0.363   # SFace's published cosine threshold; evaluate.py only ever tightens it
from avis.config import load_config, set_config_value  # noqa: E402
from avis.facepipe import FacePipe, iter_media  # noqa: E402
from avis.learning import FaceDB  # noqa: E402


def rates(genuine, impostor, t):
    far = float(np.mean(impostor >= t)) if len(impostor) else float("nan")
    frr = float(np.mean(genuine < t)) if len(genuine) else float("nan")
    return far, frr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=os.path.join(HERE, "eval"))
    ap.add_argument("--faces-db", default=os.path.join(HERE, "faces_db"))
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--step", type=int, default=3, help="use every Nth video frame")
    ap.add_argument("--target-far", type=float, default=0.01, help="max acceptable false-accept rate (1%% default)")
    ap.add_argument("--apply", action="store_true", help="write recommended thresholds to config.yaml")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cur_t = float(cfg.get("match_thresh", 0.363))
    cur_live = float(cfg.get("liveness_thresh", 0.6))
    db = FaceDB(a.faces_db)
    if not db.people:
        sys.exit("No enrolled people yet. Enroll first (Run Task > AVIS: Enroll My Face).")
    if not os.path.isdir(a.eval_dir):
        sys.exit("No eval/ folder yet. Record labeled clips first (Run Task > AVIS: Record Test Clip).")
    lower = {n.lower(): n for n in db.people}

    pipe = FacePipe(liveness=True)
    genuine, impostor, live_real, live_spoof = [], [], [], []
    correct_id, total_id, spoof_rows = 0, 0, []
    for folder in sorted(os.listdir(a.eval_dir)):
        path = os.path.join(a.eval_dir, folder)
        if not os.path.isdir(path):
            continue
        kind = folder.lower()
        true = None if kind in ("unknown", "spoof") else lower.get(kind)
        if kind not in ("unknown", "spoof") and true is None:
            print(f"skip eval/{folder}: not an enrolled name ({', '.join(db.people)})")
            continue
        n = 0
        for _, frame in iter_media(path, a.step):
            r = pipe.largest_face(frame)
            if r is None:
                continue
            emb, live, _ = r
            n += 1
            sims = {name: float(np.max(arr @ emb)) for name, arr in db.people.items()}
            best = max(sims, key=sims.get)
            if kind == "spoof":
                live_spoof.append(live)
                spoof_rows.append((sims[best], live))
            else:
                live_real.append(live)
                if true:
                    genuine.append(sims[true])
                    impostor += [s for nm, s in sims.items() if nm != true]   # other enrolled people
                    total_id += 1
                    correct_id += int(best == true and sims[best] >= cur_t)
                else:
                    impostor.append(sims[best])
        print(f"eval/{folder}: {n} face frames")

    g, im = np.array(genuine), np.array(impostor)
    lr, ls = np.array([x for x in live_real if x is not None]), np.array([x for x in live_spoof if x is not None])
    ts = np.linspace(0, 1, 401)
    lines = ["AVIS accuracy report", "=" * 40]
    lines.append(f"Samples: {len(g)} genuine, {len(im)} impostor comparisons, {len(lr)} real-face / {len(ls)} spoof liveness checks")

    rec_t = rec_live = None
    if len(g):
        far, frr = rates(g, im, cur_t)
        lines.append(f"\nAt current match_thresh {cur_t:.3f}:")
        lines.append(f"  correct identification: {100 * correct_id / max(1, total_id):.1f}% of enrolled-person frames")
        lines.append(f"  False Reject Rate (you -> Unknown):        {100 * frr:.2f}%")
        if len(im):
            lines.append(f"  False Accept Rate (stranger -> enrolled):  {100 * far:.2f}%")
            fars = np.array([rates(g, im, t)[0] for t in ts]); frrs = np.array([rates(g, im, t)[1] for t in ts])
            e = int(np.argmin(np.abs(fars - frrs)))
            lines.append(f"  Equal Error Rate: {100 * (fars[e] + frrs[e]) / 2:.2f}% at threshold {ts[e]:.3f}")
            ok = np.where(fars <= a.target_far)[0]
            need = int(np.ceil(3 / a.target_far))          # rule of three: ~3/FAR impostor samples to trust it
            if len(ok):
                cand = float(ts[ok[0]])
                safe = max(cand, SAFE_MIN_MATCH)           # never loosen below the model's validated default
                lines.append(f"  Threshold reaching FAR <= {100 * a.target_far:g}% on this data: {cand:.3f} "
                             f"(FRR there: {100 * frrs[ok[0]]:.2f}%)")
                if len(im) < need:
                    lines.append(f"  Not enough stranger data to trust a {100 * a.target_far:g}% FAR: have {len(im)}, "
                                 f"need ~{need}. Record more eval/unknown clips. (Not applying.)")
                else:
                    rec_t = safe
                    lines.append(f"  Recommended match_thresh: {rec_t:.3f}"
                                 + (f" (kept at the safe minimum {SAFE_MIN_MATCH})" if safe > cand else ""))
        else:
            lines.append("  (no eval/unknown clips -> can't measure false accepts yet)")
    else:
        lines.append("\nNo genuine clips (eval/<YourName>/) -> can't measure recognition yet.")

    if len(ls) and len(lr):
        apcer = float(np.mean(ls >= cur_live)); bpcer = float(np.mean(lr < cur_live))
        lines.append(f"\nSpoof protection at liveness_thresh {cur_live:.2f}:")
        lines.append(f"  spoofs let through (APCER): {100 * apcer:.2f}%   real faces blocked (BPCER): {100 * bpcer:.2f}%")
        band = (ts >= 0.3) & (ts <= 0.95)                  # never recommend 'block everything' or 'allow everything'
        acer = np.array([(np.mean(ls >= t) + np.mean(lr < t)) / 2 for t in ts])
        best = float(ts[band][int(np.argmin(acer[band]))])
        cur_acer = (apcer + bpcer) / 2
        lines.append(f"  Best liveness_thresh on your data: {best:.2f} (avg error {100 * acer[band].min():.2f}% vs "
                     f"{100 * cur_acer:.2f}% now)")
        if min(len(ls), len(lr)) < 100:
            lines.append(f"  Need >= 100 frames each of real and spoof to tune this (have {len(lr)} / {len(ls)}). (Not applying.)")
        elif acer[band].min() > 0.25:
            lines.append("  Model can't separate your spoof clips well (error > 25%) - not changing the threshold."
                         " Check that eval/spoof really shows photos/screens.")
        elif cur_acer - acer[band].min() > 0.02:
            rec_live = best
        thru = sum(1 for s, lv in spoof_rows if s >= cur_t and lv is not None and lv >= cur_live)
        lines.append(f"  End-to-end: spoof frames that would be accepted as an enrolled person: {thru}/{len(spoof_rows)}")
    elif len(lr):
        lines.append(f"\nLiveness on real faces: {100 * np.mean(lr >= cur_live):.1f}% pass at {cur_live:.2f} (add eval/spoof clips to test attacks)")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    with open(os.path.join(HERE, "runs", "eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    bins = np.linspace(-0.2, 1, 61)
    if len(g):
        ax[0].hist(g, bins, alpha=0.7, color="#3182bd", label="enrolled person (genuine)")
    if len(im):
        ax[0].hist(im, bins, alpha=0.7, color="#de2d26", label="stranger / other person")
    ax[0].axvline(cur_t, color="k", ls="--", label=f"current {cur_t:.3f}")
    if rec_t is not None:
        ax[0].axvline(rec_t, color="#31a354", ls=":", lw=2, label=f"recommended {rec_t:.3f}")
    ax[0].set(title="Match similarity", xlabel="cosine similarity", ylabel="frames"); ax[0].legend(fontsize=8)
    if len(g) and len(im):
        ax[1].plot(ts, [100 * rates(g, im, t)[0] for t in ts], color="#de2d26", label="False Accept %")
        ax[1].plot(ts, [100 * rates(g, im, t)[1] for t in ts], color="#3182bd", label="False Reject %")
        ax[1].axvline(cur_t, color="k", ls="--")
        ax[1].legend(fontsize=8)
    else:
        ax[1].text(0.5, 0.5, "Needs eval/<name> AND eval/unknown clips", ha="center", va="center", transform=ax[1].transAxes)
    ax[1].set(title="Error rates vs threshold", xlabel="match threshold", ylabel="%")
    lb = np.linspace(0, 1, 41)
    if len(lr):
        ax[2].hist(lr, lb, alpha=0.7, color="#31a354", label="real faces")
    if len(ls):
        ax[2].hist(ls, lb, alpha=0.7, color="#756bb1", label="spoofs (photo/screen)")
    ax[2].axvline(cur_live, color="k", ls="--", label=f"current {cur_live:.2f}")
    ax[2].set(title="Liveness score", xlabel="real-face probability"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(HERE, "runs", "eval_report.png")
    fig.savefig(png, dpi=110)
    print("\nReport -> runs/eval_report.txt, graphs -> runs/eval_report.png")

    if a.apply:
        if rec_t is None and rec_live is None:
            print("Nothing applied: not enough reliable data yet (see notes above).")
        if rec_t is not None:
            set_config_value(a.config, "match_thresh", f"{rec_t:.3f}")
            print(f"config.yaml: match_thresh = {rec_t:.3f}")
        if rec_live is not None:
            set_config_value(a.config, "liveness_thresh", f"{rec_live:.2f}")
            print(f"config.yaml: liveness_thresh = {rec_live:.2f}")


if __name__ == "__main__":
    main()
