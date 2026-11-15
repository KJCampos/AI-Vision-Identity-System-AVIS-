"""
analyze_runs.py - clean an AVIS session log and graph it.

Usage:
    python analyze_runs.py                      # newest CSV in runs/
    python analyze_runs.py "runs/AVIS Trial 2.csv"
    python analyze_runs.py --all                # every CSV in runs/

Outputs (next to the CSV):
    <name>.png          4 graphs
    <name>_clean.csv    the cleaned rows the graphs were drawn from
and prints a summary plus a data-cleaning report.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")

INT_COLS = ["frame", "people", "faces", "tracks", "tracks_measured", "tracks_predicted", "known", "unknown"]
FLOAT_COLS = ["time_s", "frame_ms", "fps_avg", "match_thresh"]
WARMUP_S = 1.0          # first second: camera + model warm-up, not representative
MAX_FRAME_MS = 2000.0   # a single frame slower than 2 s = a stall (e.g. typing a name), not real speed


def load_clean(path):
    report = {"rows_read": 0, "bad_values": 0, "duplicates": 0, "impossible": 0, "warmup": 0, "stalls": 0}
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in INT_COLS + FLOAT_COLS + ["best_similarity", "names"] if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{os.path.basename(path)} is not an AVIS session log (missing columns: {missing}). "
                     f"Old --eval-csv files use a different format.")
        for r in reader:
            report["rows_read"] += 1
            try:
                row = {c: int(r[c]) for c in INT_COLS}
                row.update({c: float(r[c]) for c in FLOAT_COLS})
                sim = r["best_similarity"].strip()
                row["best_similarity"] = float(sim) if sim else np.nan
                row["names"] = r["names"].strip()
            except (ValueError, TypeError, AttributeError):
                report["bad_values"] += 1
                continue
            if row["frame"] in seen:
                report["duplicates"] += 1
                continue
            # sanity rules: counts can't be negative, measured+predicted = tracks,
            # known+unknown can't exceed faces seen, similarity is a cosine in [-1, 1]
            if (min(row[c] for c in INT_COLS) < 0 or row["frame_ms"] <= 0
                    or row["tracks_measured"] + row["tracks_predicted"] != row["tracks"]
                    or (not np.isnan(row["best_similarity"]) and not -1.0 <= row["best_similarity"] <= 1.0)):
                report["impossible"] += 1
                continue
            seen.add(row["frame"])
            rows.append(row)

    rows.sort(key=lambda r: r["frame"])
    kept = []
    for r in rows:
        if r["time_s"] < WARMUP_S:
            report["warmup"] += 1
        elif r["frame_ms"] > MAX_FRAME_MS:
            report["stalls"] += 1
        else:
            kept.append(r)
    report["rows_kept"] = len(kept)
    return kept, report


def rolling_mean(x, n):
    if len(x) < n or n < 2:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, np.nan), out])


def analyze(path):
    rows, rep = load_clean(path)
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n=== {name} ===")
    print(f"Cleaning: read {rep['rows_read']} rows, kept {rep['rows_kept']} | dropped: "
          f"{rep['bad_values']} unreadable, {rep['duplicates']} duplicate frames, {rep['impossible']} impossible values, "
          f"{rep['warmup']} warm-up (first {WARMUP_S:.0f}s), {rep['stalls']} stalls (>{MAX_FRAME_MS / 1000:.0f}s)")
    if len(rows) < 2:
        print("Not enough clean data to graph (run the camera a bit longer).")
        return

    t = np.array([r["time_s"] for r in rows])
    fps_inst = 1000.0 / np.array([r["frame_ms"] for r in rows])
    people = np.array([r["people"] for r in rows])
    faces = np.array([r["faces"] for r in rows])
    known = np.array([r["known"] for r in rows])
    unknown = np.array([r["unknown"] for r in rows])
    sim = np.array([r["best_similarity"] for r in rows])
    thresh = rows[-1]["match_thresh"]

    n = len(rows)
    duration = t[-1] - t[0]
    face_frames = int((faces > 0).sum())
    person_frames = int((people > 0).sum())
    known_frames = int((known > 0).sum())
    id_frames = int(((known + unknown) > 0).sum())
    sims = sim[~np.isnan(sim)]

    # frames where each person was recognized
    counts = {}
    for r in rows:
        for nm in filter(None, r["names"].split(";")):
            counts[nm] = counts.get(nm, 0) + 1

    print(f"Duration: {duration:.1f}s over {n} frames")
    print(f"Speed: {np.mean(fps_inst):.1f} FPS average, {np.median(fps_inst):.1f} median, "
          f"{np.percentile(fps_inst, 5):.1f} worst-5%  (frame time {np.mean(1000 / fps_inst):.1f} ms)")
    print(f"Person detected: {100 * person_frames / n:.1f}% of frames | face found: {100 * face_frames / n:.1f}% "
          f"| face found when a person was in view: {100 * face_frames / max(1, person_frames):.1f}%")
    if id_frames:
        print(f"Recognized as an enrolled person: {100 * known_frames / id_frames:.1f}% of frames with a checked face")
    if len(sims):
        print(f"Match similarity: mean {sims.mean():.3f}, min {sims.min():.3f}, max {sims.max():.3f} (threshold {thresh:.3f})")
    for nm, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {nm.replace('_', ' ')}: {c} frames")

    # ---------- graphs ----------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    title = name if name.upper().startswith("AVIS") else f"AVIS - {name}"
    fig.suptitle(f"{title}  ({duration:.0f}s, {n} clean frames)", fontsize=14, fontweight="bold")

    a = ax[0, 0]
    a.plot(t, fps_inst, color="#9ecae1", lw=0.8, label="per frame")
    win = max(2, int(round(np.median(fps_inst))))  # ~1 second window
    a.plot(t, rolling_mean(fps_inst, win), color="#08519c", lw=2, label="1 s average")
    a.axhline(np.mean(fps_inst), color="gray", ls="--", lw=1, label=f"mean {np.mean(fps_inst):.1f}")
    a.set(title="Speed (FPS)", xlabel="time (s)", ylabel="frames per second")
    a.set_ylim(bottom=0)
    a.legend(loc="lower right", fontsize=8)

    a = ax[0, 1]
    a.step(t, people, where="post", color="#31a354", lw=1.5, label="people (YOLO)")
    a.step(t, faces, where="post", color="#e6550d", lw=1.2, alpha=0.8, label="faces (YuNet)")
    a.set(title="Detections per frame", xlabel="time (s)", ylabel="count")
    a.set_ylim(0, max(people.max(), faces.max()) + 1)
    a.yaxis.get_major_locator().set_params(integer=True)
    a.legend(loc="upper right", fontsize=8)

    a = ax[1, 0]
    if len(sims):
        mask = ~np.isnan(sim)
        colors = np.where(sim[mask] >= thresh, "#3182bd", "#de2d26")
        a.scatter(t[mask], sim[mask], c=colors, s=8)
        a.axhline(thresh, color="black", ls="--", lw=1, label=f"match threshold {thresh:.3f}")
        a.set_ylim(min(0, sims.min() - 0.05), 1)
        a.legend(loc="lower right", fontsize=8)
    else:
        a.text(0.5, 0.5, "No faces checked\n(enroll someone first)", ha="center", va="center", transform=a.transAxes)
    a.set(title="Recognition confidence (blue = known, red = unknown)", xlabel="time (s)", ylabel="cosine similarity")

    a = ax[1, 1]
    no_face = int((faces == 0).sum())
    pending = int(((faces > 0) & ((known + unknown) == 0)).sum())   # face seen, identity not checked yet
    only_unknown = int(((unknown > 0) & (known == 0)).sum())
    cats = [("No face", no_face, "#bdbdbd"), ("Not identified yet", pending, "#fdd0a2")]
    cats += [(k.replace("_", " "), v, "#3182bd") for k, v in counts.items()]
    cats += [("Unknown", only_unknown, "#de2d26")]
    cats = [c for c in cats if c[1] > 0]
    labels, values, colors = zip(*cats)
    bars = a.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{100 * v / n:.0f}%", ha="center", va="bottom", fontsize=9)
    a.set(title="Who was on screen (share of frames)", ylabel="frames")

    fig.tight_layout()
    png = os.path.splitext(path)[0] + ".png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    clean_csv = os.path.splitext(path)[0] + "_clean.csv"
    with open(clean_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()})
    print(f"Graphs -> {png}\nClean data -> {clean_csv}")


def main():
    ap = argparse.ArgumentParser(description="Clean + graph AVIS session logs.")
    ap.add_argument("csv", nargs="?", help="Session CSV (default: newest in runs/)")
    ap.add_argument("--all", action="store_true", help="Analyze every session CSV in runs/")
    args = ap.parse_args()

    logs = sorted((p for p in glob.glob(os.path.join(RUNS_DIR, "*.csv")) if not p.endswith("_clean.csv")),
                  key=os.path.getmtime)
    if args.csv:
        targets = [args.csv]
    elif args.all:
        targets = logs
    else:
        targets = logs[-1:]
    if not targets:
        sys.exit("No session logs in runs/ yet. Run the camera first.")
    for p in targets:
        analyze(p)


if __name__ == "__main__":
    main()
