"""ASTPM demo: two self-downloaded YouTube clips per event, SIFT at 2 fps over the first 30 s, a 100-word vocabulary,
then level-0, USTPM and ASTPM distance matrices (STIP is out of scope: the tool only runs on Linux)."""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yt_dlp
from sklearn.cluster import MiniBatchKMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ver.astpm import astpm
from ver.data import EVENTS

CLIPS = Path("data/astpm_demo")
OUT = Path("results")
PER_EVENT, MAX_SECONDS, FPS, N_WORDS = 2, 30, 2, 100
DISTANCES = ("level 0", "USTPM", "ASTPM")


def download():
    """First PER_EVENT short search hits per event that download; a video-only H.264 stream needs no ffmpeg."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": "in_playlist", "outtmpl": str(CLIPS / "%(id)s.%(ext)s"),
                           "format": "worstvideo[vcodec^=avc1][height>=240]/worstvideo[vcodec^=avc1]"}) as ydl:
        for event in EVENTS:
            if len(list(CLIPS.glob(f"{event}_*.mp4"))) == PER_EVENT:
                continue
            hits = ydl.extract_info(f"ytsearch20:{event} home video", download=False)["entries"]
            for hit in (h for h in hits if h.get("duration") and 20 <= h["duration"] <= 300):
                i = len(list(CLIPS.glob(f"{event}_*.mp4")))
                if i == PER_EVENT:
                    break
                try:
                    ydl.download([hit["url"]])
                except yt_dlp.utils.DownloadError:
                    continue
                next(CLIPS.glob(f"{hit['id']}.*")).rename(CLIPS / f"{event}_{i}.mp4")
            if len(list(CLIPS.glob(f"{event}_*.mp4"))) < PER_EVENT:
                sys.exit(f"only found {len(list(CLIPS.glob(f'{event}_*.mp4')))} downloadable clips for {event}")


def sift_features(path):
    """SIFT descriptors and normalised (x, y, t) positions of frames sampled at FPS over the first MAX_SECONDS."""
    cap, sift = cv2.VideoCapture(str(path)), cv2.SIFT_create(nfeatures=100)
    step = max(round(cap.get(cv2.CAP_PROP_FPS) / FPS), 1)
    descs, xyt, n_sampled = [], [], 0
    for i in range(int(MAX_SECONDS * cap.get(cv2.CAP_PROP_FPS))):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, d = sift.detectAndCompute(gray, None)
        if d is not None:
            h, w = gray.shape
            descs.append(d)
            xyt += [(kp.pt[0] / w, kp.pt[1] / h, n_sampled) for kp in keypoints]
        n_sampled += 1
    xyt = np.array(xyt)
    xyt[:, 2] /= n_sampled
    return np.vstack(descs), np.minimum(xyt, 1 - 1e-9)


def run():
    download()
    clips = [CLIPS / f"{e}_{i}.mp4" for e in EVENTS for i in range(PER_EVENT)]
    features = [sift_features(c) for c in clips]
    vocab = MiniBatchKMeans(N_WORDS, random_state=0, n_init=3).fit(np.vstack([d for d, _ in features]))
    videos = [(vocab.predict(d), xyt) for d, xyt in features]
    n = len(videos)
    D = np.zeros((len(DISTANCES), n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[:, i, j] = D[:, j, i] = astpm(videos[i], videos[j], N_WORDS, "sift")
    OUT.mkdir(exist_ok=True)
    (OUT / "astpm_demo.json").write_text(json.dumps(
        {"clips": [c.stem for c in clips], **{name: np.round(M, 4).tolist() for name, M in zip(DISTANCES, D)}}, indent=1))


def plot():
    results = json.loads((OUT / "astpm_demo.json").read_text())
    clips = results["clips"]
    events = np.array([c.rsplit("_", 1)[0] for c in clips])
    fig, axes = plt.subplots(1, len(DISTANCES), figsize=(15, 5))
    for ax, name in zip(axes, DISTANCES):
        M = np.array(results[name])
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(len(clips)), clips, rotation=90, fontsize=7)
        ax.set_yticks(range(len(clips)), clips, fontsize=7)
        ax.set_title(f"SIFT {name}")
        fig.colorbar(im, ax=ax, fraction=0.046)

        print(f"\nSIFT {name} distances")
        print(f"{'':11}" + "".join(f"{c:>11}" for c in clips))
        for c, row in zip(clips, M):
            print(f"{c:11}" + "".join(f"{x:11.4f}" for x in row))
        nearest = np.where(np.eye(len(M), dtype=bool), np.inf, M).argmin(axis=1)
        print(f"nearest neighbour shares the event: {np.mean(events[nearest] == events):.0%}")
    fig.suptitle("ASTPM demo: SIFT distances between 12 self-downloaded clips (two per event)")
    fig.tight_layout()
    fig.savefig(OUT / "astpm_demo.png", dpi=150)


def main():
    if "--plot-only" not in sys.argv:  # --plot-only redraws the figure from the saved results/astpm_demo.json
        run()
    plot()


if __name__ == "__main__":
    main()
