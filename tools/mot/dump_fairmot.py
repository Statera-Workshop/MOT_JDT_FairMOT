#!/usr/bin/env python3
"""
MOT stage (2026-10-24): run the trained FairMOT network over KITTI-MOTS val and save detections + embeddings in the
dump_dets / dump_reid format (~/mot_logs/dets/kittimots/fairmot.npz: "<seq>", "<seq>__ms", "<seq>__feat", "meta").
Then:  python tools/mot/run_tracker.py --source kittimots --dets ~/mot_logs/dets/kittimots/fairmot.npz --causal
gives FairMOT itself (tracker "fairmot") and the other trackers on the same detections / embeddings.
Latency "<seq>__ms": forward (detection + id heads) + peak decode + embedding gather, batch 1, fp32, cuda
synchronize, after 50 warm-up forwards; letterbox not timed. No box NMS (official decode). Stored: score >= 0.05.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.datasets.voc import letterbox, to_tensor  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES  # noqa: E402
from common.mot.fairmot import FairMOTNet, decode  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(C.RUNS_ROOT / "fairmot" / "fairmot_kittimots_e030.pth"))
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--out", default="~/mot_logs/dets/kittimots/fairmot.npz")
    a = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(Path(a.weights).expanduser(), map_location=device, weights_only=False)
    model = FairMOTNet(len(KITTI_CLASSES), ck["num_ids"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    root = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
    seqmap = [l.split() for l in (root / "ImageSets" / "MOT" / "val_seqmap.txt").read_text().split("\n") if l]
    first = Image.open(root / "JPEGImages" / f"{seqmap[0][0]}_000000.jpg").convert("RGB")
    x = to_tensor(letterbox(first, ck["img_size"])[0]).unsqueeze(0).to(device)
    for _ in range(50):
        decode(model(x))
    torch.cuda.synchronize()
    arrays, all_ms = {}, []
    for seq, n in seqmap:
        rows, feats, ms = [], [], []
        for f in range(int(n)):
            im = Image.open(root / "JPEGImages" / f"{seq}_{f:06d}.jpg").convert("RGB")
            lb, sc, px, py = letterbox(im, ck["img_size"])
            x = to_tensor(lb).unsqueeze(0).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r, e = decode(model(x))
            keep = r[:, 4] >= a.conf
            r, e = r[keep].cpu().numpy(), e[keep].cpu().numpy()
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
            r[:, [0, 2]] = (r[:, [0, 2]] - px) / sc
            r[:, [1, 3]] = (r[:, [1, 3]] - py) / sc
            rows.append(np.c_[np.full(len(r), f), r])
            feats.append(e)
        arrays[seq] = np.concatenate(rows).astype(np.float32)
        arrays[f"{seq}__feat"] = np.concatenate(feats).astype(np.float16)
        arrays[f"{seq}__ms"] = np.array(ms, np.float32)
        all_ms += ms
        print(f"  {seq}: {n} frames, {len(arrays[seq])} dets, {np.median(ms):.1f} ms median", flush=True)
    meta = {"version": "fairmot", "weights": str(a.weights), "source": "kittimots", "split": "val",
            "input": f"{ck['img_size']}x384", "conf": a.conf, "classes": list(KITTI_CLASSES), "family": "JDT",
            "det_ms_median": float(np.median(all_ms)), "det_ms_p95": float(np.percentile(all_ms, 95)),
            "note": "embeddings from the network id head (no separate ReID)"}
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(f"-> {out}  {meta['det_ms_median']:.1f} ms median")


if __name__ == "__main__":
    main()
