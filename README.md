# MOT_JDT_FairMOT

**FairMOT** - joint detection and tracking on KITTI-MOTS, trained from scratch.

- Paper: Zhang et al., IJCV 2021; official code: `ifzhang/FairMOT`
- Implementation: `common/mot/fairmot.py` (the module docstring below lists what was checked against the official
  code and every `[Unified]` difference)

## Results (KITTI-MOTS val)

| run | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | ms / frame |
|---|---|---|---|---|---|---|---|
| fairmot | 0.380 | 0.338 | 0.431 | 0.387 | 0.510 | 285 | 8.7 |


Comparison with tracking-by-detection on the same network: `RESULTS.md` of `MOT_Detectors_KITTIMOTS`, section 4.

## Implementation notes

```
MOT stage (2026-10-24): FairMOT (Zhang et al., IJCV 2021) network part. The association part is the tracker
"fairmot" in common/mot/trackers.py (JDETracker), fed with this network's detections and embeddings.
Official code: ifzhang/FairMOT src/lib (checked):
  opts.py            heads hm / wh (ltrb: 4) / reg / id (reid_dim 128), head_conv 256, K 500, conf_thres 0.4,
                     30 epochs, Adam 1e-4 (lr step 20), batch 12, multi_loss "uncertainty"
  trains/mot.py      det = focal hm + 0.1 L1 ltrb + L1 offset; id = cross entropy of a linear classifier on
                     emb_scale * normalize(id feature at the object centre), emb_scale = sqrt(2) * ln(nID - 1);
                     loss = 0.5 * (exp(-s_det) det + exp(-s_id) id + s_det + s_id), s_det -1.85, s_id -1.05
  datasets/jde.py    ltrb target = centre (float, output stride) to the four box sides; ids = track identity
  decode.py          3x3 peak NMS, top-K, box = (int centre + offset) -/+ ltrb, no box NMS

[Unified] decisions:
  * backbone = our CenterNet ResNet-18-DCN (detectors/centernet/build.py) with FairMOT's heads (head_conv 256),
    so CenterNet, CenterTrack and FairMOT share the backbone (official FairMOT: DLA-34, CrowdHuman pre-training).
  * two classes (car, pedestrian): class-specific heat maps, one identity classifier over all KITTI-MOTS train
    tracks (car and pedestrian ids together); the association runs per class like every tracker here.
  * detector protocol: from scratch, 30 epochs, Adam 1e-4 constant (no step at 20), effective batch 8, seed 777,
    common augmentation, input 1216 x 384, PreciseBN.
```

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

## Usage

```bash
pip install -r requirements.txt
pip install "git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a"
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/train_fairmot.py --check     # smoke test
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/train_fairmot.py             # training
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/dump_fairmot.py && python tools/mot/run_tracker.py --source kittimots --dets ~/mot_logs/dets/kittimots/fairmot.npz --trackers fairmot --causal
```

Trained weights: the GitHub Release of this repository (`fairmot_kittimots_e030.pth`, put it under `runs/fairmot/`).

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
