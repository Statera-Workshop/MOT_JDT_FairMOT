"""
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
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common.datasets.voc import hsv_jitter, letterbox, random_scale_translate, to_tensor
from detectors.centernet.build import (DOWN_RATIO, BasicBlock, PoseResNet, _neg_loss, _sigmoid,
                                       _transpose_and_gather_feat, draw_umich_gaussian, gaussian_radius,
                                       reg_l1_loss)

MAX_OBJS, REID_DIM = 128, 128


class FairMOTNet(nn.Module):
    def __init__(self, num_classes, num_ids):
        super().__init__()
        self.num_classes, self.num_ids = num_classes, num_ids
        self.net = PoseResNet(BasicBlock, [2, 2, 2, 2], {"hm": num_classes, "wh": 4, "reg": 2, "id": REID_DIM}, 256)
        self.classifier = nn.Linear(REID_DIM, num_ids)
        self.emb_scale = math.sqrt(2) * math.log(num_ids - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16

    def forward(self, x):
        return self.net(x)

    def loss(self, out, t):
        hm = _neg_loss(_sigmoid(out["hm"].clone()), t["hm"])
        wh = reg_l1_loss(out["wh"], t["mask"], t["ind"], t["wh"])
        off = reg_l1_loss(out["reg"], t["mask"], t["ind"], t["reg"])
        det = hm + 0.1 * wh + off
        feat = _transpose_and_gather_feat(out["id"], t["ind"])[t["mask"] > 0]
        if len(feat):
            idl = F.cross_entropy(self.classifier(self.emb_scale * F.normalize(feat)), t["ids"][t["mask"] > 0])
        else:
            idl = feat.sum() * 0.0
        loss = 0.5 * (torch.exp(-self.s_det) * det + torch.exp(-self.s_id) * idl + self.s_det + self.s_id)
        return loss, {"hm": float(hm), "wh": float(wh), "off": float(off), "id": float(idl),
                      "s_det": float(self.s_det), "s_id": float(self.s_id)}


class FrameDataset(torch.utils.data.Dataset):
    """KITTI-MOTS train frames -> image, CenterNet ltrb targets + identity labels (global id index)."""

    def __init__(self, gt, paths, img_size, num_classes, train=True, aug=None, hflip_prob=0.5):
        self.items = [(s, f) for s in gt for f in range(len(gt[s]))]
        self.gt, self.paths, self.size, self.C = gt, paths, img_size, num_classes
        self.train, self.aug, self.hflip = train, aug or {}, hflip_prob
        self.gid = {}
        for s in gt:
            for fr in gt[s]:
                for i in fr["ids"]:
                    self.gid.setdefault((s, int(i)), len(self.gid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, f = self.items[i]
        im, sc, px, py = letterbox(Image.open(self.paths[s][f]).convert("RGB"), self.size)
        g = self.gt[s][f]
        boxes = g["boxes"] * sc + np.array([px, py, px, py]) if len(g["boxes"]) else np.zeros((0, 4))
        ids, cls = g["ids"], g["cls"]
        arr = np.asarray(im).copy()
        if self.train:
            arr = hsv_jitter(arr, self.aug.get("hue", 0.1), self.aug.get("saturation", 1.5),
                             self.aug.get("exposure", 1.5))
            arr, boxes, keep = random_scale_translate(arr, boxes, self.aug.get("jitter", 0.2))
            boxes, ids, cls = boxes[keep], ids[keep], cls[keep]
            if np.random.random() < self.hflip:
                arr = arr[:, ::-1].copy()
                boxes = boxes.copy()
                boxes[:, [0, 2]] = arr.shape[1] - boxes[:, [2, 0]]
        H, W = arr.shape[:2]
        oh, ow = H // DOWN_RATIO, W // DOWN_RATIO
        hm = np.zeros((self.C, oh, ow), np.float32)
        wh = np.zeros((MAX_OBJS, 4), np.float32)
        reg = np.zeros((MAX_OBJS, 2), np.float32)
        ind = np.zeros(MAX_OBJS, np.int64)
        mask = np.zeros(MAX_OBJS, np.uint8)
        idt = np.zeros(MAX_OBJS, np.int64)
        for k, (b, tid, c) in enumerate(zip(boxes[:MAX_OBJS], ids[:MAX_OBJS], cls[:MAX_OBJS])):
            bo = b / DOWN_RATIO
            bo[[0, 2]] = np.clip(bo[[0, 2]], 0, ow - 1)
            bo[[1, 3]] = np.clip(bo[[1, 3]], 0, oh - 1)
            h, w = bo[3] - bo[1], bo[2] - bo[0]
            if h <= 0 or w <= 0:
                continue
            radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
            ct = np.array([(bo[0] + bo[2]) / 2, (bo[1] + bo[3]) / 2], np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[int(c)], ct_int, radius)
            wh[k] = ct[0] - bo[0], ct[1] - bo[1], bo[2] - ct[0], bo[3] - ct[1]
            ind[k] = ct_int[1] * ow + ct_int[0]
            reg[k] = ct - ct_int
            mask[k] = 1
            idt[k] = self.gid[(s, int(tid))]
        t = {"hm": hm, "wh": wh, "reg": reg, "ind": ind, "mask": mask, "ids": idt}
        return to_tensor(arr), {k: torch.from_numpy(v) for k, v in t.items()}


@torch.no_grad()
def decode(out, K=500):
    """-> rows [K, 6] x1 y1 x2 y2 score class (input pixels) and embeddings [K, 128] (L2-normalised), image 0."""
    heat = out["hm"].sigmoid()
    hmax = F.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * (hmax == heat).float()
    B, C, H, W = heat.shape
    scores, idx = heat.view(B, -1).topk(min(K, C * H * W))
    cls = idx // (H * W)
    pix = idx % (H * W)
    ys, xs = (pix // W).float(), (pix % W).float()
    n = scores.shape[1]

    def gather(t):
        return t.view(B, t.shape[1], -1).gather(2, pix[:, None, :].expand(B, t.shape[1], n))

    reg, wh = gather(out["reg"]), gather(out["wh"])
    emb = F.normalize(gather(out["id"]), dim=1)
    cx, cy = xs + reg[:, 0], ys + reg[:, 1]
    rows = torch.stack([cx - wh[:, 0], cy - wh[:, 1], cx + wh[:, 2], cy + wh[:, 3], scores, cls.float()], -1)
    rows[..., :4] *= DOWN_RATIO
    return rows[0], emb[0].T
