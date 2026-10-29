"""
Shared Darknet (.cfg) network builder.

The three legacy v3 projects each carried their own copy of models/models.py.
The parser, the convolution/route/shortcut/upsample construction and the
forward pass were identical; only the [yolo] block differed. Here the common
part exists once, and each detector passes a `yolo_factory` that builds its
own detection layer (the algorithm core).

    yolo_factory(block: dict, hyperparams: dict) -> nn.Module
        module.forward(x, targets, img_size) -> (output, loss)

[Unified] BatchNorm momentum is fixed to the PyTorch default 0.1 for every
task (legacy Complex/YOLO3D used 0.9, Image used 0.1).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_model_config(path):
    """Parse a Darknet .cfg file into a list of block dicts (legacy parser)."""
    with open(path, "r") as f:
        lines = f.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith("["):
            module_defs.append({})
            module_defs[-1]["type"] = line[1:-1].rstrip()
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


def patch_head(blocks, num_classes, num_box_attr):
    """
    Set `classes` in every [yolo] block and resize the convolution that feeds
    it to len(mask) * (num_box_attr + num_classes) filters.

    This lets the same .cfg serve VOC (20), COCO (80) or KITTI (3) and lets a
    detector change its number of regressed attributes (e.g. YOLO3D direct yaw)
    without hand-editing the file.
    """
    for i, block in enumerate(blocks):
        if block["type"] != "yolo":
            continue
        block["classes"] = str(num_classes)
        num_masks = len(block["mask"].split(","))
        j = i - 1
        while blocks[j]["type"] != "convolutional":
            if blocks[j]["type"] == "control_channels":    # [Unified] YOLOR: the implicit_mul it multiplies by
                blocks[int(blocks[j]["from"]) + 1]["filters"] = str(num_masks * (num_box_attr + num_classes))
            j -= 1
        blocks[j]["filters"] = str(num_masks * (num_box_attr + num_classes))
    return blocks


class ImplicitA(nn.Module):
    """YOLOR implicit addition (WongKinYiu/yolor utils/layers.py, unchanged)."""

    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class ImplicitM(nn.Module):
    """YOLOR implicit multiplication (WongKinYiu/yolor utils/layers.py, unchanged)."""

    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class EmptyModule(nn.Module):
    """Placeholder for route / shortcut."""


class Reorg(nn.Module):
    """
    YOLOv2 passthrough layer (space-to-depth): [B, C, H, W] -> [B, C*s*s, H/s, W/s].
    Channel ordering follows pixel-unshuffle; it only matters when loading
    original Darknet weights beyond the backbone (not used here).
    """

    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.stride
        x = x.view(b, c, h // s, s, w // s, s).permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


def create_network(blocks, yolo_factory, bn_momentum=0.1):
    hyperparams = blocks.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, block in enumerate(blocks):
        modules = nn.Sequential()
        if block["type"] == "convolutional":
            bn = int(block["batch_normalize"])
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(output_filters[-1], filters, kernel_size,
                          stride=int(block["stride"]), padding=(kernel_size - 1) // 2, bias=not bn),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=bn_momentum, eps=1e-5))
            act = block.get("activation", "linear")
            if act == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif act == "mish":   # YOLOv4
                modules.add_module(f"mish_{module_i}", nn.Mish())
            elif act in ("swish", "silu"):  # [Unified] YOLOv7 swish / YOLOR silu (= SiLU)
                modules.add_module(f"swish_{module_i}", nn.SiLU())
            elif act == "relu":
                modules.add_module(f"relu_{module_i}", nn.ReLU())
            elif act != "linear":
                raise ValueError(f"Unsupported activation: {act}")
        elif block["type"] == "maxpool":
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            if stride == 1 and kernel_size == 2:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            modules.add_module(f"maxpool_{module_i}",
                               nn.MaxPool2d(kernel_size, stride, padding=int((kernel_size - 1) // 2)))
        elif block["type"] == "upsample":
            modules.add_module(f"upsample_{module_i}", Upsample(int(block["stride"])))
        elif block["type"] == "reorg":
            s = int(block["stride"])
            filters = output_filters[-1] * s * s
            modules.add_module(f"reorg_{module_i}", Reorg(s))
        elif block["type"] == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            filters = sum(output_filters[1:][i] for i in layers)
            modules.add_module(f"route_{module_i}", EmptyModule())
        elif block["type"] == "shortcut":
            filters = output_filters[1:][int(block["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyModule())
        elif block["type"] in ("implicit_add", "implicit_mul"):    # [Unified] YOLOR implicit knowledge
            filters = int(block["filters"])
            modules.add_module(f"{block['type']}_{module_i}",
                               ImplicitA(filters) if block["type"] == "implicit_add" else ImplicitM(filters))
        elif block["type"] in ("shift_channels", "control_channels"):  # [Unified] YOLOR: x + a / x * a
            filters = output_filters[-1]
            modules.add_module(f"{block['type']}_{module_i}", EmptyModule())
        elif block["type"] == "yolo":
            modules.add_module(f"yolo_{module_i}", yolo_factory(block, hyperparams))
        else:
            raise ValueError(f"Unsupported block type: {block['type']}")
        output_filters.append(filters)
        module_list.append(modules)
    return hyperparams, module_list


class Darknet(nn.Module):
    """
    forward(x)            -> raw predictions [B, N, attr]       (inference)
    forward(x, targets)   -> (loss, raw predictions)            (training)
    """

    def __init__(self, cfg_path, yolo_factory, num_classes, num_box_attr, bn_momentum=0.1):
        super().__init__()
        self.blocks = patch_head(parse_model_config(cfg_path), num_classes, num_box_attr)
        self.hyperparams, self.module_list = create_network(
            [dict(b) for b in self.blocks], yolo_factory, bn_momentum)
        self.blocks = self.blocks[1:]  # drop [net]
        self.yolo_layers = [m[0] for m in self.module_list if hasattr(m[0], "metrics")]

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for block, module in zip(self.blocks, self.module_list):
            t = block["type"]
            if t in ("convolutional", "upsample", "maxpool", "reorg"):
                x = module(x)
            elif t == "route":
                x = torch.cat([layer_outputs[int(i)] for i in block["layers"].split(",")], 1)
            elif t == "shortcut":
                x = layer_outputs[-1] + layer_outputs[int(block["from"])]
            elif t in ("implicit_add", "implicit_mul"):     # [Unified] YOLOR: the layer outputs its vector
                x = module[0]()
            elif t == "shift_channels":
                x = layer_outputs[int(block["from"])].expand_as(x) + x
            elif t == "control_channels":
                x = layer_outputs[int(block["from"])].expand_as(x) * x
            elif t == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1)
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Load Darknet binary weights (e.g. darknet53.conv.74). Legacy code."""
        with open(weights_path, "rb") as f:
            np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        name = str(weights_path)
        cutoff = (75 if "darknet53.conv.74" in name else 23 if "darknet19_448.conv.23" in name
                  else 137 if "yolov4.conv.137" in name else None)
        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if i == cutoff:
                break
            if block["type"] != "convolutional":
                continue
            conv = module[0]
            if int(block["batch_normalize"]):
                bn = module[1]
                for tensor in (bn.bias, bn.weight, bn.running_mean, bn.running_var):
                    n = tensor.numel()
                    tensor.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(tensor))
                    ptr += n
            else:
                n = conv.bias.numel()
                conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(conv.bias))
                ptr += n
            n = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(conv.weight))
            ptr += n


def weights_init_normal(m):
    """Legacy Darknet-PyTorch initialisation, applied to every task."""
    # [Unified] type checks instead of class-name matching (YOLOv5 has a "Conv" wrapper block)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
