import torch
import torch.nn as nn

def conv_bn_relu6(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.ReLU6(inplace=True)
    )


# === Egyetlen head definíció (reg, height, dim, rot, vel, iou, hm) ===
def make_head(in_ch, out_ch, is_hm=False):
    head = nn.Sequential(
        conv_bn_relu6(in_ch, 64),
        nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
    )

    # Inicializálás
    if is_hm:
        nn.init.constant_(head[-1].bias, -2.19)
    else:
        nn.init.kaiming_normal_(head[-1].weight, mode='fan_out', nonlinearity='relu')
        if head[-1].bias is not None:
            nn.init.constant_(head[-1].bias, 0.0)

    return head


# === Egyetlen task-hoz tartozó "SepHead" ===
class SepHead(nn.Module):
    def __init__(self, num_classes=2, hm_channels_override=None):
        super(SepHead, self).__init__()

        # head → kimeneti csatornák
        self.head_defs = {
            "reg": 2,
            "height": 1,
            "dim": 3,
            "rot": 2,
            "vel": 2,
            "iou": 1,
            "hm": hm_channels_override if hm_channels_override is not None else num_classes,
        }

        for head, out_ch in self.head_defs.items():
            setattr(self, head, make_head(64, out_ch, is_hm=(head == "hm")))

    def forward(self, x):
        out = {}
        for head in self.head_defs.keys():
            out[head] = getattr(self, head)(x)
        return out


# === Fő Bbox modul, a CenterHead-nek megfelelő ===
class Bbox(nn.Module):
    def __init__(self, num_tasks=6):
        super(Bbox, self).__init__()

        # Shared convolution blokk
        self.shared_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU6(inplace=True)
        )

        # Taskok listája (6 db SepHead)
        self.tasks = nn.ModuleList()

        # A heatmap (hm) csatornaszám taskonként: [1, 2, 2, 1, 2, 2]
        hm_channels = [1, 2, 2, 1, 2, 2]
        for i in range(num_tasks):
            self.tasks.append(SepHead(num_classes=hm_channels[i]))

        self.class_names = [['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'],
                            ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']]

    def forward(self, x):
        x = self.shared_conv(x)
        outputs = []
        for task_head in self.tasks:
            outputs.append(task_head(x))
        return outputs