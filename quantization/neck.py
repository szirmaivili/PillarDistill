import torch
import torch.nn as nn
import torch.nn.functional as F

class Neck(nn.Module):
    def __init__(self, in_channels=192, mid_channels=160, out_channels=96):
        super(Neck, self).__init__()

        # --- block_5 ---
        block5_layers = [
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, bias=False),
            nn.ReLU6(inplace=True),
        ]
        for _ in range(5):
            block5_layers.extend([
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU6(inplace=True),
            ])
        self.block_5 = nn.Sequential(*block5_layers)

        # --- deblock_5 ---
        self.deblock_5 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU6(inplace=True),
        )

        # --- block_4 ---
        block4_layers = [
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, bias=False),
            nn.ReLU6(inplace=True),
        ]
        for _ in range(5):
            block4_layers.extend([
                nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU6(inplace=True),
            ])
        self.block_4 = nn.Sequential(*block4_layers)

        # --- deblock_4 ---
        self.deblock_4 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.ReLU6(inplace=True),
        )

    def forward(self, conv_4, conv_5):
        up_4 = self.deblock_4(conv_4)
        up_5 = self.deblock_5(self.block_5(conv_5))
        x = torch.cat([up_4, up_5], dim=1)  # TensorFlow axis=-1 → PyTorch dim=1 (channels)
        x = self.block_4(x)
        return x
    
class Neck_mod(nn.Module):
    def __init__(self, in_channels=192, mid_channels=160, out_channels=96):
        super(Neck_mod, self).__init__()

        # --- block_4 ---
        block4_layers = [
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, bias=False),
            nn.ReLU6(inplace=True),
        ]
        for _ in range(5):
            block4_layers.extend([
                nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU6(inplace=True),
            ])
        self.block_4 = nn.Sequential(*block4_layers)

        # --- deblock_4 ---
        self.deblock_4 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels*2, kernel_size=3, stride=1, bias=False),
            nn.ReLU6(inplace=True),
        )

    def forward(self, conv_4):
        up_4 = self.deblock_4(conv_4)
        x = self.block_4(up_4)
        return x