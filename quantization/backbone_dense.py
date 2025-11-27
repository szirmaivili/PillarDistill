import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch import ConvAlgo

class Backbone(nn.Module):
    def __init__(self, input_channels=32):
        super(Backbone, self).__init__()

        self.conv1 = nn.Sequential(
            # Első blokk (Sparse2DBasicBlockV): 3×SubMConv2d+BN + ReLU6
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(input_channels, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                ),
                nn.ReLU6(inplace=True)
            ),

            # Második blokk (Sparse2DBasicBlock): 2×SubMConv2d+BN + ReLU6
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
            nn.ReLU6(inplace=True),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
            
            nn.ReLU6(inplace=True),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
            
            nn.ReLU6(inplace=True),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),
            
            nn.ReLU6(inplace=True),
            nn.Sequential(
                nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                
                nn.ReLU6(inplace=True)
            )
        )

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv_4 = x
        x = self.conv5(conv_4)
        conv_5 = x

        return conv_4, conv_5
    
class Backbone_mod(nn.Module):

    def __init__(self, input_channels=32):
        super(Backbone_mod, self).__init__()

        self.conv1 = nn.Sequential(
            # Első blokk (Sparse2DBasicBlockV): 3×SubMConv2d+BN + ReLU6
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(input_channels, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                ),
                nn.ReLU6(inplace=True)
            ),

            # Második blokk (Sparse2DBasicBlock): 2×SubMConv2d+BN + ReLU6
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=[3,3], stride=[1,1],
                                    padding=[1,1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
            nn.ReLU6(inplace=True),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
            
            nn.ReLU6(inplace=True),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
            
            nn.ReLU6(inplace=True),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
                    
                ),
                nn.ReLU6(inplace=True)
            )
        )

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv_4 = x
        
        return conv_4

    