"""
Author: Arshmeet
- Edited IHC_Classifier to handle any input size (power of two)
"""

from torch import nn
import torch.nn.functional as F

class IHC_Classifier(nn.Module):
    def __init__(self, channel=3,capacity=16, img_size = 256):
        super(IHC_Classifier, self).__init__()
        c = capacity
        self.c = capacity
        self.channel = channel
        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=self.c, kernel_size=4, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=self.c*2, out_channels=self.c*4, kernel_size=4, stride=2, padding=1) 
        self.conv4 = nn.Conv2d(in_channels=self.c*4, out_channels=self.c*8, kernel_size=4, stride=2, padding=1) 
        self.conv5 = nn.Conv2d(in_channels=self.c*8, out_channels=self.c*16, kernel_size=4, stride=2, padding=1) 
        self.conv = nn.Conv2d(in_channels=c*16, out_channels=c*32, kernel_size=4, stride=2, padding=1) 

        """
        Each layer has stride = 2 => spatial size halved every time. 
        6 layers => spatial size is 1/64th of the original image. 
        Feature map dimensions (fm_dims) are computed accordingly. 
        """
        fm_dims = img_size // 64 
        self.linear1 = nn.Linear(c * 32 * fm_dims * fm_dims, 2)
             
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.softmax(x, dim=1)
        return x