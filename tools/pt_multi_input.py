import torch
import torch.nn as nn
import torch.nn.functional as F
class nettest(nn.Module):
    def __init__(self):
        super(nettest, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        self.conv31 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.conv32 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        #self.global_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    #假设x1与x2的通道数均为3
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x1= self.conv31(x)
        x2= self.conv32(x)

        return x1, x2

