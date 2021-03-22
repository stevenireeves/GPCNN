import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.cuda.amp import autocast

import numpy as np


class Net(nn.Module):

  #Batch shape from input x is (3, 64, 64)
    def __init__(self):
        super(Net, self).__init__()
        self.elu = nn.ELU()
#       Padding Layer for kernel size 7
        self.pad3= nn.ReflectionPad2d(3)
#       Padding Layer for kernel size 5
        self.pad2 = nn.ReflectionPad2d(2)
#       Padding Layer for kernel size 3
        self.pad1 = nn.ReflectionPad2d(1)

#       input
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(64)
#       block 1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7)
#       block 2
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=5)
#       block 3
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
#       block 4
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
#       block 5
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
#       block 6
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
#       output layer 
        self.conv14 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

        #out is (3, 64, 64)
        if(torch.cuda.is_available()):
            self.cuda()

    @autocast()        
    def forward(self, x):
#input layer
      z = self.conv1_bn(self.conv1(self.pad3(x)))
# Residual Blocks
    #block 1
      y = self.elu(self.conv2(self.pad3(z)))
      y = self.conv3(self.pad3(y))
      x = z + y
    #block 2
      y = self.elu(self.conv4(self.pad2(x)))
      y = self.conv5(self.pad2(y))
      x = x + y
    #block 3 
      y = self.elu(self.conv6(self.pad2(x)))
      y = self.conv7(self.pad2(y))
      x = x + y
    #block 4 
      y = self.elu(self.conv8(self.pad1(x)))
      y = self.conv9(self.pad1(y))
      x = x + y
    #block 5 
      y = self.elu(self.conv10(self.pad1(x)))
      y = self.conv11(self.pad1(y))
      x = x + y
    #block 6 
      y = self.elu(self.conv12(self.pad1(x)))
      y = self.conv13(self.pad1(y))
      x = x + y
# Output Layer
 #     x = self.conv14(x)
      x = x + z
      x = self.conv14(x)
      return x 

    def _initialize_weights(self):
      I.orthogonal_(self.conv1.weight)
      I.orthogonal_(self.conv2.weight)
      I.orthogonal_(self.conv3.weight)
      I.orthogonal_(self.conv4.weight)
      I.orthogonal_(self.conv5.weight)
      I.orthogonal_(self.conv6.weight)
      I.orthogonal_(self.conv7.weight)
      I.orthogonal_(self.conv8.weight)
      I.orthogonal_(self.conv9.weight)
      I.orthogonal_(self.conv10.weight)
      I.orthogonal_(self.conv11.weight)
      I.orthogonal_(self.conv12.weight)
      I.orthogonal_(self.conv13.weight)
      I.orthogonal_(self.conv14.weight)

