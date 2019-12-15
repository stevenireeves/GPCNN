import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np


class Net(nn.Module):

  #Batch shape from input x is (3, 64, 64)
    def __init__(self):
        super(Net, self).__init__()
        
        # First Layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=4)
        
        # Second
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
       
        # Third
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2) 
       
        # Final 
        #128 * 58 * 58 = 430592
        self.dense = nn.Linear(430592,12288)
        #out is (3, 64, 64)
        
        if(torch.cuda.is_available()):
            self.cuda()

        
    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = F.relu(self.conv3(x)) 
      x = x.view(x.size(0), -1) # We expect the output values to be positive 
      x = F.relu(self.dense(x)) # Combine the efforts of the convultions to deconcolve image 
      x = x.view(3,64,64) 
      return x

