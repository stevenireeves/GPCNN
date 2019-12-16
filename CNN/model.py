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
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=4)
        # Second
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        # Third
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=8,kernel_size=2) 
        # Final 
        #8 * 58 * 58 = 26912
        self.dense = nn.Linear(26912,12288)
        #out is (3, 64, 64)
        
        if(torch.cuda.is_available()):
            self.cuda()

        
    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x)) 
      x = x.view(x.size(0), -1) 
      x = self.dense(x) # Combine the efforts of the convultions to deconcolve image 
      x = x.view(-1,3,64,64) 
      return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight)
        init.orthogonal_(self.conv2.weight)
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.dense.weight, init.calculate_gain('relu'))
