import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from interp import interp as gp2d 

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # First Layer
        self.conv1 = nn.Conv2d(1, 32, 4)
#				self.pool1  = nn.MaxPool2d(2,2)
        
        # Second
        self.conv2 = nn.Conv2d(32, 64, 3)
# 				self.pool2  = nn.MaxPool2d(2,2)
       
        # Third
        self.conv3 = nn.Conv2d(64, 128, 2) 
# 				self.pool3  = nn.MaxPool2d(2,2)
       
        # Final 
        self.dense = nn.Linear(262144,4096) #TODO Calculate Size in and out 
        
        if(torch.cuda.is_available()):
            self.cuda()

        
    def forward(self, x):
			x = self.conv1(x)
			x = self.conv2(x)
			x = self.conv3(x)
			x = self.dense(x) # Combine the efforts of the convultions into an "enhanced image" 
			x = x.view(64,64,-1) 
			x = gp2d(64,64, 2,2,x,x) # Interpolate using a Guassian Process Model to get a HD-HR image
      return x
