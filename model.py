import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # First Layer
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        # Second
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Third
        self.conv3 = nn.Conv2d(64, 128, 2) 
        
        # Final 
        self.dense = nn.Linear(,) #TODO Calculate Size in and out 
        
        if(torch.cuda.is_available()):
            self.cuda()

        
    def forward(self, x):
			x = self.conv1(x)
			x = self.conv2(x)
			x = self.conv3(x)
			x = self.dense(x) # Combine the efforts of the convultions into an "enhanced image" 
			x = gpinterp(x) # Interpolate using a Guassian Process Model to get a HD-HR image
        return x
