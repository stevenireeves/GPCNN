import cv2
import torch
import numpy as np
from interp import interpolate
from model2 import Net

device = torch.device('cuda')
model = torch.load('models_2/model_epoch_150.pth')
img = torch.from_numpy(cv2.imread(FILE)).to(device)
img2 = torch.empty((img.shape[0]*2, img.shape[1]*2, img.shape[2]), dtype=torch.float32, device=device)
interpolate(img[:,:,0].data_ptr, img[:,:,1].data_ptr, img[:,:,2].data_ptr,
            img2[:,:,0].data_ptr, img2[:,:,1].data_ptr, img2[:,:,2].data_ptr,
            np.array([2,2], dtype=np.int32), 
            np.array([img.shape[0], img.shape[1]], dtype=np.int32))

img2 = model(img2.view(-1, 64, 64, 3))
cv2.imwrite(img2.cpu().numpy())
