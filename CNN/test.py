import os

import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import matplotlib.pyplot as plt

import cv2
from data_loader import  get_test_objs, get_img, get_test_image
from model import Net

def trans(img, i):
  if(i == 1):
    return cv2.flip(img, 0)
  elif(i == 2): 
    return cv2.flip(img, 1)
  elif(i == 3): 
    return cv2.flip(img, -1)
  elif(i == 4):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
  elif(i == 5): 
    return cv2.rotate(cv2.flip(img, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)
  elif(i == 6):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
  elif(i == 7):
    return cv2.rotate(cv2.flip(img, 0), cv2.ROTATE_90_CLOCKWISE) 
  else:
    return img

def itrans(img, i):
  if(i < 4):
    return trans(img, i)
  elif(i == 4):
    return trans(img, 6)
  elif(i == 5):
    return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 0)
  elif(i == 6):
    return trans(img, 4)
  elif(i == 7):
    return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
  else:
    return img

def get_batch(patches, batch_size, shift):
  m = len(patches)//batch_size
  block = len(patches) if (shift+1)*batch_size > len(patches) else (shift+1)*batch_size
  return patches[shift*batch_size:block].type(torch.float32)

def place_batch(patches, patch, batch_size, shift):
  m = len(patches)//batch_size
  block = len(patches) if (shift+1)*batch_size > len(patches) else (shift+1)*batch_size
  patches[shift*batch_size:block] = patch.type(torch.float32).data.cpu()

VALID_PATH_GP = '/home/steven/dissertation/GP_image/data/GP/GP4_test/'
VALID_PATH_GT = '/home/steven/dissertation/GP_image/data/test_HR/'
OUTPATH = '/home/steven/dissertation/GP_image/data/GPCNN4/'

device =torch.device("cuda")
model = Net()
model.load_state_dict(torch.load('models4/model_epoch_80.pth').module.state_dict())#.to(device)
model.eval()
validation_loss = []
gp = []
gain = []
ntrans = 1 # 8

def test(valid_gp, valid_gt, direc):
    avg = 0
    print('------------------ Test ----------------------')
    for j in range(len(valid_gt)):
        img = np.zeros(valid_gp[j].shape, dtype=np.float32)
        for k in range(ntrans):
          im1 = trans(valid_gp[j], k)
          test_x, shape = get_test_image(im1)
          test_x = torch.from_numpy(test_x)#.to(device)
          for i in range(len(test_x)//64):
            patch = get_batch(test_x, 64, i).to(device)
            patch = (patch + model(patch))
            place_batch(test_x, patch, 64, i)
          im1 = get_img(test_x.numpy(), shape, im1.shape)
          im1 = itrans(im1, k)
          img = img + im1*(1./ntrans)
        cv2.imwrite(OUTPATH+direc[j], img)
        crit = psnr(valid_gt[j], img)
        crit2 = psnr(valid_gt[j], valid_gp[j])
        gain.append(crit-crit2)
        validation_loss.append(crit)
        gp.append(crit2)
        print("Image", direc[j], "GP", crit2, "GP+CNN", crit)

direc = os.listdir(VALID_PATH_GT)
gp_imgs, gt_imgs = get_test_objs(VALID_PATH_GP, VALID_PATH_GT, 100)

with torch.no_grad():
  test(gp_imgs, gt_imgs, direc)

validation_loss = np.asarray(validation_loss)
gp = np.asarray(gp)
gain = np.asarray(gain)
print('Max Gain in Image', direc[np.argmax(gain)])
N = validation_loss.size
print('Mean PSNR GP+CNN', validation_loss.mean())
print('Mean PSNR GP', gp.mean())
print('Average Gain', gain.mean())
#np.savetxt('mean_geoav2x_psnr_80.txt',validation_loss)
#plt.plot(np.arange(N), validation_loss, 'b.', label='GP+CNN')
#plt.plot(np.arange(N), validation_loss.mean()*np.ones(N), 'b-', label='GP+CNN mean') 
#plt.plot(np.arange(N), gp, 'g.', label = 'GP')
#plt.plot(np.arange(N), gp.mean()*np.ones(N), 'b-', label='GP mean') 
#plt.xlabel('Image ID')
#plt.ylabel('PSNR')
#plt.legend()
#plt.show()
#


