import torch
import cv2
import numpy as np
import os

def get_train_objs(path_train, path_gt):
  td = os.listdir(path_train)
  n = len(td)
  train_obj = np.zeros((n, 128, 128, 3), dtype=np.uint8)
  gt_obj = np.zeros((n, 128, 128, 3), dtype = np.uint8)
  for i, fil in enumerate(td):
      train_obj[i] = cv2.imread(path_train+fil)
      gt_obj[i] = cv2.imread(path_gt+fil)
  return train_obj, gt_obj

def get_test_objs(gp_test, gt_test, num):
# gp_test and gt_test are paths to data
  vd = os.listdir(gp_test)
  valid_gp = []
  valid_gt = []
  for i, fil in enumerate(vd):
      if(i < num):
        valid_gp.append(cv2.imread(gp_test+fil))
        valid_gt.append(cv2.imread(gt_test+fil))
      else:
        break
  return valid_gp, valid_gt

def transform_patches(p_in, p_gt):
  ind = np.arange(len(p_in))
  np.random.shuffle(ind)
  p_in = p_in[ind]
  p_gt = p_gt[ind]
  for i in range(len(p_in)):
    if i % 2 == 0:
      return
#    elif i % 4 == 1:
#      p_in[i] = cv2.rotate(p_in[i], cv2.ROTATE_90_CLOCKWISE)
#      p_gt[i] = cv2.rotate(p_gt[i], cv2.ROTATE_90_CLOCKWISE)
#    elif i % 4 == 2:
#      p_in[i] = cv2.rotate(p_in[i], cv2.ROTATE_180)
#      p_gt[i] = cv2.rotate(p_gt[i], cv2.ROTATE_180)
    elif i % 2 == 1:
      p_in[i] = cv2.flip(p_in[i],0)
      p_gt[i] = cv2.flip(p_gt[i],0)
  

#load crops of images 
def get_batch(train_obj, gt_obj, batch_size, shift, shuffle=False):
  m = len(train_obj)//batch_size
  block = m*batch_size - shift*batch_size if (shift+1)*batch_size > m*batch_size else (shift+1)*batch_size
  patches_train = train_obj[shift*batch_size:block].astype(np.float32)
  patches_gt = gt_obj[shift*batch_size:block].astype(np.float32) - patches_train
  if(shuffle):
    transform_patches(patches_train, patches_gt)
  patches_train = torch.from_numpy(patches_train.transpose((0,3,1,2)))
  patches_gt = torch.from_numpy(patches_gt.transpose((0,3,1,2)))
  return patches_train, patches_gt

def get_test_image(img):
  if img.shape[0]%128 >0:
    padx = 128 - img.shape[0]%128
  else:
    padx = 0
  if img.shape[1]%128 >0:
    pady = 128 - img.shape[1]%128
  else:
    pady:(i+1) = 0
  img1 = np.zeros((img.shape[0]+padx, img.shape[1]+pady, 3), dtype=np.float32)
  shape = img1.shape
  img1[:img.shape[0], :img.shape[1], :] = img.astype(np.float32)
  patches = img1.reshape(img1.shape[0]//128, 128, img1.shape[1]//128, 128, 3).swapaxes(1,2).reshape(-1,128,128,3)
  patches = patches.transpose((0,3,1,2))
  return patches, shape

def get_img(img, shape, gtshape):
  img = img.transpose((0,2,3,1))
  img = img.reshape(shape[0]//128, shape[1]//128, 128, 128, 3).swapaxes(1,2)
  img = img.reshape(shape[0], shape[1], 3)#.astype(np.uint8)
  img = img[:gtshape[0], :gtshape[1], :]
  return img 

