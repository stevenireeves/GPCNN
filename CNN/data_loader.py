import torch
import cv2
import numpy as np
import os
import ctypes
import numpy.ctypeslib as npct

libinterp = ctypes.cdll.LoadLibrary('../interp/cpp_pipeline/interp.so')
interpolate = libinterp.interpolate_color
interpolate.restype = None
interpolate.argtypes = [npct.ndpointer(ctypes.c_ubyte), npct.ndpointer(ctypes.c_ubyte),
                        npct.ndpointer(ctypes.c_ubyte), npct.ndpointer(ctypes.c_float),
                        npct.ndpointer(ctypes.c_float), npct.ndpointer(ctypes.c_float),
                        npct.ndpointer(ctypes.c_int), npct.ndpointer(ctypes.c_int)]

def gp(img, up):
    size = np.asarray(img.shape, dtype=np.int32)
    upsamp = np.array([up, up],dtype=np.int32)
    b = np.zeros((size[0]*up, size[1]*up), dtype=np.float32)
    g = np.zeros((size[0]*up, size[1]*up), dtype=np.float32)
    r = np.zeros((size[0]*up, size[1]*up), dtype=np.float32)
    interpolate(img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten(),
                b, g, r, upsamp, np.array([size[1], size[0]], dtype=np.int32))

    img_out = np.stack([b, g, r], axis=2).astype(np.float16)
    return img_out
'''
  interpolate(img[:,:,0].flatten(), b, upsamp, np.array([size[1], size[0]], dtype=np.int32))
  interpolate(img[:,:,1].flatten(), g, upsamp, np.array([size[1], size[0]], dtype=np.int32))
  interpolate(img[:,:,2].flatten(), r, upsamp, np.array([size[1], size[0]], dtype=np.int32))
'''

def ubyteize(img):
    img[img>255] = 255
    img[img<0] = 0
    return img.astype(np.uint8)

def split(img, size):
  sx, sy = img.shape[:2]
  rx = sx - np.remainder(sx, size)
  ry = sy - np.remainder(sy, size)
  img = img[:rx, :ry, :]
#  img = ubyteize(img)
  return img.reshape((-1,size,size,3))

def get_train_objs(path_train, path_gt, size):
  td = os.listdir(path_train)
  n = len(td)
  train_obj = [] # = np.zeros((n, 128, 128, 3), dtype=np.uint8)
  gt_obj = [] #np.zeros((n, 128, 128, 3), dtype = np.uint8)
  for i, fil in enumerate(td):
      train_obj.append(split(gp(cv2.imread(path_train+fil),2),size))
      gt_obj.append(split(cv2.imread(path_gt+fil),size))
  return np.vstack(train_obj), np.vstack(gt_obj)

def get_test_objs(gp_test, gt_test, num, size):
# gp_test and gt_test are paths to data
  vd = os.listdir(gp_test)
  valid_gp = []
  valid_gt = []
  for i, fil in enumerate(vd):
      if(i < num):
        valid_gp.append(split(gp(cv2.imread(gp_test+fil),2),size))
        valid_gt.append(split(cv2.imread(gt_test+fil),size))
      else:
        break
  return np.vstack(valid_gp), np.vstack(valid_gt)

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
  patches_train = train_obj[shift*batch_size:block]#.astype(np.float16)
  patches_gt = gt_obj[shift*batch_size:block] - patches_train
  if(shuffle):
    transform_patches(patches_train, patches_gt)
  patches_train = torch.from_numpy(patches_train.transpose((0,3,1,2)))
  patches_gt = torch.from_numpy(patches_gt.transpose((0,3,1,2)))
  return patches_train, patches_gt

def get_test_image(img, size):
  if img.shape[0]%size >0:
    padx = size - img.shape[0]%size
  else:
    padx = 0
  if img.shape[1]%size >0:
    pady = size - img.shape[1]%size
  else:
    pady:(i+1) = 0
  img1 = np.zeros((img.shape[0]+padx, img.shape[1]+pady, 3), dtype=np.float16)
  shape = img1.shape
  img1[:img.shape[0], :img.shape[1], :] = img.astype(np.float16)
  patches = img1.reshape(img1.shape[0]//size, size, img1.shape[1]//size, size, 3).swapaxes(1,2).reshape(-1,size,size,3)
  patches = patches.transpose((0,3,1,2))
  return patches, shape

def get_img(img, shape, gtshape, size):
  img = img.transpose((0,2,3,1))
  img = img.reshape(shape[0]//size, shape[1]//size, size, size, 3).swapaxes(1,2)
  img = img.reshape(shape[0], shape[1], 3)#.astype(np.uint8)
  img = img[:gtshape[0], :gtshape[1], :]
  return img 

