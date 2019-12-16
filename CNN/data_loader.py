import torch
import cv2
import numpy as np

def get_train_obj(path_ds, path_gt, img):
  train_img = cv2.imread(path_ds+img)
  padx = 64 - train_img.shape[0]%64 if train_img.shape[0]%64 >0 else 0
  pady = 64 - train_img.shape[1]%64 if train_img.shape[1]%64 >0 else 0
  img1 = np.zeros((train_img.shape[0]+int(padx), train_img.shape[1]+int(pady), train_img.shape[2]), dtype = np.float32)
  img1[:train_img.shape[0], :train_img.shape[1], :train_img.shape[2]] = train_img
  gt_img = cv2.imread(path_gt+img)
  patches_train = img1.reshape(img1.shape[0]//64, 64, img1.shape[1]//64, 64, 3).swapaxes(1,2).reshape(-1,64,64,3)
  patches_train = patches_train.transpose((0,3,1,2))
  img2 = np.zeros((train_img.shape[0]+int(padx), train_img.shape[1]+int(pady), train_img.shape[2]), dtype = np.float32)
  img2[:train_img.shape[0], :train_img.shape[1], :train_img.shape[2]] = gt_img
  patches_gt = img2.reshape(img2.shape[0]//64, 64, img2.shape[1]//64, 64, 3).swapaxes(1,2).reshape(-1,64,64,3)
  patches_gt = patches_gt.transpose((0,3,1,2))
  return patches_train, patches_gt

def get_batch(train_in, train_gt, batch_size, shift):
  train_x = torch.from_numpy(train_in[shift*batch_size:(shift+1)*batch_size])
  train_y = torch.from_numpy(train_gt[shift*batch_size:(shift+1)*batch_size])
  return train_x, train_y

def get_valid_obj(path_ds, path_gt, img):
  test_img = cv2.imread(path_ds+img)
  padx = 64 - test_img.shape[0]%64 if test_img.shape[0]%64 >0 else 0
  pady = 64 - test_img.shape[1]%64 if test_img.shape[1]%64 >0 else 0
  img1 = np.zeros((test_img.shape[0]+int(padx), test_img.shape[1]+int(pady), test_img.shape[2]), dtype = np.float32)
  img1[:test_img.shape[0], :test_img.shape[1], :test_img.shape[2]] = test_img
  gt_img = cv2.imread(path_gt+img)
  patches_test = img1.reshape(img1.shape[0]//64, 64, img1.shape[1]//64, 64, 3).swapaxes(1,2).reshape(-1,64,64,3)
  patches_test = patches_test.transpose((0,3,1,2))
  img2 = np.zeros((test_img.shape[0]+int(padx), test_img.shape[1]+int(pady), test_img.shape[2]), dtype = np.float32)
  img2[:test_img.shape[0], :test_img.shape[1], :test_img.shape[2]] = gt_img
  patches_gt = img2.reshape(img2.shape[0]//64, 64, img2.shape[1]//64, 64, 3).swapaxes(1,2).reshape(-1,64,64,3)
  patches_gt = patches_gt.transpose((0,3,1,2))
  return patches_test, patches_gt

