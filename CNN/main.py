import os
import random 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler 

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

from model2 import Net
from data_loader import get_train_objs, get_test_objs, get_batch, get_img, get_test_image

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
BASE = '/dockerx/img_data/'
TRAIN_PATH_GP = BASE +'train/d4/'
TRAIN_PATH_GT = BASE + 'train/gt/'
VALID_PATH_GP = BASE + 'valid/d4/'
VALID_PATH_GT = BASE + 'valid/gt/'
SIZE=64 #Images cut into SIZExSIZE patches
print('===> Building model')
model = Net() 
model._initialize_weights()
model = torch.load('models_2/model_epoch_100.pth')
load=True
if not load:
  if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7]).to(device)
  else:
    model = model.to(device)
#model.load_state_dict(torch.load('models/model_epoch_40.pth'))
criterion = nn.L1Loss()
#optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scaler = GradScaler()
optimizer = optim.Adam(model.parameters(), lr=5e-5) 
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
batch_size = 1024 #400
training_loss = []
validation_loss = []
num = 10
def train(epoch, train_obj, gt_obj, i):
    model.train()
    epoch_loss = 0
    m = len(train_obj)//batch_size
#shuffle training set
    batch_loss = 0
    for idy in range(m):
      train_x, train_y = get_batch(train_obj, gt_obj, batch_size, idy, True)
      optimizer.zero_grad()
      train_x = train_x.to(device)
      train_y = train_y.to(device)
      loss = criterion(model(train_x), train_y)
      epoch_loss += loss.item()
      batch_loss += loss.item()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      if(idy%10==0):
        if(idy > 0):
          batch_loss/=10 
        print("===> Epoch[{}], round[{}]({}/{}): Batch Loss: {:.8f}".format(epoch, i, idy, m, batch_loss))
        batch_loss = 0
      if(idy==m-1):
        batch_loss/=(m%100)
        print("===> Epoch[{}], round[{}]({}/{}): Batch Loss: {:.8f}".format(epoch, i, idy, m, batch_loss))
    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss/(m)))
    training_loss.append(epoch_loss/(m))

def test(valid_gp, valid_gt):
    avg = 0
    gp_av = 0
    print('------------------ Validation----------------------')
    model.eval()
    with torch.no_grad():
      for j in range(len(valid_gt)):
        test_x, shape = get_test_image(valid_gp[j], SIZE)
        test_x = torch.from_numpy(test_x).to(device)
        for i in range(len(test_x)): 
          prediction = model(test_x[i:i+1])
          test_x[i] = test_x[i] + prediction
        img = get_img(test_x.data.cpu().numpy(), shape, valid_gp[j].shape, SIZE)
        crit = psnr(valid_gt[j], img)
        gp_t = psnr(valid_gt[j], valid_gp[j])
        gp_av += gp_t
        avg += crit
        if(j%50==0):
            print("PSNR ---> GPAEN {:.8f}, GP {:.8f}".format(crit, gp_t))
    avg /= len(valid_gt)
    gp_av /= len(valid_gt)
    print("===> Avg. PSNR : {:.8f}".format(avg))
    print("===> Avg. GP: {:.8f}".format(gp_av))
    validation_loss.append(avg)


def checkpoint(epoch, check):
    if(epoch%check==0):
      model_out_path = 'models_2/'+"model_epoch_{}.pth".format(epoch)
      torch.save(model, model_out_path)
      print("Checkpoint saved to {}".format(model_out_path))

print('Reading in Data Set')
train_obj, gt_obj = get_train_objs(TRAIN_PATH_GP, TRAIN_PATH_GT, SIZE)
gp_imgs, gt_imgs = get_test_objs(VALID_PATH_GP, VALID_PATH_GT, num, SIZE)
print('Data Set Loaded')
#means = np.zeros(train_obj[0].shape, dtype=np.float32) # train_obj.mean(axis=0).astype(np.float32) 
for epoch in range(101,150):
    for i in range(1): 
#train
      indices = np.arange(len(train_obj))
      np.random.shuffle(indices)
      train_obj = train_obj[indices]
      gt_obj = gt_obj[indices]
      train(epoch, train_obj, gt_obj, i)
#test
    test(gp_imgs, gt_imgs)
    checkpoint(epoch, num)
#    if(epoch>1 and epoch%40==0):
#      scheduler.step()
checkpoint(epoch, 1)

training_loss = np.asarray(training_loss)
validation_loss = np.asarray(validation_loss)
plt.subplot(2,1,1)
plt.plot(np.arange(len(training_loss)), training_loss, 'k-')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.subplot(2,1,2)
plt.plot(np.arange(len(validation_loss)), validation_loss, 'k-')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.savefig('loss.png')
#plt.show()
#training_loss.flatten.save('training_loss.out')
#validation_loss.flatten.save('validation_loss.out')
