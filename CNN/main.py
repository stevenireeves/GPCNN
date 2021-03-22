import os
import random 

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as psnr

from model import Net
from data_loader import get_train_objs, get_test_objs, get_batch, get_img, get_test_image

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
BASE = '/dockerx/img_data/'
TRAIN_PATH_GP = BASE +'train_d4/'
TRAIN_PATH_GT = BASE + 'train_HR/'
VALID_PATH_GP = BASE + 'valid_d4/'
VALID_PATH_GT = BASE + 'test_HR/'
print('===> Building model')
model = Net() 
#model._initialize_weights()
#model = torch.load('models2/model_epoch_40.pth')
if torch.cuda.device_count()>1:
  model = nn.DataParallel(model, device_ids=[0,1]).to(device)
else:
  model = model.to(device)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.001) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
batch_size = 32 #400
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
      loss.backward()
      optimizer.step()
      if(idy%100==0):
        if(idy > 0):
          batch_loss/=100 
        print("===> Epoch[{}], round[{}]({}/{}): Batch Loss: {:.8f}".format(epoch, i, idy, m, batch_loss))
        batch_loss = 0
      if(idy==m-1):
        batch_loss/=(m%100)
        print("===> Epoch[{}], round[{}]({}/{}): Batch Loss: {:.8f}".format(epoch, i, idy, m, batch_loss))
    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss/(m)))
    training_loss.append(epoch_loss/(m))

def test(valid_gp, valid_gt):
    avg = 0
    print('------------------ Validation----------------------')
    model.eval()
    with torch.no_grad():
      for j in range(len(valid_gt)):
        test_x, shape = get_test_image(valid_gp[j])
        test_x = torch.from_numpy(test_x).to(device)
        for i in range(len(test_x)): 
          prediction = model(test_x[i:i+1])
          test_x[i] = test_x[i] + prediction
        img = get_img(test_x.data.cpu().numpy(), shape, valid_gp[j].shape)
        crit = psnr(valid_gt[j], img)
        avg += crit
#        print("PSNR ---> {:.8f}".format(crit))
    avg /= len(valid_gt)
    print("===> Avg. PSNR : {:.8f}".format(avg))
    validation_loss.append(avg)


def checkpoint(epoch, check):
    if(epoch%check==0):
      model_out_path = 'models/'+"model_epoch_{}.pth".format(epoch)
      torch.save(model, model_out_path)
      print("Checkpoint saved to {}".format(model_out_path))

print('Reading in Data Set')
train_obj, gt_obj = get_train_objs(TRAIN_PATH_GP, TRAIN_PATH_GT)
gp_imgs, gt_imgs = get_test_objs(VALID_PATH_GP, VALID_PATH_GT, num)
print('Data Set Loaded')
#means = np.zeros(train_obj[0].shape, dtype=np.float32) # train_obj.mean(axis=0).astype(np.float32) 
for epoch in range(50):
    for i in range(2): 
#train
      indices = np.arange(len(train_obj))
      np.random.shuffle(indices)
      train_obj = train_obj[indices]
      gt_obj = gt_obj[indices]
      train(epoch, train_obj, gt_obj, i)
#test
    test(gp_imgs, gt_imgs)
    checkpoint(epoch, 1)
    if(epoch>1 and epoch%40==0):
      scheduler.step()
#checkpoint(epoch, 1)

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
training_loss.flatten.save('training_loss.out')
validation_loss.flatten.save('validation_loss.out')
