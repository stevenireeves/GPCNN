from math import log10
import os
import random 

import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from data_loader import get_train_obj, get_batch, get_valid_obj


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH_GP = '/home/steven/dissertation/GP_image/data/GP/GP4/'
TRAIN_PATH_GT = '/home/steven/dissertation/GP_image/data/train_HR/'
VALID_PATH_GP = '/home/steven/dissertation/GP_image/data/GP/GP4_valid/'
VALID_PATH_GT = '/home/steven/dissertation/GP_image/data/valid_HR/'
print('===> Building model')
model = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
batch_size = 128


def train(epoch):
    epoch_loss = 0
    direc = os.listdir(TRAIN_PATH_GP)
    random.shuffle(direc)
    for idy, fil in enumerate(direc):
      patches_train, patches_gt = get_train_obj(TRAIN_PATH_GP, TRAIN_PATH_GT, fil)
      m = len(patches_train)//batch_size
      ids = random.sample(range(m), m)
      for i in range(m):
        idx = ids[i]
        train_x, train_y = get_batch(patches_train, patches_gt, batch_size, idx)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
      print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, idy, len(direc), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss /(len(direc)*m*batch_size)))


def test():
    avg_psnr = 0
    direc = os.listdir(VALID_PATH_GP)
    with torch.no_grad():
        for fil in direc:
          patches_valid, patches_gt = get_valid_obj(VALID_PATH_GP, VALID_PATH_GT, fil)
          m = len(patches_valid)//batch_size
          for i in range(m):
            test_x, test_y = get_batch(patches_valid, patches_gt, batch_size, i)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            prediction = model(test_x)
            mse = criterion(prediction, test_y)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr /(len(direc)*m*batch_size)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(25):
    train(epoch)
    test()
    checkpoint(epoch)
