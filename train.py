from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from model import dsc
from sklearn.metrics import recall_score,f1_score


def metrics(predics,gts):
  predics = (predics.view(-1,1)>0.5)*1
  gts = gts.view(-1,1)
  predics = predics.cpu().numpy()
  gts = gts.cpu().numpy()
  recall = recall_score(predics,gts)
  f1 = f1_score(gts,predics)
  return recall,f1




def train(train_loader, optimizer,model,device,epoch,epoches):
  running_loss = 0.0
  running_dsc = 0.0
  data_size = len(train_loader)
  model.to(device)
  model.train()
  dataIter = tqdm(train_loader,desc="Training...",
                  bar_format="{l_bar}{r_bar}",
                  dynamic_ncols=True)
  for i,data in enumerate(dataIter):
    inputs, masks = data
    inputs, masks= inputs.to(device), masks.to(device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):

      outputs = model(inputs)
      # print(masks.shape,outputs.shape)
      # backward
      loss = nn.BCELoss()(outputs.squeeze(1), masks.squeeze(1).float())
      acc = dsc(outputs,masks)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      running_dsc += acc.item()

      dataIter.set_description(f'Training-epoch:{epoch}/{epoches}-{i+1}/{data_size}-loss:{round(loss.item(),2)}-dsc:{round(acc.item(),2)}')

  epoch_loss = running_loss / data_size
  epoch_dsc = running_dsc /data_size
  return epoch_loss,epoch_dsc

def test(test_loader, model,device):
  running_loss = 0.0
  running_dsc=0.0
  running_rec = 0.0
  running_f1 = 0.0

  data_size = len(test_loader)
  model.to(device)
  model.eval()

  for i, data in tqdm(enumerate(test_loader)):
    inputs, masks = data
    inputs, masks = inputs.to(device), masks.to(device)
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      loss = nn.BCELoss()(outputs.squeeze(1), masks.squeeze(1).float())
      acc = dsc(outputs, masks)
      running_loss += loss.item()
      running_dsc += acc.item()
      rec,f1 = metrics(outputs,masks)
      running_rec += rec
      running_f1 +=f1

  epoch_loss = running_loss / data_size
  epoch_dsc = running_dsc / data_size
  epoch_rec = running_rec/data_size
  epoch_f1 = running_f1/data_size
  return epoch_loss,epoch_dsc,epoch_rec,epoch_f1,(inputs,outputs,masks)

def rle_encode(im):
  '''
  im: numpy array, 1 - mask, 0 - background
  Returns run length as string formated
  '''
  pixels = im.flatten(order = 'F')
  pixels = np.concatenate([[0], pixels, [0]])
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
  runs[1::2] -= runs[::2]
  return ' '.join(str(x) for x in runs)