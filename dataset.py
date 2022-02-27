import os
from glob import glob
from random import shuffle
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
ipt_size = (64,64)


def get_datas(root_dir):
    imgs = glob(os.path.join(root_dir,'img','*.jpg'))
    gts = glob(os.path.join(root_dir,'gt','*.jpg'))
    datas = [[im,gt] for im,gt in zip(imgs,gts)]
    shuffle(datas)
    return datas


class SizeCheck():
    def __init__(self):
        self.__name__="size_check"

    def __call__(self,image):
        if image.shape[0]==1:
            image = image.repeat(3,1,1)
        return image

def get_trans():
    transform = T.Compose([
        T.Resize(ipt_size),
        T.PILToTensor(),
        SizeCheck(),
    ])
    return transform


class ToOne():
    def __init__(self):
        self.name='ToOne'
    def __call__(self,image):
        thresh = 0.1
        image = image>thresh
        image = image*1
        return image


class MdDataset(Dataset):
    def __init__(self,root_dir,trans,mode='train'):
        self.datas = get_datas(root_dir)
        self.trans = trans
        self.mode = mode
        self.num = int(0.8*len(self.datas))
        self.trainds = self.datas[:self.num]
        self.valds = self.datas[self.num:]

        self.vt = T.Compose([
            T.Resize(ipt_size),
            T.PILToTensor(),
            ToOne(),
        ])
    def __len__(self):
        if self.mode =='train':
            return self.num
        else:
            return len(self.datas)-self.num

    def __getitem__(self, item):
        item = item%self.__len__()
        if self.mode=='train':
            img,lb = self.trainds[item]
        else:
            img, lb = self.valds[item]
        img = Image.open(img)
        gt = Image.open(lb)

        img = self.trans(img)
        gt = self.vt(gt)
        return img,gt

if __name__ == '__main__':
    trans = get_trans()
    datas = MdDataset('data',trans,'train')
    img,lb = datas[0]
    print(img.shape,lb.shape)
    plt.figure()
    plt.imshow(np.squeeze(lb)>0,cmap='gray')
    plt.show()



