from model import Unet
import torch
from dataset import MdDataset,get_trans
from torch.utils.data import DataLoader
from train import train,test
import matplotlib.pyplot as plt
import numpy as np


def main(root_dir,epoch,batch_size,learning_rate,patience,pt_path=None):
    '''
    :param root_dir: the folder of datas
    :param epoch: the trianing times
    :param batch_size: the num of pic for training each time
    :param learning_rate: learnning rate
    :param patience: early stop patience
    :return: trained weights will be save as 'dic.pkl'
    '''
    trainDatas = MdDataset(root_dir, get_trans(), 'train')
    valDatas = MdDataset(root_dir, get_trans(), 'val')
    train_loader = DataLoader(trainDatas, batch_size=batch_size)
    val_loader = DataLoader(valDatas, batch_size=batch_size)

    net = Unet(1)
    if pt_path:
        dic = torch.load(pt_path)
        net.load_state_dict(dic)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    histories = {}
    histories['losses'] = {}
    histories['dscs'] = {}
    histories['losses']['train_loss'] = []
    histories['losses']['val_loss'] = []
    histories['dscs']['train_dsc'] = []
    histories['dscs']['val_dsc'] = []
    best_dsc = 0.0
    best_dict = net.state_dict().copy()
    best_tmp = None
    patience_c = 0
    for epoch_ in range(1, epoch + 1):
        train_loss, train_dsc = train(train_loader, optimizer, net, device,epoch_,epoch+1)
        print('\n', f'epoch: {epoch_} train_loss: {round(train_loss, 2)} train_dsc: {round(train_dsc, 2)} ')
        val_loss, val_dsc,val_rec,val_f1, tmp = test(val_loader, net, device)
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            best_dict = net.state_dict().copy()
            best_tmp = tmp
            patience_c = 0
        else:
            patience_c += 1

        if patience < patience_c:
            print('\n',f'training {epoch_} epoches,and the best dsc is {round(best_dsc,2)}')
            break

        # val_dsc = best_dsc
        histories['losses']['train_loss'].append(train_loss)
        histories['losses']['val_loss'].append(val_loss)
        histories['dscs']['train_dsc'].append(train_dsc)
        histories['dscs']['val_dsc'].append(val_dsc)

        print(f'epoch: {epoch_} - val_loss: {round(val_loss, 2)} val_dsc: {round(val_dsc, 2)} val_rec:{round(val_rec,2)} val_f1:{round(val_f1,2)}')
    torch.save(best_dict, 'dic.pkl')

    for hist in histories.keys():
        plt.figure()
        for it in histories[hist].keys():
            plt.plot(histories[hist][it], label=it)
            plt.ylabel(hist)
            plt.xlabel('epoch')
            plt.ylim(0, 1)
            plt.legend()
        plt.savefig(hist)
        plt.show()

    or_imgs, out_puts, masks = best_tmp
    n_pics = or_imgs.shape[0]
    plt.figure()
    for i in range(n_pics):
        or_img = or_imgs[i, :, :, :]
        out_put = out_puts[i, :, :, :]
        mask = masks[i, :, :, :]
        plt.subplot(n_pics,3,  i*3+1)
        img = np.squeeze(or_img.cpu().numpy())
        img = np.transpose(img, [1, 2, 0])
        plt.axis('off')
        plt.imshow(img)
        if i==0:
            plt.title('image')

        plt.subplot( n_pics,3, i*3+2)
        plt.imshow(np.squeeze(mask.cpu().numpy()), cmap='gray')
        if i==0:
            plt.title('GT')
        plt.axis('off')

        plt.subplot(n_pics,3,  i*3+3)
        op = np.squeeze(out_put.cpu().numpy())
        op = (op>0.5)*1
        plt.imshow(op, cmap='gray')
        if i==0:
            plt.title('predict')
        plt.axis('off')

    plt.savefig('val.png')
    plt.show()

if __name__ == '__main__':
    # Load data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root_dir = 'data'
    epoch = 100
    patience = 10
    batch_size = 6
    learning_rate = 0.01
    pt_path = 'dic.pkl'
    # ====================
    main(root_dir, epoch, batch_size, learning_rate, patience,pt_path=None)
    torch.cuda.empty_cache()





