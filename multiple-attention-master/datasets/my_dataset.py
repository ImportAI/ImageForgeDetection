import os
import time

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch


class ForgeDataset(Dataset):
    def __init__(self, root_path, phase='phase1', usage='train', augment='augment0',
                 normalize=None):
        if normalize is None:
            normalize = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalize = normalize
        assert usage in ['train', 'val', 'test']
        self.epoch = 0
        self.usage = usage
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalize)])
        self.phase = phase
        self.data_root_path = os.path.join(root_path, phase)
        # self.aug = augmentations[augment]
        if self.usage == 'train':
            # self.label_txt_path = os.path.join(self.data_root_path, "{}set_sample_label.txt".format(self.usage))
            self.label_txt_path = os.path.join(self.data_root_path, "{}set_sample_label.txt".format(self.usage))
        elif self.usage == 'val':
            self.label_txt_path = os.path.join(self.data_root_path, "{}set_label.txt".format(self.usage))
        else:
            self.label_txt_path = os.path.join(self.data_root_path,'valset2_nolabel')
        if usage != 'test':
            with open(self.label_txt_path, 'r') as f:
                self.dataset = [line.strip().split(' ') for line in f]
        else:
            with open(self.label_txt_path, 'r') as f:
                self.dataset = [line.strip() for line in f]
        self.dataset = self.dataset[:8]

    def read_img(self,img_path):
        try:
            # img_name = '1'
            image = Image.open(img_path)
            image = np.asarray(image)
            # image = self.aug(image=image)['image']
            image = self.transform(image.copy())
            # print(type(image))
        except Exception as e:
            print('---------')
            return torch.zeros((3, 512, 512)),False
        else:
            return image,True

    def __getitem__(self, item):
        if self.usage != 'test':
            img_name, label = self.dataset[item]
            img_data_path = os.path.join(self.data_root_path, 'trainset_sample') if self.usage == 'train' else \
                os.path.join(self.data_root_path, 'valset')
            # image = cv2.imread(os.path.join(img_data_path, img_name + '.jpg'))[:, :, ::-1] # RGB
            image, flag = self.read_img(os.path.join(img_data_path, img_name + '.jpg'))
            if flag:
                return image, int(label),img_name
            else:
                return image, -1,img_name
        else:
            img_name = self.dataset[item]
            img_data_path = os.path.join(self.data_root_path, '%sset' % self.usage)
            image,flag = self.read_img(os.path.join(img_data_path, img_name + '.jpg'))
            return image,img_name

    @staticmethod
    def collate_fn(batch):
        # print(batch)
        batch = [_ for _ in batch if _[1] != -1]
        img, label, img_name = zip(*batch)
        return torch.stack(img), torch.LongTensor(label),img_name

    @classmethod
    def test(cls):
        print('bb')

    def __len__(self):
        return len(self.dataset)

    def check_dataset(self):
        pass

    def next_epoch(self):
        self.epoch += 1


if __name__ == '__main__':
    # data_root_path = os.path.abspath('../../data')
    # train_dataset = ForgeDataset(data_root_path, usage='val')
    # train_loader = DataLoader(train_dataset, batch_size=5, collate_fn=ForgeDataset.collate_fn)
    # for img, label,img_name in train_loader:
    #     # print(img)
    #     print(img.shape)
    #     print(label)
    #     print(label.shape)
    #     print(img_name)
    #     break

    # a = torch.zeros((1,3,512,512))
    # print((torch.sum(a)==0).bool())
    # if (torch.sum(a)==0).bool():
    #     print('aa')
    # ForgeDataset.collate_fn()
    # ForgeDataset.test()
    import pandas as pd
    # a = [[1,2,3],['a','b','c']]
    # df_a = pd.DataFrame(a)
    # print(df_a)
    #
    # a = dict().fromkeys(['a','b'],[])
    # print(a)

    # df_result = pd.read_csv('../evaluations/test(3).txt',sep=' ',names=['name','preds'])
    # print(len(df_result))
    # df_result.drop_duplicates(subset=['name'],inplace=True)
    # print(len(df_result))

    a = torch.tensor([[1.0,-2.0],[-3.0,4.0]])
    b = torch.nn.functional.softmax(a,dim=1)
    print(b)
    print(b[:,1])
    print(torch.max(a,dim=1))

    with open('test.csv','a+') as f:
        print(f.readlines())




