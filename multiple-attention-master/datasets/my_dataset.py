import os

from torch.utils.data import Dataset
import cv2
from datasets.augmentations import augmentations

class ForgeDataset(Dataset):
    def __init__(self,root_path,phase='phase1',usage='train',augment='augment0'):
        assert usage in ['train','val','test']
        self.usage = usage
        self.phase = phase
        self.data_root_path = os.path.join(root_path,phase)
        self.aug = augmentations[augment]
        self.label_txt_path = "../data/phase1/{}set_label.txt".format(self.usage)
        if usage != 'test':
            with open(self.label_txt_name,'r') as f:
                self.dataset = [line.split(' ') for line in f]

        pass

    def __getitem__(self, item):
        if self.type != 'test':
            img_name, label = self.dataset[item]
            img_data_path = os.path.join(self.data_root_path,'trainset_sample') if self.type=='train' else \
                os.path.join(self.data_root_path,'valset')
            image = cv2.imread(os.path.join(img_data_path,img_name+'.jpg')) # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = self.aug(image=image)['image']
            return image,label
        else:
            return -1