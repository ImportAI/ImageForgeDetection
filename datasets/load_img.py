import sys, os
sys.path.append('..')
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def read_txt(root):
    img_label = []
    with open(root, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            img_label.append(line)
        return img_label


class face_dataset(Dataset):
    def __init__(self, img_label_root, img_root, transform=None):
        self.img_label = read_txt(img_label_root)  # list [[],[],...[]]
        self.imgs = [os.path.join(img_root, self.img_label[i][0]+'.jpg') for i in range(len(self.img_label))]
        self.labels = [int(self.img_label[i][1]) for i in range(len(self.img_label))]
        self.transform = transform
    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        plt.imshow(img)
        plt.show()
        label = torch.tensor(self.labels[index])
        if self.transform:
            img = self.transform(img)
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.475, 0.451, 0.390], std=[0.261, 0.255, 0.259])])
            img = self.transform(img)

        return img, label

if __name__ == '__main__':
    train_txt_root = '../data/phase1/trainset_label.txt'
    train_img_root = '../data/phase1/trainset'
    # train_img_label = read_txt(train_txt_root)
    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.475, 0.451, 0.390], std=[0.261, 0.255, 0.259]),
                                    ])
    train_dataset = face_dataset(train_txt_root, train_img_root)
    train_iter = DataLoader(train_dataset, batch_size=4, shuffle=True)

    for img, label in train_iter:
        print(img.size, label.size)
    print(1)
