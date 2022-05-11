from PIL import Image
from sklearn.utils import shuffle
from torch.utils import data
import torch
import numpy as np


class DataGenerator(data.Dataset):

    def __init__(self, anno_file, transform=None, type=None, neg_prop=0):
        super().__init__()
        imgs_path, imgs_label = [], []
        with open(anno_file, 'r') as f:
            ls = f.readlines()
            for l in ls:
                path, cls = l.strip('\n').split(' ')
                imgs_path.append(path)
                imgs_label.append(int(cls))
        imgs_path = np.array(imgs_path)
        imgs_label = np.array(imgs_label)
        if type in ['train', 'val'] and neg_prop != 0:
            pos_keep = np.where(imgs_label == 1)[0]
            neg_keep = shuffle(np.where(imgs_label == 0)[0])[
                :len(pos_keep)*neg_prop]
            keep = np.concatenate([pos_keep, neg_keep])
            keep = shuffle(keep)
            imgs_path = imgs_path[keep]
            imgs_label = imgs_label[keep]
            print(type, len(pos_keep), len(neg_keep))
        self.imgs_path = imgs_path
        self.imgs_label = imgs_label
        self.transform = transform

    def get_weighted(self):
        w = []
        for label in self.imgs_label:
            if label == 1:
                w.append(5)
            else:
                w.append(1)
        return w

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.imgs_label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
