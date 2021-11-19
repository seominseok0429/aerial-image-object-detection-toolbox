from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from PIL import Image, ImageFilter
import glob

LABEL = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 'harbor', 'bridge',
        'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']

class DOTA_loader(Dataset):
    def __init__(self, is_train=1):
        self.is_train = is_train
        if is_train == 1:
            self.img_path = glob.glob('./DATA/train/images/*')
            self.label_path = './DATA/train/label'
        else:
            self.img_path = glob.glob('./DATA/val/images/*')

    def __len__(self):
        return len(self.img_path)

    def __randomResize__(self, img, label, h, w):
        if h < 600 or w < 600:
            r_h = 600
            r_w = 600
        else:
            r_h = np.random.randint(600, h)
            r_w = np.random.randint(600, w)
        img = cv2.resize(img, dsize=(r_w,r_h))
        new_label = []

        for i in label:
            xmin, xmax, ymin, ymax = (i[0]/w)*r_w, (i[1]/w)*r_w, (i[2]/h)*r_h, (i[3]/h)*r_h
            new_label.append([xmin, xmax, ymin, ymax, i[4]])
        return img, new_label

    def __randomCrop__(self, img, label):
        h, w, c = img.shape
        if h ==600:
            return img, label

        c_h = np.random.randint(0, h-600)
        c_w = np.random.randint(0, w-600)
        img = img[c_h:c_h+600, c_w:c_w+600, :]
        new_label = []

        for i in label:
            if i[0] < c_w or i[1] > c_w+600 or i[2] < c_h or i[3] > c_h + 600:
                pass
            else:
                new_label.append([i[0]-c_w, i[1]-c_w, i[2]-c_h, i[3]-c_h, i[4]])

        return img, new_label

    def __txt2anno__(self, path, h, w):
        path = os.path.join(self.label_path, path[:-3]+'txt')

        with open(path, 'r') as f:
            txt_label = f.readlines()

        label = []

        for line in txt_label[2:]:
            lines = line[:-1].split(' ')
            x1, y1, x2, y2, x3, y3, x4, y4 = [max(float(point), 0) for point in lines[:-2]]
            x1, y1, x2, y2 = min(x1, w - 1), min(y1, h - 1), min(x2, w - 1), min(y2, h - 1)
            x3, y3, x4, y4 = min(x3, w - 1), min(y3, h - 1), min(x4, w - 1), min(y4, h - 1)
            xmin, xmax, ymin, ymax = max(min(x1, x2, x3, x4), 0), max(x1, x2, x3, x4), max(min(y1, y2, y3, y4), 0), max(y1, y2, y3, y4)
            label.append([xmin, xmax, ymin, ymax, LABEL.index(lines[8])])

        return label

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        h, w, c = img.shape
        label = self.__txt2anno__(os.path.basename(self.img_path[idx]), h, w)
        img, label = self.__randomResize__(img, label, h, w)
        img, label = self.__randomCrop__(img, label)

        return img, label

if __name__ == '__main__':
    import torch
    loader = DOTA_loader()
    #loader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=True, num_workers=2)
    for idx, (img, label) in enumerate(loader):
