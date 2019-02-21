import os

import numpy as np
import torch
from PIL import Image


if os.name == "posix":
    path = "/home/hndx/projects/TGS_Salt_Identification/data/"
    train_img_path = "{}train/images/".format(path)
    train_msk_path = "{}train/masks/".format(path)
    test_img_path = "{}test_images/".format(path)
else:
    path = "D:\\data\\TGS\\"
    depths = "{}depths.csv".format(path)

    train_img_path = "{}train\\images\\".format(path)
    train_msk_path = "{}train\\masks\\".format(path)
    test_img_path = "{}test\\images\\".format(path)


def train_loader(batch, shuffle=False):
    """ 返回 batch * (img, msk) """
    file_names = list(os.walk(train_img_path))[0][2]
    if shuffle:
        np.random.shuffle(file_names)
    i, num = 0, len(file_names)
    while i < num:
        imgs, msks = [], []
        for ii in range(i, min(i + batch, num)):
            name = file_names[ii]
            img = np.array(Image.open("{}{}".format(train_img_path, name)))
            msk = np.array(Image.open("{}{}".format(train_msk_path, name)))
            imgs.append(img)
            msks.append(msk)
        i += batch
        imgs = np.array(imgs).transpose(0, 3, 2, 1)
        msks = np.array(msks).transpose(0, 2, 1) / 65535
        imgs_tensor = torch.from_numpy(imgs).view(len(msks), 3, 101, 101).float()
        msks_tensor = torch.from_numpy(msks).view(len(msks), 1, 101, 101).float()
        yield imgs_tensor, msks_tensor


def test_loader(batch):
    """ 返回 img """
    file_names = list(os.walk(test_img_path))[0][2]
    i, num = 0, len(file_names)
    while i < num:
        names, imgs = [], []
        for ii in range(i, min(num, i + batch)):
            name = file_names[ii]
            names.append(name.split(".")[0])
            img = np.array(Image.open("{}{}".format(test_img_path, name)))
            imgs.append(img)
        i += batch
        imgs = np.array(imgs).transpose(0, 3, 2, 1)
        imgs_tensor = torch.from_numpy(imgs).view(len(names), 3, 101, 101).float()
        yield names, imgs_tensor
