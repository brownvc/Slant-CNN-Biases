import math
import os.path

import cv2
from glob import glob
import numpy as np
import re
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class load_stimuli_mono(Dataset):
    """ Load sine wave stimuli (monocular). Data and texture files are not matched. """

    def __init__(self, datadir, texturedir, shape=256, channel=3):

        self.data_files = glob(os.path.join(datadir, "*png"))
        self.texture_files = glob(os.path.join(texturedir, "*png"))
        assert len(self.data_files), "No image file given to ImageFromFile!"
        assert len(self.texture_files), "No texture file given to ImageFromFile!"
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        # self.shuffle = shuffle
        self.indexes = list(range(len(self.data_files)))
        self.shape = shape

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     np.random.shuffle(self.indexes)

        df = self.data_files[idx]
        tf = self.texture_files[np.random.randint(0, len(self.texture_files))]

        filename = os.path.basename(df)
        depth = float(filename.split("_")[1])
        texture_nb = int(filename.split("_")[3].split(".")[0])

        # f = os.path.join(self.main_dir, filename)
        img = cv2.imread(df, self.imread_mode)
        texture = cv2.imread(tf, self.imread_mode)

        assert img is not None, df
        assert texture is not None, tf

        img = cv2.resize(img, (self.shape, self.shape))
        texture = cv2.resize(texture, (self.shape, self.shape))
        img = np.transpose(img, (2, 0, 1))
        texture = np.transpose(texture, (2, 0, 1))

        return img, texture, depth, texture_nb


class LoaderSimple(Dataset):
    def __init__(self, img_res=(256, 256), dataset_path="./dataset", is_testing=False):
        # self.dataset_name = dataset_name
        self.img_res = img_res
        self.dataset_path = dataset_path

        self.data_type = "train" if not is_testing else "test"
        self.img_paths = glob(os.path.join(self.dataset_path, '%s/*' % self.data_type))
        assert len(self.img_paths) > 0, "No image read in dataloader!"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     np.random.shuffle(self.indexes)

        img = self.img_paths[idx]
        im = cv2.imread(img)
        im = cv2.resize(im, self.img_res)
        im = im / 255.  # range 0 to 1
        im = np.transpose(im, (2, 0, 1))
        info = img.split(".png")[0].split('/')[-1].split("_")
        texture_nb = int(info[2])
        fov = float(info[4])
        optical_slant = float(info[7])
        physical_slant = float(info[10])
        convexity = 0 if info[0] == 'concave' else 1

        return im, texture_nb, fov, optical_slant, physical_slant, convexity


class LoaderDotSizeVar(Dataset):
    def __init__(self, img_res=(256, 256), dataset_path="./dataset", is_testing=False, nb_chan=1):
        # self.dataset_name = dataset_name
        self.img_res = img_res
        self.dataset_path = dataset_path
        self.nb_chan = nb_chan

        self.data_type = "train" if not is_testing else "test"
        self.img_paths = glob(os.path.join(self.dataset_path, '%s/*' % self.data_type))
        assert len(self.img_paths) > 0, "No image read in dataloader!"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     np.random.shuffle(self.indexes)

        img = self.img_paths[idx]
        if self.nb_chan == 1:
            im = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, self.img_res)
            im = im[None, ...]
        else:
            im = cv2.imread(img)
            im = cv2.resize(im, self.img_res)
            im = np.transpose(im, (2, 0, 1))
        im = im / 127.5 - 1.  # range -1 to 1
        info = img.split(".png")[0].split('/')[-1].split("_")
        texture_nb = int(info[2])
        fov = float(info[4])
        optical_slant = float(info[7])
        physical_slant = float(info[10])
        if len(info) > 13:
            size_var = int(info[13])
        else:
            size_var = 0
        convexity = 0 if info[0] == 'concave' else 1

        return im, texture_nb, fov, optical_slant, physical_slant, convexity, size_var


class LoaderDotSizeVarPerVar(Dataset):
    def __init__(self, img_res=(224, 224), dataset_path="./dataset", is_testing=False, nb_chan=1, var_level='1'):
        # self.dataset_name = dataset_name
        self.img_res = img_res
        self.dataset_path = dataset_path
        self.nb_chan = nb_chan
        self.var_level = var_level

        self.data_type = "train" if not is_testing else "test"
        self.img_paths = glob(os.path.join(self.dataset_path, '%s/*_var_loc_%s.png' % (self.data_type, self.var_level)))
        assert len(self.img_paths) > 0, "No image read in dataloader!"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     np.random.shuffle(self.indexes)

        img = self.img_paths[idx]
        if self.nb_chan == 1:
            im = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, self.img_res)
            im = im[None, ...]
        else:
            im = cv2.imread(img)
            im = cv2.resize(im, self.img_res)
            im = np.transpose(im, (2, 0, 1))
        im = im / 127.5 - 1.  # range -1 to 1
        info = img.split(".png")[0].split('/')[-1].split("_")
        texture_nb = int(info[2])
        fov = float(info[4])
        optical_slant = float(info[7])
        physical_slant = float(info[10])
        if len(info) > 13:
            size_var = int(info[13])
        else:
            size_var = 0
        convexity = 0 if info[0] == 'concave' else 1

        return im, texture_nb, fov, optical_slant, physical_slant, convexity, size_var
