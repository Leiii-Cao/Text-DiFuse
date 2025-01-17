import copy
import math
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_Images(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def loader(path):
    return Image.open(path)

def test_transform(img,MAX_SIZE):
    if min(*img.size) >= MAX_SIZE:
        while min(*img.size) >= 2 * MAX_SIZE:
            img = img.resize(
                tuple(x // 2 for x in img.size), resample=Image.BOX
            )
        scale = MAX_SIZE / min(*img.size)
        img = img.resize(
            tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC
        )
    img = np.array(img)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.transpose(img, [2, 0, 1])  # HWC -> CHW
    _, h, w = img.shape
    img = img[:, :h-h%16, :w-w%16]
    return torch.from_numpy(img)
    
def to_numpy_image(images):
    images = copy.copy(images)
    images = images.detach().cpu().numpy() + 1
    images = (images * 127.5).round().astype("uint8")
    images = np.transpose(images, [0, 2, 3, 1])  # NCHW -> NHWC
    return images

class GET_TestDataset(Dataset):
    def __init__(self, SourceImage1_Path, SourceImage2_Path, MAX_SIZE):

        Image1_All = get_Images(SourceImage1_Path)
        Image2_All = get_Images(SourceImage2_Path)
        assert len(Image1_All) == len(Image2_All),"Data not matched"
        if len(Image1_All) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.Image1_All = Image1_All
        self.Image2_All = Image2_All
        self.transform = test_transform
        self.loader = loader
        self.MAX_SIZE = MAX_SIZE

    def __getitem__(self, index):
        path1 = self.Image1_All[index]
        path2 = self.Image2_All[index]
        Image1 = self.loader(path1)
        Image2 = self.loader(path2)
        if self.transform is not None:
            img1_ = self.transform(Image1.convert('YCbCr'),self.MAX_SIZE)[0:1]
            img2_ = self.transform(Image2.convert('YCbCr'),self.MAX_SIZE)[0:1]
            img3_ = self.transform(Image1.convert('YCbCr'),self.MAX_SIZE)[2:3] if Image1.mode!='L' else self.transform(Image2.convert('YCbCr'),self.MAX_SIZE)[2:3]
            img4_ = self.transform(Image1.convert('YCbCr'),self.MAX_SIZE)[1:2] if Image1.mode!='L' else self.transform(Image2.convert('YCbCr'),self.MAX_SIZE)[1:2]
        return img1_,img2_,img3_,img4_,path1,path2

    def __len__(self):
        return len(self.Image1_All)
