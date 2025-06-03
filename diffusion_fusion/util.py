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
from math import exp
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    
    
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    assert img1.min() >= 0 and img1.max() <= 1, "Expect input in range [0,1]"
    if val_range is None:
        raise ValueError("val_range must be specified (e.g. 1.0 for [0,1] range or 255.0 for [0,255] range)")
    L = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2+ 1e-6
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range if val_range is not None else 1.0

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = (self.window).to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average,val_range=self.val_range)

class Sobelxy(torch.nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        self.kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        self.kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        weightx = torch.nn.Parameter(data=self.kernelx, requires_grad=False).to(x.device)
        weighty = torch.nn.Parameter(data=self.kernely, requires_grad=False).to(x.device)
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), weighty, padding=1)
                add_0 = torch.sqrt(sobelx_0 ** 2 + sobely_0 ** 2 + 1e-8) 
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))
        return torch.cat(batch_list, dim=0)


def high_freq_enhancement_loss(x0t_grad: torch.Tensor,
                               target_modulated: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:

    if target_modulated.dim() == 4 and target_modulated.shape[1] == 1:
        mask = target_modulated.expand(-1, x0t_grad.shape[1], -1, -1)
    else:
        mask = target_modulated

   
    grad_mag = x0t_grad.abs()

 
    masked_grad = grad_mag * mask

   
    sum_grad   = masked_grad.sum(dim=(1,2,3))        # B
    count_mask = mask.sum(dim=(1,2,3)).clamp(min=eps)  # B

    loss_per_batch = - sum_grad / count_mask       # B
    loss = loss_per_batch.mean()                   # scalar

    return loss
    
def local_contrast_enhancement_loss(x: torch.Tensor,
                                    mask: torch.Tensor,
                                    kernel_size: int = 5,
                                    eps: float = 1e-6) -> torch.Tensor:
  
    B, C, H, W = x.shape
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.expand(-1, C, -1, -1)  

    padding = kernel_size // 2
    kernel = torch.ones((C, 1, kernel_size, kernel_size),
                        device=x.device, dtype=x.dtype) / (kernel_size**2)
   
    local_mean = F.conv2d(x, weight=kernel, bias=None,
                          stride=1, padding=padding, groups=C)      
    local_mean_sq = F.conv2d(x * x, weight=kernel, bias=None,
                             stride=1, padding=padding, groups=C)       

    local_var = local_mean_sq - local_mean.pow(2)

    local_std = (local_var.clamp(min=eps)).sqrt()                            

    masked_std = local_std * mask
    sum_std   = masked_std.sum(dim=(1,2,3))                               
    count     = mask.sum(dim=(1,2,3)).clamp(min=eps)                         

    loss_per_sample = - sum_std / count                                      
    loss = loss_per_sample.mean()                                          

    return loss


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
    
def train_transform(img1: Image.Image,
                    img2: Image.Image,
                    MAX_SIZE: int,
                    CROP_SIZE: int,
                    flip_prob: float = 0.0,
                    jitter: bool = True):
    def resize_min_side(im):
    
        if min(*im.size) < CROP_SIZE:
            scale = CROP_SIZE / min(*im.size)
            im = im.resize(tuple(round(x * scale) for x in im.size), resample=Image.BICUBIC)
        if min(*im.size) >= MAX_SIZE:
            while min(*im.size) >= 2 * MAX_SIZE:
                im = im.resize(tuple(x // 2 for x in im.size), resample=Image.BOX)
            scale = MAX_SIZE / min(*im.size)
            im = im.resize(tuple(round(x * scale) for x in im.size), resample=Image.BICUBIC)
        return im
        
    img1 = resize_min_side(img1)
    img2 = resize_min_side(img2)
    if random.random() < flip_prob:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < flip_prob:
        img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
        img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img1.size
    if w >= CROP_SIZE and h >= CROP_SIZE:
        left = random.randint(0, w - CROP_SIZE)
        top  = random.randint(0, h - CROP_SIZE)
        box = (left, top, left + CROP_SIZE, top + CROP_SIZE)
        img1 = img1.crop(box)
        img2 = img2.crop(box)
    def to_tensor_YCB(im):
        arr = np.array(im.convert('YCbCr')).astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        c, hh, ww = arr.shape
        arr = arr[:, :hh - hh % 16, :ww - ww % 16]
        return torch.from_numpy(arr)

    t1 = to_tensor_YCB(img1)
    t2 = to_tensor_YCB(img2)
    y1, cb1, cr1 = t1[0:1], t1[1:2], t1[2:3]
    y2, cb2, cr2 = t2[0:1], t2[1:2], t2[2:3]
    if img1.mode == 'L':
        cb, cr = cb2, cr2
    else:
        cb, cr = cb1, cr1

    return y1, y2, cr, cb

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

class GET_TrainDataset(Dataset):
    def __init__(self, SourceImage1_Path, SourceImage2_Path, MAX_SIZE, CROP_SIZE):

        Image1_All = get_Images(SourceImage1_Path)
        Image2_All = get_Images(SourceImage2_Path)
        assert len(Image1_All) == len(Image2_All),"Data not matched"
        if len(Image1_All) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.Image1_All = Image1_All
        self.Image2_All = Image2_All
        self.transform = train_transform
        self.loader = loader
        self.MAX_SIZE = MAX_SIZE
        self.CROP_SIZE=CROP_SIZE

    def __getitem__(self, index):
        path1 = self.Image1_All[index]
        path2 = self.Image2_All[index]
        Image1 = self.loader(path1)
        Image2 = self.loader(path2)
        if self.transform is not None:
            y1, y2, cr, cb = self.transform(Image1, Image2, self.MAX_SIZE, self.CROP_SIZE)
        return y1, y2, cr, cb, path1,path2

    def __len__(self):
        return len(self.Image1_All)

class GetDictDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = sorted([
            f for f in os.listdir(folder_path) if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        data = torch.load(file_path)

        
        output_GT1 = data['output_GT1']
        output_GT2 = data['output_GT2']
        x_F_t = data['x_F_t']
        cond = data['cond']
        cond1 = data['cond1']
        filename=data["filename"]
        
        return output_GT1, output_GT2, x_F_t,cond,cond1,filename
