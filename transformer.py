import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
import os
from PIL import Image
import numpy as np
import tqdm
s=[]
def main(path):
    img_channels = 3
    img_names = os.listdir(path)
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for img_name in img_names:
        img_path = os.path.join(path, img_name)
        img = np.array(Image.open(img_path))
        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(img_names)
    std = cumulative_std / len(img_names)
    a,b,c=mean[0],mean[1],mean[2]
    d,e,f=std[0],std[1],std[2]
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    s.append(e)
    s.append(f)
def mean_and_variance(a, b, c):
    mean = (a + b + c) / 3
    variance = ((a - mean) ** 2 + (b - mean) ** 2 + (c - mean) ** 2) / 3

    return mean, variance

def ms(path):
    main(path)

    result1 = mean_and_variance(s[0],s[1],s[2])
    result2 = mean_and_variance(s[3],s[4],s[5])
    return round(result1[0],3),round(result2[0],3)

class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if train:
            self.mean,self.std=ms('./data/'+data_name+'/train/images')
        else:
            self.mean,self.std=ms('./data/'+data_name+'/val/images')

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk

