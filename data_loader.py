import torch
from torch.utils.data import Dataset
from PIL import Image,ImageDraw,ImageFont,ImageEnhance
from glob import glob
import os
import numpy as np
from einops import rearrange
from torchvision import transforms
from torch.nn import functional as F

def default_fn(image_path):
    return Image.open(image_path).convert("RGB")


def get_batch_secret_key(image_size:tuple=(400,400), transform=None, batch_size=16, exclusive=((255,0,0),(0,255,0))):
    """
    Use the pure color for example
    :param color:
    :return:
    """
    batch_secret_key = []
    while True:
        red = np.random.randint(0,256)
        green = np.random.randint(0,256)
        blue = np.random.randint(0,256)
        random_color = (red, green, blue)
        if random_color in exclusive:
            continue
        key = transform(Image.new("RGB", size=image_size, color=random_color))
        batch_secret_key.append(key.unsqueeze(dim=0))
        if len(batch_secret_key) == batch_size:
            break
    return torch.cat(batch_secret_key, dim=0)


def get_secret_key(image_size:tuple=(400,400), color=(0,0,0)):
    """
    Use the pure color for example
    :param color:
    :return:
    """
    return Image.new("RGB", size=image_size, color=color)


class StyleTransferDataset(Dataset):
    def __init__(self, data_dir, copyright_list, transform=None,default_fn=default_fn):
        self.default_fn = default_fn
        self.transform = transform
        path_list = glob(os.path.join(data_dir, "*.jpg"))
        datas = []
        for path in path_list:
            rnd_idx = np.random.randint(0, len(copyright_list))
            datas.append((path, copyright_list[rnd_idx]))
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image2style_path, copyright_path = self.datas[index]
        image2style = self.default_fn(image2style_path)
        copyright = self.default_fn(copyright_path)

        if self.transform is not None:
            image2style = self.transform(image2style)
            copyright = self.transform(copyright)
        return image2style, copyright



class StyleTransferDatasetTest(Dataset):
    def __init__(self, data_dir, copyright_list, transform=None, default_fn=default_fn):
        self.default_fn = default_fn
        self.transform = transform
        path_list = glob(os.path.join(data_dir, "*.jpg"))
        datas = []
        for path in path_list:
            rnd_idx = np.random.randint(0, len(copyright_list))
            datas.append((path, copyright_list[rnd_idx]))
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image2style_path, copyright_path = self.datas[index]
        image2style = self.default_fn(image2style_path)
        copyright = self.default_fn(copyright_path)

        if self.transform is not None:
            image2style = self.transform(image2style)
            copyright = self.transform(copyright)

        return image2style, copyright, os.path.basename(image2style_path)


class StyleTransferDatasetEval(Dataset):
    def __init__(self, data_dir, target_dir, transform=None, retest_decoded=False, default_fn=default_fn):
        self.default_fn = default_fn
        self.transform = transform
        self.retest_decoded = retest_decoded
        if "//" in data_dir:
            data_dir = data_dir.replace("//", "/")
            target_dir = target_dir.replace("//", "/")
        path_list = os.listdir(data_dir)
        datas = []
        for filename in path_list:
            data_path = os.path.join(data_dir, filename)
            if os.path.isfile(target_dir):
                target_path = target_dir
            else:
                target_path = os.path.join(target_dir, filename)
            datas.append((data_path, target_path))
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image2style_path, target_path = self.datas[index]

        image2style = self.default_fn(image2style_path)

        if target_path.endswith(".jpg"):
            try:
                target = self.default_fn(target_path)
            except:
                target = self.default_fn(target_path.replace(".jpg",".png"))
        else:
            try:
                target = self.default_fn(target_path)
            except:
                target = self.default_fn(target_path.replace(".png",".jpg"))

        if self.transform is not None:
            image2style = self.transform(image2style)
            target = self.transform(target)
        if self.retest_decoded:
            return image2style, target, os.path.basename(image2style_path)
        else:
            return image2style, target


class ImageNetDataset(Dataset):
    def __init__(self, data_dir, copyright_list, transform=None, default_fn=default_fn, return_filename=False):
        self.default_fn = default_fn
        self.transform = transform
        path_list = glob(os.path.join(data_dir, "*/*.JPEG"))
        datas = []
        for path in path_list:
            rnd_idx = np.random.randint(0, len(copyright_list))
            datas.append((path, copyright_list[rnd_idx]))
        self.datas = datas
        self.return_filename = return_filename

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image2style_path, copyright_path = self.datas[index]
        image2style = self.default_fn(image2style_path)
        copyright = self.default_fn(copyright_path)

        if self.transform is not None:
            image2style = self.transform(image2style)
            copyright = self.transform(copyright)
        if not self.return_filename:
            return image2style, copyright
        else:
            return image2style, copyright, os.path.basename(image2style_path)


