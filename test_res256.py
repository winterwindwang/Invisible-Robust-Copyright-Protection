import torch
from model import CopyrightEncoderImage, CopyrightDecoderImage, build_copyroght_image_for_test
from jpeg_compression import DiffJPEG
from torch.utils.data import DataLoader, Subset
from data_loader import StyleTransferDatasetTest, default_fn, StyleTransferDatasetEval, ImageNetDataset
from torchvision import transforms
import os
import argparse
import shutil
from PIL import Image
from pytorch_msssim import SSIM
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISABLE_DEVICES'] = "0"

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def get_copyright_image(path, transform):
    results = []
    for pth in path:
        image = default_fn(pth)
        results.append(transform(image))
    return results


def save_images(images, filenames, save_dir):
    for i, batch in enumerate(zip(images, filenames)):
        img, path = batch
        try:
            dest_path = os.path.join(save_dir, path.replace(".jpg", ".png"))
        except:
            dest_path = os.path.join(save_dir, path)
        image = torch.clamp(img, 0, 1)
        image = (image * 255).cpu().data.numpy().astype(np.uint8)
        image = np.transpose(image, (1,2,0))
        Image.fromarray(image).save(dest_path)
        # transforms.ToPILImage()(img).save(dest_path)
    

def get_args():
    parser = argparse.ArgumentParser(description="Random Search Parameters")
    parser.add_argument("--save_dir", type=str, default="Results")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    return args


def load_checkpoint(encoder, decoder, ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    return encoder, decoder


def test(data_dict, evaluate_name='monet2photo_vangon_stanford'):
    args = get_args()
    seed_torch(seed=1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    jpeg = DiffJPEG.DiffJPEG(height=args.height, width=args.width, device=device)
    jpeg = jpeg.to(device)

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
    ])

    data_dir = data_dict[evaluate_name]['data_path']
    copyright_image = data_dict[evaluate_name]['copyright_path']
    if "summer2winter" in data_dir.lower():
        val_dataset = StyleTransferDatasetTest(data_dir, copyright_image, transform=transform)
        val_len = val_dataset.__len__()
    if "summer2winter_vangogh" in data_dir.lower():
        val_dataset = StyleTransferDatasetTest(data_dir, copyright_image, transform=transform)
        val_len = val_dataset.__len__()
    elif "imagenet" in data_dir.lower():
        val_datasets = ImageNetDataset(data_dir, copyright_image, transform=transform, return_filename=True)
        np.random.seed(1024)
        val_len = 1000
        sample_indices = np.random.permutation(range(val_datasets.__len__()))[:val_len]
        val_dataset = Subset(val_datasets, sample_indices)


    dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=0)
    encoder = CopyrightEncoderImage()
    decoder = CopyrightDecoderImage()
    encoder, decoder = load_checkpoint(encoder, decoder, data_dict[evaluate_name]['ckpt_path'], device)

    base_dir = data_dict[evaluate_name]['ckpt_path'][:data_dict[evaluate_name]['ckpt_path'].find("checkpoint")]
    args.save_dir = os.path.join(base_dir, "Results", evaluate_name)
    print("File save at: ", args.save_dir)

    save_dir_clean = os.path.join(args.save_dir, "Test_clean")
    save_dir_encoded = os.path.join(args.save_dir, "Test_encoded")
    save_dir_decoded = os.path.join(args.save_dir, "Test_decoded")
    save_dir_cover_residual = os.path.join(args.save_dir, "Test_cover_residual")
    save_dir_secret_residual = os.path.join(args.save_dir, "Test_secret_residual")
    save_dir_copyright = os.path.join(args.save_dir, "Test_copyright_image")

    if not os.path.exists(save_dir_clean):
        os.makedirs(save_dir_clean)
        os.makedirs(save_dir_encoded)
        os.makedirs(save_dir_decoded)
        os.makedirs(save_dir_cover_residual)
        os.makedirs(save_dir_secret_residual)
        os.makedirs(save_dir_copyright)

    disruption_type = "compression"

    for i, batch in enumerate(dataloader):
        image2style, copyright, filenames = batch
        image2style, copyright = image2style.to(device), copyright.to(device)

        encoded_images, decoded_images, residual = build_copyroght_image_for_test(
            encoder,
            decoder,
            image2style,
            copyright,
            jpeg,
            disruption_type,
            args,
        )
        residual_cover = encoded_images - image2style
        residual_secret = decoded_images - copyright
        save_images(image2style, filenames, save_dir_clean)
        save_images(copyright, filenames, save_dir_copyright)
        save_images(encoded_images, filenames, save_dir_encoded)
        save_images(decoded_images, filenames, save_dir_decoded)
        save_images((residual_cover*10 + 0.5), filenames, save_dir_cover_residual)
        save_images((residual_secret*10 + 0.5), filenames, save_dir_secret_residual)

    print("Image generated Finished!!")


checkpoint_dict = {
    "summer2winter": {
        "ckpt_path": "checkpoints/Res256_copyright_image_07-04-13-30/copyright_image_140000.pth",
        "data_path": "F:/DataSource/StyleTransfer/summer2winter_yosemite/testA/",
        "copyright_path": [
            'copyright_image/peking_university.png',
            'copyright_image/stanford_university.png',
            'copyright_image/Tsinghua.jpg',
            'copyright_image/ucla_university.png',
            'copyright_image/zhejiang_university.png',
            'copyright_image/UN.png',
        ]
    },
}

def check_out_files(data_dict):
    for key, value in data_dict.items():
        ckpt_path = value['ckpt_path']
        data_path = value['data_path']
        if not os.path.exists(ckpt_path):
            raise ValueError(f"{key}: {ckpt_path} not exist!!")
        if not os.path.exists(data_path):
            raise ValueError(f"{key}: {data_path} not exist!!")
    return True


def test_batch(data_dicts):
    for key, values in data_dicts.items():
        test(data_dicts, key)
        print(f"{key} proceeded!!!")


if __name__ == "__main__":
    if check_out_files(checkpoint_dict):
        print("Filed check passed!!")
    # test()
    test_batch(checkpoint_dict)
