import os
from collections import OrderedDict
from torch import nn
import numpy as np
import torch
from ignite.engine import *
from ignite.metrics import *
from pytorch_fid.inception import InceptionV3
from lpips_pytorch import LPIPS
from data_loader import StyleTransferDatasetEval, default_fn
from torchvision import transforms
from torch.utils.data import DataLoader
from glob import glob
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4,sci_mode=False)

# wrapper class as feature_extractor
class WrapperInceptionV3(nn.Module):

    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


# create default evaluator for doctests
def eval_step(engine, batch):
    return batch


def get_metric_evaluator(device):
    evaluator = Engine(eval_step)

    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # wrapper model to pytorch_fid model
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.eval()
    metric = FID(num_features=2048, feature_extractor=wrapper_model)
    metric.attach(evaluator, "fid")

    metric = SSIM(data_range=1.0)
    metric.attach(evaluator, 'ssim')

    psnr = PSNR(data_range=1.0)
    psnr.attach(evaluator, 'psnr')
    return evaluator


def calculate_metric(dataloader, lpips):
    lpips_list = []
    APD = []
    MSE = []
    total = 0
    for i, batch in enumerate(dataloader):
        images, target_images = batch
        images, target_images = images.to(device), target_images.to(device)
        APD.append((images - target_images).abs().mean().item() * 255)
        MSE.append(torch.nn.functional.mse_loss(images, target_images).item())
        lpips_score = lpips(images, target_images)
        lpips_list.append(lpips_score.item())
        total += images.shape[0]
    lpips_avg = round(np.mean(lpips_list) / total, 6)
    apd_avg = round(np.mean(APD), 6)
    mse_avg = round(np.mean(MSE), 6)

    return lpips_avg, apd_avg, mse_avg


def get_file_dict(file_root='Results/'):
    file_dict = {}
    for file_dir in os.listdir(file_root):
        if "v1" not in file_dir and "summer2winter_vangon_multi" not in file_dir:
            continue
        clean_path = os.path.join(file_root, file_dir, "Test_clean")
        encoded_path = os.path.join(file_root, file_dir, "Test_encoded")
        if "multi" in file_dir:
            copyright_path = os.path.join(file_root, file_dir, "Test_copyright_image")
        else:
            copyright_path = glob(os.path.join(file_root, file_dir, "*.png"))
            copyright_path += glob(os.path.join(file_root, file_dir, "*.jpg"))
            copyright_path = copyright_path[0]
        decoded_path = os.path.join(file_root, file_dir, "Test_decoded")
        if file_dir not in file_root:
            file_dict[file_dir] = {
                "Test_clean": clean_path,
                "Test_encoded": encoded_path,
                "copyright_path": copyright_path,
                "decoded_path": decoded_path,
            }
    return file_dict

if __name__ == "__main__":
    BATCH_SIZE = 25
    HEIGHT = 256
    WIDTH = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_evaluator = get_metric_evaluator(device)
    metric_lpips = LPIPS(net_type="vgg")
    metric_lpips.to(device)

    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
    ])

    evaluation_directories = [
        "Results/summer2winter",
    ]

    for dir in evaluation_directories:
        print(f"Evalution method: {dir}, \nSSIM\t PSNR \t FID \t LPIPS \t APD \t MSE")
        clean_input_dir = f'{dir}/Test_clean'
        test_encoded = f'{dir}/Test_encoded'
        dataset = StyleTransferDatasetEval(test_encoded, clean_input_dir, transform)
        dataloader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=BATCH_SIZE, num_workers=0)
        lpips, apd, mse= calculate_metric(dataloader, metric_lpips)
        state = metric_evaluator.run(dataloader)
        ssim, psnr, fid = state.metrics['ssim'], state.metrics['psnr'], state.metrics['fid']
        print("Clean and Encoded Image: {:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(ssim, psnr, fid, lpips, apd, mse))
        copyright_input_dir = f'{dir}/Test_copyright_image'
        decoded_dir = f'{dir}/Test_decoded'


        dataset_decoded = StyleTransferDatasetEval(decoded_dir, copyright_input_dir, transform)
        dataloader_decoded = DataLoader(dataset_decoded, shuffle=True, pin_memory=True, batch_size=BATCH_SIZE, num_workers=0)
        lpips, apd, mse = calculate_metric(dataloader_decoded, metric_lpips)
        state = metric_evaluator.run(dataloader_decoded)
        ssim, psnr, fid = state.metrics['ssim'], state.metrics['psnr'], state.metrics['fid']
        print("Copyright and Decoded Image: {:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(ssim, psnr, fid, lpips, apd, mse))
        print("="*15, "Done", "="*15, )