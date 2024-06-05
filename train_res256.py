from lpips_pytorch import LPIPS, lpips
import torch
from model import CopyrightEncoderImage, CopyrightDecoderImage, build_copyroght_image
from jpeg_compression import DiffJPEG
from torch.utils.data import DataLoader
from data_loader import StyleTransferDataset, default_fn
from torchvision import transforms
from tensorboardX import SummaryWriter
import time
import os
import argparse
import yaml
import numpy as np
import random
from pytorch_msssim import SSIM


import warnings
warnings.filterwarnings('ignore')

torch.set_printoptions(precision=4,sci_mode=False)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_copyright_image(path, transform):
    results = []
    for pth in path:
        image = default_fn(pth)
        results.append(transform(image))
    return results


def get_args():
    parser = argparse.ArgumentParser(description="Random Search Parameters")
    ################# config.yml #################
    parser.add_argument("--yaml_file", type=str, default="config.yml", help="the settings config")
    ################# load config.yml    ##################
    known_args, remaining = parser.parse_known_args()
    with open(known_args.yaml_file, 'r', encoding="utf-8") as fr:
        yaml_file = yaml.safe_load(fr)
        parser.set_defaults(**yaml_file)
    ################# 指定其他文件 #################
    parser.add_argument("--save_dir", type=str, default="saved_textures")
    args = parser.parse_args(remaining)
    return args


def save_checkpoint(encoder, decoder, loss, optimizer, args, save_path):
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "loss": loss,
        "optimizer": optimizer.state_dict(),
        "args": args
    }, save_path)


def main():
    args = get_args()
    seed_torch(seed=1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion_lpips = LPIPS(net_type="vgg")
    criterion_mse = torch.nn.MSELoss()
    criterion_ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    criterion_lpips.to(device)
    criterion_mse.to(device)
    criterion_ssim.to(device)

    jpeg = DiffJPEG.DiffJPEG(height=args.height, width=args.width, device=device)
    jpeg = jpeg.to(device)

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
    ])

    copyright_image = [
        'copyright_image/stanford_university.png',
        'copyright_image/peking_university.png',
        'copyright_image/ucla_university.png',
        'copyright_image/zhejiang_university.png',
        'copyright_image/ucla_university.png',
        'copyright_image/UN.png',
        'copyright_image/Tsinghua.jpg',
    ]

    dataset = StyleTransferDataset(args.data_dir, copyright_image, transform)
    dataloader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)
    encoder = CopyrightEncoderImage()
    decoder = CopyrightDecoderImage()
    encoder.to(device)
    decoder.to(device)
    protected_model = None

    optimizer = torch.optim.Adam([
        {"params": encoder.parameters()},
        {"params": decoder.parameters()}
    ], lr=args.lr)

    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    save_dir = f"{args.checkpoints}/{args.exp_name}_{time_str}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    writer = SummaryWriter(f"{args.run_path}/{args.exp_name}_{time_str}_copyright_image")

    args.checkpoints = save_dir
    args.run_path = f"{args.run_path}/{args.exp_name}_{time_str}_copyright_image"
    print(args)

    global_step = 1
    args.epochs = (args.num_steps * args.batch_size) // dataset.__len__() + 1

    for epoch in range(1, args.epochs):
        for i, batch in enumerate(dataloader):
            image2style, copyright = batch
            image2style, copyright = image2style.to(device), copyright.to(device)

            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
                                   args.l2_edge_gain)
            loss = build_copyroght_image(
                protected_model,
                encoder,
                decoder,
                jpeg,
                criterion_lpips,
                criterion_ssim,
                image2style,
                copyright,
                l2_edge_gain,
                [l2_loss_scale, lpips_loss_scale, secret_loss_scale],
                torch.tensor([args.y_scale, args.u_scale, args.v_scale]).to(device),
                args,
                writer,
                device,
                global_step
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step +=1

            if global_step % 1000 == 0:
                save_path = f"{save_dir}/copyright_image_{global_step}.pth"
                save_checkpoint(encoder, decoder, loss.item(), optimizer, args, save_path)
    writer.close()
    print("Training Finished!!")


if __name__ == "__main__":
    main()
