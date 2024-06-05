import torch
import torch.nn as nn
import torchvision.transforms
from PIL import Image
import utils
from stn.spatial_transformer_network import SpatialTransformerNetwork as stn_model
import torch.nn.functional as F
from jpeg_compression import DiffJPEG
from torchvision import models, transforms
import numpy as np
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=out_feature, out_channels=out_feature*2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=out_feature*2, out_channels=in_feature, kernel_size=1)


    def forward(self, x):
        residual = F.relu(self.conv1(x))
        residual = F.relu(self.conv2(residual))
        residual = F.relu(self.conv3(residual))
        residual = residual + x
        return residual

##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(Pix2PixGenerator, self).__init__()

        self.encoder_copyright = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.residual_block = ResidualBlock(in_feature=16, out_feature=32)
        self.encoder_copyright1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, inputs):

        style_image, copyright = inputs

        copyright = self.encoder_copyright(copyright)
        copyright = self.residual_block(copyright)
        copyright = self.encoder_copyright1(copyright)

        x = torch.cat([copyright, style_image], dim=1)

        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class CopyrightEncoderImage(nn.Module):
    def __init__(self):
        super(CopyrightEncoderImage, self).__init__()

        self.encoder_copyright = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.residual_block = ResidualBlock(in_feature=16, out_feature=32)
        self.encoder_copyright1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(6, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)

        self.up6 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 128, 3, 1, 1)
        self.up7 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 64, 3, 1, 1)
        self.up8 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 32, 3, 1, 1)
        self.up9 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv9 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, 3, 1)

    def forward(self, inputs):
        style_image, copyright = inputs

        copyright = self.encoder_copyright(copyright)
        copyright = self.residual_block(copyright)
        copyright = self.encoder_copyright1(copyright)

        inputs = torch.cat([copyright, style_image], dim=1)

        # images
        conv1 = F.relu(self.conv1(inputs))  # 1*32*400*400
        conv2 = F.relu(self.conv2(conv1))  # 1 * 32 *200*200
        conv3 = F.relu(self.conv3(conv2))  # 1 * 64 *100*100
        conv4 = F.relu(self.conv4(conv3))  # 1 * 128 *50*50
        conv5 = F.relu(self.conv5(conv4))  # 1 * 256 *25*25

        up6 = self.up6(nn.Upsample(scale_factor=2)(conv5))  # 1 * 128 *25*25
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6(merge6))  # 1 * 128 * 50 * 50

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))  # 1 * 64 * 100 * 100
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7(merge7))  # 1 * 64 * 100 * 100

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))  # 1 *32 *200 *200
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8(merge8))  # 1 *32 *200 *200

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))  # 1 *32 * 400 * 400
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = F.relu(self.conv9(merge9))

        conv10 = F.relu(self.conv10(conv9))
        style_image_with_copyrigth = self.residual(conv10)
        return style_image_with_copyrigth



class CopyrightDecoderImage(nn.Module):
    def __init__(self):
        super(CopyrightDecoderImage, self).__init__()
        self.stn = stn_model()
        self.stn = stn_model(fc_unit=10 * 13 * 13)  # (10 * 13 * 13) => 256*256,  (10 * 22 * 22) =>400

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)

        self.up6 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 128, 3, 1, 1)
        self.up7 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 64, 3, 1, 1)
        self.up8 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 32, 3, 1, 1)
        self.up9 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv9 = nn.Conv2d(64, 32, 3, 1 ,1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1 ,1)
        self.residual = nn.Conv2d(32, 3, 1)



    def forward(self, image):

        image = image - .5
        transformed_image = self.stn(image)
        # transformed_image = image

        conv1 = F.relu(self.conv1(transformed_image))  # 1*32*400*400
        conv2 = F.relu(self.conv2(conv1))  # 1 * 32 *200*200
        conv3 = F.relu(self.conv3(conv2))  # 1 * 64 *100*100
        conv4 = F.relu(self.conv4(conv3))  # 1 * 128 *50*50
        conv5 = F.relu(self.conv5(conv4))  # 1 * 256 *25*25

        up6 = self.up6(nn.Upsample(scale_factor=2)(conv5))  # 1 * 128 *25*25
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6(merge6))  # 1 * 128 * 50 * 50

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))  # 1 * 64 * 100 * 100
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7(merge7))  # 1 * 64 * 100 * 100

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))  # 1 *32 *200 *200
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8(merge8))  # 1 *32 *200 *200

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))  # 1 *32 * 400 * 400
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = F.relu(self.conv9(merge9))

        conv10 = F.relu(self.conv10(conv9))
        residual = self.residual(conv10)
        return residual



def transfer_net(encoded_image, jpeg, args, summarywriter, device, global_step):

    ramp_fn = lambda ramp: torch.minimum(torch.tensor(global_step) / ramp, torch.tensor(1.))


    # blur
    if args.random_blur:
        if global_step >= 80000:
            blur_kernal_size = [3, 5, 7, 9]
        elif 80000 > global_step >= 50000:
            blur_kernal_size = [3, 5, 7]
        elif 50000 > global_step >= 30000:
            blur_kernal_size = [3, 5]
        else:
            blur_kernal_size = [3]

        random_size = np.random.choice(blur_kernal_size, 1)[0]
        encoded_image = transforms.GaussianBlur((random_size, random_size))(encoded_image)
        encoded_image = torch.clamp(encoded_image, 0, 1)

    if args.jpeg_compression:
        jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
        encoded_image = jpeg(encoded_image, quality=jpeg_quality)

    if args.text_enhance:
        text_mask = utils.get_mask(args, encoded_image.shape[0])
        text_mask = text_mask.to(encoded_image.device)
        encoded_image = torch.where(text_mask == 0, encoded_image, text_mask)

    if global_step % 1000 == 0:
        summarywriter.add_scalar("transformer/jpeg_quality", jpeg_quality, global_step)
    return encoded_image



def build_copyroght_image(
        protected_model,
        encoder,
        decoder,
        jpeg,
        lpips_loss,
        mse_loss,
        image2style,
        copyright_image_input,
        l2_edge_gain,
        loss_scales,
        yuv_scales,
        args,
        summarywriter,
        device,
        global_step):
    if protected_model is not None:
        styled_image = protected_model(image2style)
    else:
        styled_image = image2style
    residual = encoder((styled_image, copyright_image_input))
    encoded_image = styled_image + residual

    # introducing the random distortion operation to remain the clean image quality
    if np.random.rand() < 0.5:
        transferred_image = transfer_net(encoded_image, jpeg, args, summarywriter, device, global_step)
    else:
        transferred_image = encoded_image

    decoded_image = decoder(transferred_image)

    loss_content = lpips_loss(styled_image.detach(), encoded_image)
    loss_copyright = mse_loss(copyright_image_input, decoded_image)


    size = (int(image2style.shape[2]),int(image2style.shape[3]))
    falloff_speed = 4
    falloff_im = np.ones(size)
    for i in range(int(falloff_im.shape[0]/falloff_speed)):
        falloff_im[-i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
        falloff_im[i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
    for j in range(int(falloff_im.shape[1]/falloff_speed)):
        falloff_im[:,-j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
        falloff_im[:,j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
    falloff_im = 1-falloff_im
    falloff_im = falloff_im.astype(np.float32())
    falloff_im *= l2_edge_gain
    falloff_im = torch.from_numpy(falloff_im).to(device)

    encoded_image_yuv = utils.rgb_to_yuv(encoded_image)
    image_input_yuv = utils.rgb_to_yuv(image2style)
    im_diff = encoded_image_yuv-image_input_yuv
    im_diff += im_diff * falloff_im.unsqueeze(dim=0)
    yuv_loss_op = torch.mean(torch.square(im_diff), dim=[0,2,3])
    image_loss_op = torch.tensordot(yuv_loss_op, yuv_scales, dims=1)
    loss = loss_content * loss_scales[0] + loss_copyright * loss_scales[1] + image_loss_op * loss_scales[2]
    print(f"【{global_step}/{args.num_steps}】: loss: {loss.item():.6f}\t, loss_content:{loss_content.item():.6f},\t "
          f"loss_copyright: {loss_copyright.item():.6f},\t image_loss_yuv: {image_loss_op.item():.6f}")

    if global_step % 1000 == 0:
        # scalar
        summarywriter.add_scalar("train/loss", loss.item(), global_step)
        summarywriter.add_scalar("train/loss_content", loss_content.item(), global_step)
        summarywriter.add_scalar("train/loss_copyright", loss_copyright.item(), global_step)
        summarywriter.add_scalar("color_loss/Y_loss", yuv_loss_op[0].item(), global_step)
        summarywriter.add_scalar("color_loss/U_loss", yuv_loss_op[1].item(), global_step)
        summarywriter.add_scalar("color_loss/V_loss", yuv_loss_op[2].item(), global_step)

        # images
        image_to_summary(styled_image[0], "styled_image", summarywriter, global_step, family="input")
        image_to_summary(encoded_image[0], "encoded_image", summarywriter, global_step, family="encoded")
        image_to_summary(residual[0] + 0.5, "residual", summarywriter, global_step, family="encoded")
        image_to_summary(transferred_image[0], "transferred_image", summarywriter, global_step, family="transformed")
        image_to_summary(decoded_image[0], "decoded_image", summarywriter, global_step, family="decoded")

    if global_step % 5000 == 0:
        torchvision.transforms.ToPILImage()(encoded_image[0]).save(f"{args.saved_path}/{global_step}_encoded_image.png")
        torchvision.transforms.ToPILImage()(residual[0]+0.5).save(f"{args.saved_path}/{global_step}_residual.png")
        torchvision.transforms.ToPILImage()(decoded_image[0]).save(f"{args.saved_path}/{global_step}_decoded_image.png")

    return loss

def jpeg_compression(image, jpeg, image_quality):
    encoded_image = jpeg(image, quality=image_quality)
    return encoded_image

def gaussian_blur(image, kernel_size):
    encoded_image = transforms.GaussianBlur((kernel_size, kernel_size))(image)
    return encoded_image

def text_image(image, args):
    text_mask = utils.get_mask(args, image.shape[0])
    text_mask = text_mask.to(image.device)
    encoded_image = torch.where(text_mask == 0, image, text_mask)
    return encoded_image

def combine_disruption(image, jpeg, args):
    image = jpeg_compression(image, jpeg, args.image_quality)
    image = gaussian_blur(image, args.kernel_size)
    image = text_image(image, args)
    return image


def build_copyroght_image_for_test(
        encoder,
        decoder,
        styled_image,
        copyright_image_input,
        jpeg,
        disruption_type,
        args):

    # 1. encode the image
    residual = encoder((styled_image, copyright_image_input))
    encoded_image = styled_image + residual
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # 2. disrupt the encoded image
    if disruption_type == 'jpeg':
        disrupted_images = jpeg_compression(encoded_image, jpeg, args.jpeg_quality)
    elif disruption_type == "blur":
        disruption_type = gaussian_blur(encoded_image, args.gaussian_blur)
    elif disruption_type == "text":
        disruption_type = text_image(encoded_image, args)
    elif disruption_type == "combine":
        disruption_type = combine_disruption(encoded_image, jpeg, args)
    else:
        disruption_type = encoded_image

    # 3. decode the encoded image
    encoded_image = torch.clamp(disruption_type, 0, 1)
    decoded_image = decoder(encoded_image)

    return encoded_image, decoded_image, residual


def image_to_summary(image, name, summary_writer, global_step, family='train'):
    image = torch.clamp(image, 0, 1)
    image = (image * 255).cpu().data.numpy().astype(np.uint8)
    summary_writer.add_image(f"{family}/{name}", image, global_step=global_step)


def save_image(image, save_path):
    image = torch.clamp(image, 0, 1)
    image = (image * 255).cpu().data.numpy().astype(np.uint8)
    image = np.transpose(image, (1,2,0))
    Image.fromarray(image).save(save_path)
