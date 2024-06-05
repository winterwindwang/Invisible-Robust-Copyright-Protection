import numpy as np
import random
import torch
from PIL import Image,ImageDraw,ImageFont,ImageEnhance
import os
import torch
from torchvision import transforms


def get_mask(args, batch_size):
    text_mask_tensor = []
    for _ in range(batch_size):
        angle = np.random.randint(-45, 46)
        font_size = np.random.randint(12, 48)
        content = np.random.choice(args.text_contents)
        position = np.random.randint(0, args.height - font_size * len(content) if args.height - font_size * len(
            content) > 0 else font_size)
        image = Image.new(mode="RGB", size=(args.height, args.width))
        watermark = Image.new('RGBA', image.size)
        draw = ImageDraw.Draw(watermark, 'RGBA')
        font = ImageFont.truetype("Arial.ttf", font_size)
        # x y 坐标
        draw.text([position, position], content, font=font)
        # 旋转45度
        watermark = watermark.rotate(angle, Image.BICUBIC)
        # 透明的
        alpha = watermark.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(0.7)
        watermark.putalpha(alpha)
        # 合成新的图片
        image = Image.composite(watermark, image, watermark)
        image = transforms.ToTensor()(image)
        text_mask_tensor.append(image.unsqueeze(dim=0))
    text_mask_tensor = torch.cat(text_mask_tensor, 0)
    return text_mask_tensor


# https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out

def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    g: torch.Tensor = y + -0.396 * u - 0.581 * v
    b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out
