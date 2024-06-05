# Pytorch
import torch
import torch.nn as nn
# Local
from . import compress_jpeg, decompress_jpeg, compress_jpeg_origin, decompress_jpeg_origin
from .utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, device, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(device=device, rounding=rounding)
        self.decompress = decompress_jpeg(height, width, device=device, rounding=rounding)

    def forward(self, x, quality=80):
        '''
        '''
        # factor = quality_to_factor(factor)
        factor = quality_to_factor(quality)
        y, cb, cr = self.compress(x, factor)
        recovered = self.decompress(y, cb, cr, factor)
        return recovered


class DiffJPEGOrigin(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEGOrigin, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg_origin(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg_origin(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''
        '''
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered