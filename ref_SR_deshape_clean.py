import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
import os, sys, glob
import cv2
from tensorboardX import SummaryWriter
import time,math
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch_msssim
from image_utils import rgb2yuv,Rescale,addGaussianNoise,yuv2rgb

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class conv_block(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU(inplace=True), is_BN=False):
        super(conv_block, self).__init__()
        if is_BN:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("bn", nn.BatchNorm2d(outc)),
                ("act", activation)
            ]))
        elif activation is not None:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("act", activation)
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
            ]))

    def forward(self, input):
        return self.conv(input)

class fc(nn.Module):
    def __init__(self, inc, outc, activation=None, is_BN=False):
        super(fc, self).__init__()
        if is_BN:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
                ("bn", nn.BatchNorm1d(outc)),
                ("act", activation),
            ]))
        elif activation is not None:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
                ("act", activation),
            ]))
        else:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
            ]))

    def forward(self, input):
        return self.fc(input)

class Guide2(nn.Module):
    '''
    pointwise neural net
    '''
    def __init__(self, mode="PointwiseNN"):
        super(Guide2, self).__init__()
        if mode == "PointwiseNN":
            self.mode = "PointwiseNN"
            self.conv1 = conv_block(1, 16, kernel_size=3,stride=1, padding=1, is_BN=False)
            self.conv2 = conv_block(16, 1, kernel_size=1, padding=0, activation=nn.Tanh())

        elif mode == "PointwiseCurve":
            # ccm: color correction matrix
            self.ccm = nn.Conv2d(3, 3, kernel_size=1)

            pixelwise_weight = torch.FloatTensor([1, 0, 0, 0, 1, 0, 0, 0, 1]) + torch.randn(1) * 1e-4
            pixelwise_bias = torch.FloatTensor([0, 0, 0])

            self.conv1x1.weight.data.copy_(pixelwise_weight.view(3, 3, 1, 1))
            self.conv1x1.bias.data.copy_(pixelwise_bias)

            # per channel curve
            pass

            # conv2d: num_output = 1
            self.conv1x1 = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        if self.mode == "PointwiseNN":
            guidemap = self.conv2(self.conv1(x))

        return guidemap
    
class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        N, _, H, W = guidemap.shape
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(N,1,1,1)
        yy = yy.view(1,1,H,W).repeat(N,1,1,1)
        xx = 2.0*xx/max(W-1,1)-1.0
        yy = 2.0*yy/max(H-1,1)-1.0
        grid = torch.cat((xx,yy),1).float()
        if guidemap.is_cuda:
            grid = grid.cuda()
        
        guidemap_guide = torch.cat([grid,guidemap], dim=1).permute(0,2,3,1).contiguous().unsqueeze(1)

        coeff = f.grid_sample(bilateral_grid, guidemap_guide)

        

        return coeff.squeeze(2)

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def forward(self, coeff, full_res_input):
        # Y =  coeff[:, 0:3, :, :]
        # U = torch.sum(full_res_input * coeff[:, 6:7, :, :], dim=1, keepdim=True) + coeff[:, 3:6, :, :]
        # V = torch.sum(full_res_input * coeff[:, 10:11, :, :], dim=1, keepdim=True) + coeff[:, 7:10, :, :]
        Y = full_res_input * coeff[:, 3:4, :, :] + torch.sum(coeff[:, 0:3, :, :], dim=1, keepdim=True)
        U = full_res_input * coeff[:, 7:8, :, :] + torch.sum(coeff[:, 4:7, :, :], dim=1, keepdim=True)
        V = full_res_input * coeff[:, 11:12, :, :]+ torch.sum(coeff[:, 8:11, :, :], dim=1, keepdim=True)
        return torch.cat([Y, U, V], dim=1)

class adjustChrome(nn.Module):
    def __init__(self):
        super(adjustChrome,self).__init__()
        self.conv1 = conv_block(1, 16, kernel_size=1, padding=0, is_BN=False)
        self.conv2 = conv_block(16, 1, kernel_size=1, padding=0, activation=nn.Tanh())

    def forward(self,chromeInfo):

        chromemap = self.conv1(chromeInfo)
        # print 'success'
        chromemap = self.conv2(chromemap)
        return chromemap

class global_brach(nn.Module):
    def __init__(self, inc=64, outc=64,BN=True):
        super(global_brach, self).__init__()
        self.average_0 = nn.AdaptiveAvgPool2d((1,1))
        self.conv_1 = conv_block(inc, 2*inc,kernel_size = 3,padding=1,stride=2,is_BN=BN)        
        self.average_1 = nn.AdaptiveAvgPool2d((1,1))
        self.conv_2 = conv_block(2*inc, 4*inc,kernel_size = 3,padding=1,stride=2,is_BN=BN)
        self.average_2 = nn.AdaptiveAvgPool2d((1,1))
        self.fuse_1 = conv_block(7*inc, 4*inc,kernel_size = 1,padding=0,is_BN=BN)  
        self.fuse_2 = conv_block(4*inc, 2*inc,kernel_size = 1,padding=0,is_BN=BN) 
        self.fuse_3 = conv_block(2*inc, 1*inc,kernel_size = 1,padding=0,is_BN=BN)      
        # self.downsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self,x):
        a0 = self.average_0(x)
        
        x = self.conv_1(x)
        a1 = self.average_1(x)
        
        x = self.conv_2(x)
        a2 = self.average_2(x)
        
        a = torch.cat((a0,a1,a2), dim=1)
        # print("!!!",a.shape)
        a = self.fuse_1(a)
        # print("!!!",a.shape)
        a = self.fuse_2(a)
        a = self.fuse_3(a)
        # print("!!!",a.shape)
        return a 

class HDRNetwoBN(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(HDRNetwoBN, self).__init__()
        self.inc = inc
        self.outc = outc

        # self.downsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)

        # -----------------------------------------------------------------------
        splat_layers = []
        for i in range(4):
            if i == 0:
                splat_layers.append(conv_block(self.inc, (2**i) * 8, kernel_size=3, padding=1, stride=2, activation=self.activation, is_BN=False))
            else:
                splat_layers.append(conv_block((2**(i-1) * 8), (2**(i)) * 8, kernel_size=3, padding=1, stride=2, activation=self.activation, is_BN=False))

        self.splat_conv = nn.Sequential(*splat_layers)

        # -----------------------------------------------------------------------
        self.global_brach = global_brach(64,64,BN=False)

        # -----------------------------------------------------------------------
        local_layers = [
            conv_block(64, 64, activation=self.activation, is_BN= False),
            conv_block(64, 64, use_bias=False, activation=None, is_BN=False),
        ]
        self.local_conv = nn.Sequential(*local_layers)

        # -----------------------------------------------------------------------
        self.linear = nn.Conv2d(64, 96, kernel_size=1)

        self.guide_func = Guide2()
        self.slice_func = Slice()
        self.transform_func = Transform()
        self.adjustChromeU = adjustChrome()
        self.adjustChromeV = adjustChrome()

    def forward(self, low_res_input,full_res_input):

        bs, _, _, _ = low_res_input.size()
        _, _, hh, hw= full_res_input.size()
        fake_res_input = f.interpolate(low_res_input,size=(hh,hw),mode='bilinear')
        # print 'color_image size:',low_res_input.size()
        splat_fea = self.splat_conv(low_res_input)
        # print('use_feature size:',splat_fea.size())
        local_fea = self.local_conv(splat_fea)
        # print 'local_feature size:',local_fea.size()
        global_fea = self.global_brach(splat_fea)
        # print 'global_fea size:',global_fea.size()
        fused = self.activation(global_fea.view(-1, 64, 1, 1) + local_fea)
        # print 'fused_initial size',fused.size()
        fused = self.linear(fused)
        # print('fused_re size',fused.size())
        
        f_n,f_c,f_h,f_w = fused.size()
        bilateral_grid = fused.view(-1, 12, 8, f_h, f_w)
        guidemap = self.guide_func(full_res_input)
        coeff = self.slice_func(bilateral_grid, guidemap)
        # print 'coeff size',coeff.size()
        # buffer YUV represents the direct result of net
        bufferYUV = self.transform_func(coeff, full_res_input)
        # print 'bufferYUV size',bufferYUV.size()
        # print 'chromeu',bufferYUV[:,1,:,:].shape
        U = self.adjustChromeU(bufferYUV[:,1,:,:].unsqueeze(1)) + fake_res_input[:,1,:,:].unsqueeze(1)
        # print 'U',U.size()
        V = self.adjustChromeV(bufferYUV[:,2,:,:].unsqueeze(1)) + fake_res_input[:,2,:,:].unsqueeze(1)
        output = torch.cat([bufferYUV[:,0,:,:].unsqueeze(1),U,V],dim=1)
        # print 'success'
        return output


    # y[0,1] u[-0.5 ,0.5] v[-0.5,0.5]

