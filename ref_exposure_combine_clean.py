import numpy as np
import cv2,os,math,time
from rawNoise import addISPNoise_clean
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numbers
import torch_msssim
from model_init import *
def yuv2rgb(yuvTensor):
          # y[0,1] u[-0.5 ,0.5] v[-0.5,0.5]
    size = yuvTensor.shape
    rgbTensor = torch.zeros(size)
    if yuvTensor.is_cuda:
        rgbTensor =rgbTensor.cuda()
    # r
    rgbTensor[:,0,:,:] = yuvTensor[:,0,:,:] + 1.403 * yuvTensor[:,2,:,:]
    # g
    rgbTensor[:,1,:,:] = yuvTensor[:,0,:,:] - 0.344 * yuvTensor[:,1,:,:] - 0.714 * yuvTensor[:,2,:,:]
    # b
    rgbTensor[:,2,:,:] = yuvTensor[:,0,:,:] + 1.770 * yuvTensor[:,1,:,:]

    return rgbTensor

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    
    def __init__(self, inplanes, planes, activation = nn.LeakyReLU(0.1),stride=1, downsample=None,expansion = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = activation
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)       
        self.stride = stride
        if downsample == None:
            if stride != 1 or inplanes != planes * expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * expansion),
                )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

class ScaleYUVBlock(nn.Module):

    def __init__(self, channel=64, kernel_size=3):
        super(ScaleYUVBlock, self).__init__()
        self.conv0 = nn.Conv2d(2, channel, kernel_size*3, padding=4)
        self.maxpool = nn.MaxPool2d(kernel_size*3, stride=4, padding=4)
        self.avgpool = nn.AvgPool2d(kernel_size*3, stride=4, padding=4)
        self.conv1 = nn.Conv2d(channel*2, channel, kernel_size*3, padding=4)
        self.conv2 = nn.Conv2d(channel*2, channel, kernel_size*3, padding=4)
        self.conv3 = nn.Conv2d(channel, 3, 1)
        self.eps = 10e-6
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x,ref_y):
        # x_max = torch.max(x, dim=3, keepdim=True)
        x_scale = torch.cat((x, ref_y), dim=1)
        global_intensity = self.ReLU(self.conv0(x_scale))
        global_intensity_max =self.maxpool(global_intensity)
        global_intensity_avg = self.avgpool(global_intensity)

        global_intensity_L = self.ReLU(self.conv1(torch.cat([global_intensity_max,global_intensity_avg], dim=1)))
        global_intensity_L_max =self.maxpool(global_intensity_L)
        global_intensity_L_avg = self.avgpool(global_intensity_L)
        global_intensity_LL = self.ReLU(self.conv2(torch.cat([global_intensity_L_max,global_intensity_L_avg], dim=1)))
        global_intensity_LL = f.interpolate(global_intensity_LL, scale_factor=4, mode='bilinear', align_corners=False)
        
        global_intensity_scale = global_intensity_L + global_intensity_LL
        global_intensity_scale= f.interpolate(global_intensity_scale, scale_factor=4, mode='bilinear', align_corners=False)
        global_scale= self.ReLU(self.conv3(global_intensity_scale))                    
        return global_scale

class SingleDecomNetSplit(nn.Module):

    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(SingleDecomNetSplit, self).__init__()
        self.layer_num = layer_num
        self.conv0 = nn.Conv2d(3, channel, kernel_size*3, padding=4)
        feature_conv = []
        for idx in range(layer_num):
            feature_conv.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size, padding=1, groups=2),
                nn.ReLU()
            ))
        self.conv = nn.ModuleList(feature_conv)
        self.conv1 = nn.Conv2d(channel, 3, kernel_size, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x_max = torch.max(x, dim=3, keepdim=True)
        # x1 = torch.cat((x, ref_y), dim=1).requires_grad_()
        residual = x 

        out = self.conv0(x)
        for idx in range(self.layer_num):
            out = self.conv[idx](out)
        out = self.conv1(out)        
        out = self.tanh(out)
        img2 = out+residual
        # img2[:,1,:,:] -=0.5
        # img2[:,2,:,:] -=0.5

        # out = out.permute(0, 2, 3, 1)
        # r_part = out[:, :, :, 0:3]
        # l_part = out[:, :, :, 3:4]

        return img2

class DecomYUVScaleNetSplit(nn.Module):
    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomYUVScaleNetSplit, self).__init__()
        self.layer_num = layer_num
        self.global_scale = ScaleYUVBlock(channel,kernel_size)
        self.enhancement =  SingleDecomNetSplit(layer_num,channel,kernel_size)
        
    def forward(self, x,ref_y,limit=False):
        # x_max = torch.max(x, dim=3, keepdim=True)
        x_y = x[:,0,:,:].unsqueeze(1)
        global_scale= self.global_scale(x_y,ref_y)       
        refine_x = x*global_scale
        if limit:
            refine_x[:,0,:,:].clamp_(min=0,max=1)
            refine_x[:,1:,:,:].clamp_(min=-0.5,max=0.5)
        final_output = self.enhancement(refine_x)

        return final_output, global_scale