import numpy as np
import os,math,time
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numbers
import torch_msssim
from PIL import Image
import torchvision.utils
import PWCNet
from datetime import datetime
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch.backends.cudnn
import torch.utils.data



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

class DecomNet_attention(nn.Module):

    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomNet_attention, self).__init__()
        self.layer_num = layer_num
        self.conv0 = nn.Conv2d(7, channel, kernel_size*3, padding=4)
        # feature_conv = []
        self.conv_l1 = BasicBlock(channel,channel,activation=nn.ReLU(inplace=True))
        self.conv_l2 = BasicBlock(channel,channel,activation=nn.ReLU(inplace=True))
        self.conv_l3 = BasicBlock(channel,channel,activation=nn.ReLU(inplace=True))
        self.conv_l4 = BasicBlock(channel,channel,activation=nn.ReLU(inplace=True))
        self.conv_l5 = BasicBlock(channel,channel,activation=nn.ReLU(inplace=True))
        # for idx in range(layer_num):
        #     feature_conv.append(nn.Sequential(BasicBlock(channel,channel,activation=nn.ReLU)))
            # feature_conv.append(nn.Sequential(
            #     nn.Conv2d(channel, channel, kernel_size, padding=1),
            #     nn.ReLU()
            # ))
        # self.conv = nn.ModuleList(feature_conv)
        self.conv1 = nn.Conv2d(channel, 6, kernel_size, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x,warp_x,ref_y,strong_mask=False,output_mask=False):
        # x_max = torch.max(x, dim=3, keepdim=True)
        x = torch.cat((x,warp_x, ref_y), dim=1)
        

        out = self.conv0(x)
        out = self.conv_l1(out)
        out = self.conv_l2(out)
        out = self.conv_l3(out)
        out = self.conv_l4(out)
        out = self.conv_l5(out)
        # for idx in range(self.layer_num):
        #     out = self.conv[idx](out)
        out = self.conv1(out)
        out = self.sig(out)
        img2,mask = out.clone()[:,0:3,:,:],out.clone()[:,3:,:,:]
        img2[:,1,:,:] -=0.5
        img2[:,2,:,:] -=0.5
        if strong_mask:
            mask = 1/(1+torch.exp(-10*(mask-0.5)))
            # mask[mask>0.5]=1
            # mask[mask<=0.5]=0
        out_refine = img2*mask+ warp_x*(1-mask)
        # out = out.permute(0, 2, 3, 1)
        # r_part = out[:, :, :, 0:3]
        # l_part = out[:, :, :, 3:4]
        if output_mask:
            return out_refine,mask
        else:
            return out_refine

