import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch_msssim
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
import os, sys, glob
import cv2
import time,math
from image_utils import rgb2yuv,Rescale,addGaussianNoise,yuv2rgb
from datetime import datetime
import inspect
from collections import OrderedDict
from PIL import Image
from datetime import datetime 
from loadDataset import PostProDataset
from myLoss import PSNR_loss
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--predicted_dir',type=str,default='',help='common')
parser.add_argument('--predicted_prefix',type=str,default='',help='common')
parser.add_argument('--gt_dir',type=str,default='',help='common')
parser.add_argument('--gt_prefix',type=str,default='',help='common')
parser.add_argument('--proposal_name',type=str,default='',help='common')
parser.add_argument('--line_index',type=int,default=0,help='common')
parser.add_argument('--tensor_log_path',type=str,default='test_list/overall_pipeline_new',help='common')

args = parser.parse_args() 
# base settings 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
RGB2YUV = rgb2yuv()
# loss definition
loss_dict={}
loss_dict['msssim']=torch_msssim.MS_SSIM(max_val=1).to(device)
loss_dict['psnr-yuv'] = PSNR_loss(flag_YUV= True)
loss_dict['psnr-uv'] = PSNR_loss(flag_UV= True)
# save directory 
tensor_log_path = args.tensor_log_path
if not os.path.exists(tensor_log_path):
    os.mkdir(tensor_log_path)

proposal_name = args.proposal_name

if not os.path.exists(tensor_log_path+'/'+proposal_name):
    os.mkdir(tensor_log_path+'/'+proposal_name)

metric_pd = pd.DataFrame({"TIMESTAMP":[],"model_name":[],\
        "predicted_dataset":[],"gt_dataset":[]})





for i_index in range(1):

    root_dir = {'predicted':'','predicted_prefix':'','gt':'','gt_prefix':''}
    root_dir['predicted'] = args.predicted_dir
    root_dir['predicted_prefix'] = args.predicted_prefix
    root_dir['gt'] = args.gt_dir
    root_dir['gt_prefix'] = args.gt_prefix
    

    test_dataset = PostProDataset(root_dir)
    BATCH_SIZE= 1 
    test_loader = DataLoader(dataset = test_dataset, batch_size= BATCH_SIZE, shuffle = False, num_workers = 8,pin_memory=True)
    epoch_loss = {}
    for loss_name in loss_dict.keys():
        epoch_loss[loss_name] = 0
    STEP_MAX = math.ceil(len(test_loader.dataset) / BATCH_SIZE )
    metric_pd=metric_pd.append({"TIMESTAMP":TIMESTAMP,"model_name":proposal_name,\
            "predicted_dataset":root_dir['predicted'],"gt_dataset":root_dir['gt']},ignore_index=True)
    metric_pd.loc[metric_pd.shape[0]+1]= {"TIMESTAMP":TIMESTAMP,"model_name":proposal_name,\
            "predicted_dataset":root_dir['predicted'],"gt_dataset":root_dir['gt']}

    initial_line = metric_pd.shape[0]
    for stept, sample in enumerate(test_loader):
        step = initial_line+stept-1
        yuv_image = sample['predict'].to(device)
        labels_image = sample['label'].to(device)
        for key in loss_dict.keys():
            # if key not in metric_pd.keys():
            #     metric_pd[key] = 0
            if key != "msssim":
                metric_pd.loc[step+1,key] = (loss_dict[key](RGB2YUV(yuv_image),RGB2YUV(labels_image))).item()
                epoch_loss[key] += metric_pd.loc[step+1,key]
            else:
                metric_pd.loc[step+1,key]= loss_dict[key]((yuv_image),(labels_image)).item()
                epoch_loss[key] += metric_pd.loc[step+1,key]
    
    print(1-(epoch_loss['msssim']/float(stept+1)),epoch_loss['psnr-yuv']/float(stept+1),epoch_loss['psnr-uv']/float(stept+1))
    metric_pd=metric_pd.append({'model_name':'average','msssim':(epoch_loss['msssim']/float(stept+1)),\
        'psnr-yuv':(epoch_loss['psnr-yuv']/float(stept+1)),'psnr-uv':(epoch_loss['psnr-uv']/float(stept+1))},ignore_index=True)
    metric_pd=metric_pd.append({'psnr-yuv':10 * math.log10(1/metric_pd.loc[metric_pd.shape[0]-1,'psnr-yuv']),\
        'psnr-uv':10 * math.log10(1/metric_pd.loc[metric_pd.shape[0]-1,'psnr-uv'])},ignore_index=True)
    

if not os.path.exists(tensor_log_path+'/'+proposal_name+'/logging.csv'):
    metric_pd.to_csv(tensor_log_path+'/'+proposal_name+'/logging.csv', mode='w', header=True, index=None)
else:
    metric_pd.to_csv(tensor_log_path+'/'+proposal_name+'/logging.csv', mode='a', header=True, index=None)