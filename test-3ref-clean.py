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
from torch.utils.data import Dataset, DataLoader
import torch_msssim
from image_utils import rgb2yuv,Rescale,addGaussianNoise,yuv2rgb,flow2rgb
from datetime import datetime
import PWCNet
import inspect
from gpu_mem_track import MemTracker
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid.clone())
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid.clone())

   # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask,mask

class myBilateralDataset(Dataset):
    def __init__(self,root_txt,color_transform = None,mono_transform = None,mono_label_transform = None,color_label_transform =None, max_num=39999):
        self.left_pic_list = []
        self.right_pic_list = []
        self.left_label_pic_list = []
        self.right_label_pic_list = []
        with open(root_txt,'r') as file:
            i = 0 
            for line in file.readlines():
                if i >max_num:
                    break
                else:
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[0])
                    self.right_pic_list.append(curLine[1])
                    self.right_label_pic_list.append(curLine[1])
                    self.left_label_pic_list.append(curLine[2])
                i+=1
        file.close()
        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform
        self.mono_label_transform = mono_label_transform
        self.color_label_transform = color_label_transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # print(self.left_pic_list[idx],self.right_pic_list[idx])
        if np.array(Image.open(self.left_pic_list[idx])) is None:
            color_image = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_pic_list[idx])
        else:
            color_image = np.array(Image.open(self.left_pic_list[idx])).astype(np.float32).transpose(2,0,1)/255.0
        # else:
            # print(self.left_pic_list[idx])
        if np.array(Image.open(self.right_pic_list[idx])) is None:   
            mono_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_pic_list[idx])
        else:
            mono_source = np.array(Image.open(self.right_pic_list[idx])).astype(np.float32).transpose(2,0,1)/255.0
            # mono_source = np.array(Image.open(self.right_pic_list[idx])).astype(np.float32)/255.0
        if np.array(Image.open(self.right_label_pic_list[idx])) is None:   
            mono_label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_label_pic_list[idx])
        else:
            mono_label_source = np.array(Image.open(self.right_label_pic_list[idx])).astype(np.float32).transpose(2,0,1)/255.0
        if np.array(Image.open(self.left_label_pic_list[idx])) is None:   
            color_label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_label_pic_list[idx])
        else:
            color_label_source = np.array(Image.open(self.left_label_pic_list[idx])).astype(np.float32).transpose(2,0,1)/255.0
        # else:s
            # print(self.right_pic_list[idx])
        colorTensor = torch.from_numpy(color_image)
        monoTensor_ = torch.from_numpy(mono_source)
        h,w = monoTensor_.shape[-2:]
        monoTensor = torch.zeros(1,h,w)
        # monoTensor =monoTensor_.unsqueeze(0)
        monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        mono_labelTensor = torch.from_numpy(mono_label_source)
        color_labelTensor = torch.from_numpy(color_label_source)


        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)
        if self.mono_label_transform:
          mono_labelTensor = self.mono_label_transform(mono_labelTensor)
        if self.color_label_transform:
          color_labelTensor = self.color_label_transform(color_labelTensor)

        # mono_out_patch,decolor_out_patch,index_out = self.searchArea(monoTensor[0,:,:],colorTensor[0,:,:])

        sample = {'mono':monoTensor,'color':colorTensor,'label':mono_labelTensor,'label_color':color_labelTensor}
        return sample


def test_new_bilateral_simulate(args):
    from ref_exposure_combine_clean import DecomYUVScaleNetSplit
    from test_flow_sample_refine_res_clean import DecomNet_attention
    from ref_SR_deshape_clean import HDRNetwoBN
    from loadDataset import RealCaptureDataset,myTestRealColorDataset,myTestOverallBilateralDataset
    from test_device import is_gpu_supported_by_pytorch13
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    frame = inspect.currentframe()
    
    ### check whether the GPU supports PyTorch 1.3.0 based on compute capability. If not, use the CPU mode.
    supported, message = is_gpu_supported_by_pytorch13()
    device = torch.device('cuda' if supported else "cpu")
    print(device)
    
    pre_model_path= ['ckpt/ref_IE.pkl',\
        'ckpt/ref_AT_low.pkl', \
            'ckpt/ref_AT_high.pkl', \
                'ckpt/ref_AT_low_refine.pkl', \
                    'ckpt/ref_AT_high_refine.pkl', \
                        'ckpt/ref_SR.pkl']
   

    for i in range(1):
        ref_exp_net = DecomYUVScaleNetSplit()        
        if supported:
            ref_clo_net1 = PWCNet.PWCDCNet(input_channel=1)
            ref_clo_net2 = PWCNet.PWCDCNet(input_channel=1)
        else:
            ref_clo_net1 = PWCNet.PWCDCNetCPU(input_channel=1)
            ref_clo_net2 = PWCNet.PWCDCNetCPU(input_channel=1)
        ref_clo_net_refine1 = DecomNet_attention()
        ref_clo_net_refine2 = DecomNet_attention()
        ref_sr_net = HDRNetwoBN()


        pre_model_path_E = pre_model_path[6*i]
        ref_exp_net.load_state_dict(torch.load(pre_model_path_E))
        pre_model_path1 = pre_model_path[6*i+1]
        ref_clo_net1.load_state_dict(torch.load(pre_model_path1))
        pre_model_path2 = pre_model_path[6*i+2]
        ref_clo_net2.load_state_dict(torch.load(pre_model_path2))
        pre_model_path3 = pre_model_path[6*i+3]
        ref_clo_net_refine1.load_state_dict(torch.load(pre_model_path3))
        pre_model_path4 = pre_model_path[6*i+4]
        ref_clo_net_refine2.load_state_dict(torch.load(pre_model_path4))        
        pre_model_path_S = pre_model_path[6*i+5]
        ref_sr_net.load_state_dict(torch.load(pre_model_path_S))

        if supported:
            ref_exp_net.to(device)
            ref_clo_net1.to(device)
            ref_clo_net_refine1.to(device)
            ref_clo_net2.to(device)
            ref_clo_net_refine2.to(device)
            ref_sr_net.to(device)

        
        dataset_path='/gpy/update/testimageList-14-perfect-L1-L1-im0e6-im1e3-all.txt'
        # dataset_path='test_pair_list.txt'

        image_set_path = f'result/140811'
        if not os.path.exists(image_set_path):
            os.mkdir(image_set_path)
        tensor_log_path = image_set_path
        if not os.path.exists(tensor_log_path):
            os.mkdir(tensor_log_path)
        
        BATCH_SIZE = 1
        file2 = open(tensor_log_path+'/test.txt','a')
        file2.write('============='+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'===============\n')
        file2.write('device:'+str(device)+'\n')
        file2.write('pre_load_path:'+str(pre_model_path)+'\n')
        file2.write('dataset_path:'+dataset_path+'\n')
        file2.write('epoch'+' step'+' mse_loss'+' color_loss u_loss v_loss msssim_loss l1_loss\n')

        low_img_size= (256,256)
        high_img_size= (1024,1024)
        test_dataset = myBilateralDataset(dataset_path,\
                            mono_transform = transforms.Compose([Rescale(high_img_size)]),\
                            color_transform =transforms.Compose([Rescale(low_img_size),rgb2yuv()]),\
                            mono_label_transform =transforms.Compose([Rescale(high_img_size)]),\
                            color_label_transform =transforms.Compose([Rescale(high_img_size)]),        max_num=39999)
         
    
        Real_FLAG = False
        test_loader = DataLoader(dataset = test_dataset, batch_size= BATCH_SIZE, shuffle = False, num_workers = 1,pin_memory=True)
        criterion = torch.nn.MSELoss()  # Define MSE loss
        criterion2 = nn.CosineSimilarity(dim=1, eps=1e-08)
        criterion4  = torch.nn.L1Loss()
        criterion3 = torch_msssim.MS_SSIM(max_val=1).to(device)
        RGB2YUV = rgb2yuv()        
        epoch = 0
        
        EPOCH_TOTAL = 1
        EPOCH_RECORD = 2
        STEP_MAX = math.ceil(len(test_loader.dataset) / BATCH_SIZE )
        print('total_pic:{0}x4'.format(len(test_loader.dataset)))
        epoch_min = 65555
        with torch.no_grad():
            for epoch in range(EPOCH_TOTAL):
                test_epoch_loss = 0
                ref_exp_net.eval()
                ref_clo_net1.eval()
                ref_clo_net_refine1.eval()
                ref_clo_net2.eval()
                ref_clo_net_refine2.eval()
                ref_sr_net.eval()
                        
                epoch_loss1= 0
                epoch_loss2=0
                epoch_loss3 = 0
                epoch_loss_color = 0
                epoch_loss_color_uv = 0
                epoch_loss_color_sm = 0
                torch.cuda.empty_cache()
                for stept, samplet in enumerate(test_loader):
                    test_yuv_image = samplet['color']
                    
                    test_mono_image  = samplet['mono']
                    nBatch,nChan,nH,nW = test_yuv_image.size()
                    pad_H=0
                    pad_W=0
                    # if nH //64 !=0 or nW//64 !=0:
                    #     pad_H = (64-(nH%64))//2
                    #     pad_W = (64-(nW%64))//2
                    #     padding_layer_coarse = nn.ZeroPad2d((pad_W,pad_W,pad_H,pad_H))
                    #     padding_layer_fine = nn.ZeroPad2d((4*pad_W,4*pad_W,4*pad_H,4*pad_H))
                    #     test_yuv_image= padding_layer_coarse(test_yuv_image)
                    #     test_mono_image= padding_layer_fine(test_mono_image) 



                    if not Real_FLAG:
                        test_labels_image = samplet['label']
                        test_labels_color_image = samplet['label_color']

                    
                    Coarse_Scale = Rescale((test_yuv_image.shape[2],test_yuv_image.shape[3]))

                    torch.cuda.empty_cache()
                    if supported:
                        test_yuv_image = test_yuv_image.to(device)
                        test_mono_image = test_mono_image.to(device)
                        if not Real_FLAG:                        
                            test_labels_image= test_labels_image.to(device)
                            test_labels_color_image= test_labels_color_image.to(device)
                            
                    # gpu_tracker.track()
                    test_coarse_mono = Coarse_Scale(test_mono_image).to(device)
                    temp = torch.zeros(test_coarse_mono.size()).to(device)
                    fine_temp = torch.zeros(test_mono_image.size()).to(device)
                    print(test_yuv_image.shape,test_mono_image.shape)
                    test_mono_add = torch.cat((test_coarse_mono,temp,temp),1)
                    ###----------------- run RefIE
                    test_adjusted_yuv,test_scale_mask = ref_exp_net(test_yuv_image,test_coarse_mono,True)
                    test_adjusted_yuv[:,0,:,:].clamp_(min=0,max=1)
                    test_adjusted_yuv[:,1,:,:].clamp_(min=-0.5,max=0.5)
                    test_adjusted_yuv[:,2,:,:].clamp_(min=-0.5,max=0.5)
                    
                    ###----------------- run RefAT            
                    test_flow = ref_clo_net1(torch.cat((test_coarse_mono, (test_adjusted_yuv[:,0,:,:]).unsqueeze(1)),1))
                    test_flow = f.interpolate(test_flow,scale_factor=4,mode='bilinear')*20
                    # rgb_fow2 = flow2rgb(test_flow,None)
                    # plt.imsave(tensor_log_path+f'/low_flow_{stept:06d}.png',rgb_fow2.transpose(1,2,0))
            
                    test_aligned_yuv_, test_mask = warp(test_adjusted_yuv [:,:,:,:],test_flow)                    
                    test_aligned_yuv_coarse = test_aligned_yuv_ +(1-test_mask)*test_mono_add
                    test_aligned_yuv,test_fuse_mask = ref_clo_net_refine1(test_adjusted_yuv,test_aligned_yuv_coarse,test_coarse_mono,output_mask=True)
                    
                                
                    
                    test_high_color_mono_coarse_1 = f.interpolate(test_adjusted_yuv[:,0,:,:].unsqueeze(1),scale_factor=4,mode='bilinear')                 
                    test_flow_inverse_1 = ref_clo_net2(torch.cat((test_high_color_mono_coarse_1,test_mono_image),1))
                    test_flow_inverse_1 = f.interpolate(test_flow_inverse_1,scale_factor=4,mode='bilinear')*20
                    # rgb_fow= flow2rgb(test_flow_inverse_1,None)
                    # plt.imsave(tensor_log_path+f'/high_flow_{stept:06d}.png',rgb_fow.transpose(1,2,0))
            
                    test_high_color_mono_coarse_11, test_mask_inverse_1 = warp(test_mono_image,test_flow_inverse_1)
                    test_high_color_mono_coarse_11 = test_high_color_mono_coarse_11 +(1-test_mask_inverse_1)*test_high_color_mono_coarse_1
                    high_color_mono,test_fuse_high_mask= ref_clo_net_refine2(torch.cat((test_mono_image,fine_temp,fine_temp),dim=1),torch.cat((test_high_color_mono_coarse_11,fine_temp,fine_temp),dim=1),\
                                                 test_high_color_mono_coarse_1,strong_mask=False,output_mask=True)
                    test_high_color_mono_fine_1=high_color_mono[:,0,:,:].unsqueeze(1)
                    test_high_color_mono = test_high_color_mono_fine_1
                    
                    ###----------------- run RefSR
                    test_aligned_yuv[:,0,:,:].clamp_(min=0,max=1)
                    test_aligned_yuv[:,1,:,:].clamp_(min=-0.5,max=0.5)
                    test_aligned_yuv[:,2,:,:].clamp_(min=-0.5,max=0.5)
                    test_high_color_mono.clamp_(min=0,max=1)
                    test_aligned_image = ref_sr_net(test_aligned_yuv,test_mono_image) 
                    test_sr_yuv_image = ref_sr_net(test_adjusted_yuv,test_high_color_mono)

                    start_time = time.time()  
                    if not Real_FLAG:    
                        test_loss_color= criterion(test_sr_yuv_image,RGB2YUV(test_labels_color_image).to(device)) 
                        test_loss = criterion(test_aligned_image,RGB2YUV(test_labels_image).to(device))
                        test_u = criterion(test_aligned_image[:,1,:,:],RGB2YUV(test_labels_image).to(device)[:,1,:,:])
                        test_v = criterion(test_aligned_image[:,2,:,:],RGB2YUV(test_labels_image).to(device)[:,2,:,:])    
                        test_u_color = criterion(test_sr_yuv_image[:,1,:,:],RGB2YUV(test_labels_color_image).to(device)[:,1,:,:])
                        test_v_color = criterion(test_sr_yuv_image[:,2,:,:],RGB2YUV(test_labels_color_image).to(device)[:,2,:,:]) 

                        test_loss2 = torch.mean(-criterion2(test_aligned_image,RGB2YUV(test_labels_image).to(device))/2+0.5)
                        test_val = 1-criterion3(yuv2rgb(test_aligned_image).to(device),test_labels_image,levels=5)
                        test_val_color = 1-criterion3(yuv2rgb(test_sr_yuv_image).to(device),test_labels_color_image,levels=5)
                        test_loss4 = criterion4(test_aligned_image,RGB2YUV(test_labels_image).to(device))

                        test_epoch_loss += (test_loss + test_loss2).item()
                        file2.write(str(epoch)+' '+str(stept)+' '+str(test_loss_color.item())+' '+str(test_loss.item())+' '+str(test_loss2.item())+' '+str(test_u.item())+' '+str(test_v.item())+' '+str(test_val.item())+' '+str(test_loss4.item())+'\n')
                        epoch_loss1 += test_loss.item()
                        epoch_loss2 += test_val.item()
                        epoch_loss3 += (test_u +test_v)/2

                        epoch_loss_color += test_loss_color.item()
                        epoch_loss_color_uv += ((test_u_color.item() +test_v_color.item())/2)
                        epoch_loss_color_sm += test_val_color.item()
                        
                    if stept !=-1:
                        cnt=0
                        if pad_H !=0 or pad_W !=0:
                            test_yuv_image =test_yuv_image[:,:,pad_H:-pad_H,pad_W:-pad_W]
                            test_adjusted_yuv =test_adjusted_yuv[:,:,pad_H:-pad_H,pad_W:-pad_W]
                            test_high_color_mono = test_high_color_mono[:,:,4*pad_H:-4*pad_H,4*pad_W:-4*pad_W] 

                            test_mono_image =test_mono_image[:,:,4*pad_H:-4*pad_H,4*pad_W:-4*pad_W] 
                            test_aligned_image = test_aligned_image[:,:,4*pad_H:-4*pad_H,4*pad_W:-4*pad_W] 
                            test_sr_yuv_image = test_sr_yuv_image[:,:,4*pad_H:-4*pad_H,4*pad_W:-4*pad_W] 
            
                        low_res = yuv2rgb(test_yuv_image).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        low_adjust = yuv2rgb(test_adjusted_yuv).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        low_aligned = yuv2rgb(test_aligned_yuv).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        low_aligned_coarse = yuv2rgb(test_aligned_yuv_coarse).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        high_res = test_mono_image.clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).squeeze(3).numpy().astype("uint8")
                        erro_img = (1-test_mask).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        # print(test_mask_inverse_1.shape)
                        high_aligned = test_high_color_mono.clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).squeeze(3).numpy().astype("uint8")
                        high_aligned_coarse = test_high_color_mono_coarse_11.clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).squeeze(3).numpy().astype("uint8")
                        
                        high_erro_img = (1-test_mask_inverse_1).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).squeeze(3).numpy().astype("uint8")
                        out_img = yuv2rgb(test_aligned_image).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        out_color = yuv2rgb(test_sr_yuv_image).clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        if not Real_FLAG:
                            label_img = test_labels_image.clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                            label_img_color = test_labels_color_image.clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).numpy().astype("uint8")
                        # print(low_res[cnt].shape,'\n')
                        for  cnt in range(BATCH_SIZE):
                            cnt1 = stept
                            low = Image.fromarray(low_res[cnt])
                            low_adjust = Image.fromarray(low_adjust[cnt])
                            low_align = Image.fromarray(low_aligned[cnt])
                            low_align_coarse = Image.fromarray(low_aligned_coarse[cnt])
                            high = Image.fromarray(high_res[cnt])
                            high_align = Image.fromarray(high_aligned[cnt])
                            high_align_coarse = Image.fromarray(high_aligned_coarse[cnt])
                            erro_mask = Image.fromarray(erro_img[cnt])
                            high_erro_mask = Image.fromarray(high_erro_img[cnt])
                            fake = Image.fromarray(out_img[cnt])
                            high_color_mono = Image.fromarray(test_high_color_mono.clamp(min = 0,max =1).mul_(255).detach().cpu().permute(0,2,3,1).squeeze(3).numpy().astype("uint8")[cnt])
                            fake_color = Image.fromarray(out_color[cnt])
                            

                            low.save(tensor_log_path+'/input-color-'+str(cnt1).zfill(6)+'.png')
                            high.save(tensor_log_path+'/input-mono-'+str(cnt1).zfill(6)+'.png')
                            fake.save(tensor_log_path+'/fake-'+str(cnt1).zfill(6)+'.png')
                            fake_color.save(tensor_log_path+'/fake-color-'+str(cnt1).zfill(6)+'.png')
                            if not Real_FLAG:
                                real = Image.fromarray(label_img[cnt])
                                real_color = Image.fromarray(label_img_color[cnt])
                                real.save(tensor_log_path+'/real-'+str(cnt1).zfill(6)+'.png')
                                real_color.save(tensor_log_path+'/real-color'+str(cnt1).zfill(6)+'.png')

                        torch.cuda.empty_cache()   
                    epoch = stept
                   
                
                test_epoch_loss = test_epoch_loss/(stept+1)    
                
                torch.cuda.empty_cache()
                epoch_loss1 = epoch_loss1 / (stept+1)
                epoch_loss2 = epoch_loss2 / (stept+1)
                epoch_loss3 = epoch_loss3 / (stept+1)
                epoch_loss4 = epoch_loss_color / (stept+1)
                epoch_loss5 = epoch_loss_color_uv / (stept+1)
                epoch_loss6 = epoch_loss_color_sm / (stept+1)
                err = 10 * math.log10(1/epoch_loss1)
                err2 = 10 * math.log10(1/epoch_loss3)
                err3 = 10 * math.log10(1/epoch_loss4)
                err4 = 10 * math.log10(1/epoch_loss5)

            file2.close()
           


import argparse
if __name__ == "__main__":
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir_set',type=int,default=4,help='main')
    parser.add_argument('--view_baseline_set',type=int,default=15,help='dual with new model pipeline')
    args = parser.parse_args()
    # test()               
    # test_bilateral()
    # test_new_bilateral(args)
    # test_new_bilateral_video(args)
    test_new_bilateral_simulate(args)
    # test_bilateral_for_comparison()
    # test_new_bilateral_add_noise()