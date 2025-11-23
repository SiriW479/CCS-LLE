from torch.utils.data import Dataset, DataLoader
import os 
import numpy as np
from PIL import Image
import torch
import cv2
class PostProDataset(Dataset):
    def __init__(self,root_dict,color_transform = None,label_transform = None,max_num=39999):
        self.predicted_dir = root_dict['predicted']
        self.predicted_prefix = root_dict['predicted_prefix']
        self.gt_dir = root_dict['gt']
        self.gt_prefix = root_dict['gt_prefix']
        self.predict_pic_list= []
        self.label_pic_list= []

        for filename in os.listdir(self.predicted_dir):
            if self.predicted_prefix in filename:
                # print(filename)
                self.predict_pic_list.append(os.path.join(self.predicted_dir, filename))
        self.predict_pic_list.sort()
        for filename1 in os.listdir(self.gt_dir):
            if self.gt_prefix in filename1:
                self.label_pic_list.append(os.path.join(self.gt_dir, filename1))
        self.label_pic_list.sort()
        self.num = len(self.predict_pic_list)
        print(self.predict_pic_list,self.label_pic_list)
        self.color_transform = color_transform
        self.label_transform = label_transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # print(self.left_pic_list[idx],self.right_pic_list[idx])
        if np.array(Image.open(self.predict_pic_list[idx])) is None:
            color_image = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_pic_list[idx])
        else:
            color_image = np.array(Image.open(self.predict_pic_list[idx])).astype(np.float32).transpose(2,0,1)/255.0
                  
        if np.array(Image.open(self.label_pic_list[idx])) is None:   
            label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.label_pic_list[idx])
        else:
            label_source = np.array(Image.open(self.label_pic_list[idx])).astype(np.float32).transpose(2,0,1)/255.0
        # else:s
            # print(self.right_pic_list[idx])
        colorTensor = torch.from_numpy(color_image)
        labelTensor = torch.from_numpy(label_source)
        if self.color_transform:
            colorTensor= self.color_transform(colorTensor)
        if self.label_transform:
            labelTensor = self.label_transform(labelTensor)

        sample = {'predict':colorTensor,'label':labelTensor}
        return sample

class myTestEnhanceDataset(Dataset):
    def __init__(self,root_txt,line_index,color_adjust=1,mono_adjust=1,color_transform = None,mono_transform = None,label_transform = None,noise_std=None):
        self.left_pic_list = []
        self.right_pic_list = []
        self.label_pic_list = []
        with open(root_txt,'r') as file:
            i = 0 
            for line in file.readlines():
                if line_index<0:
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[0])
                    self.right_pic_list.append(curLine[1])
                    self.label_pic_list.append(curLine[2])
                elif i ==line_index:                    
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[0])
                    self.right_pic_list.append(curLine[1])
                    self.label_pic_list.append(curLine[2])
                i+=1
        file.close()
        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform
        self.label_transform = label_transform
        self.noise_std=noise_std
        self.color_adjust=color_adjust
        self.mono_adjust=mono_adjust


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if Image.open(self.left_pic_list[idx]) is None:
            color_image = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_pic_list[idx])
        else:
            color_image = np.array(Image.open(self.left_pic_list[idx]).rotate(0, expand = True)).astype(np.float32).transpose(2,0,1)/255.0

        if Image.open(self.right_pic_list[idx]) is None:   
            mono_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_pic_list[idx])
        else:

            mono_source = np.array(Image.open(self.right_pic_list[idx]).rotate(0, expand = True)).astype(np.float32).transpose(2,0,1)/255.0
        if Image.open(self.label_pic_list[idx]) is None:   
            label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.label_pic_list[idx])
        else:
            label_source = np.array(Image.open(self.label_pic_list[idx]).rotate(0, expand = True)).astype(np.float32).transpose(2,0,1)/255.0

        color_adjust = self.color_adjust
        color_image = color_image*color_adjust

        colorTensor = torch.from_numpy(color_image).clamp(min=0,max=1)
        if self.noise_std:
            equal_noise = torch.randn(colorTensor.shape)
            colorTensor = self.noise_std*equal_noise+colorTensor

        mono_adjust = self.mono_adjust
        mono_source *=mono_adjust
        label_source *= mono_adjust

        monoTensor_ = torch.from_numpy(mono_source).clamp(min=0,max=1)
        h,w = monoTensor_.shape[-2:]
        monoTensor = torch.zeros(1,h,w)
        # cv input:RGB
        monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        labelTensor = torch.from_numpy(label_source).clamp(min=0,max=1)
        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)
        if self.label_transform:
          labelTensor = self.label_transform(labelTensor)


        sample = {'mono':monoTensor,'color':colorTensor,'label':labelTensor}
        return sample

class err_myTestBilateralDataset(Dataset):
    def __init__(self,root_txt,line_index,color_transform = None,mono_transform = None,mono_label_transform = None,color_label_transform =None,noise_std = None, max_num=39999):
        self.left_pic_list = []
        self.right_pic_list = []
        self.left_label_pic_list = []
        self.right_label_pic_list = []
        with open(root_txt,'r') as file:
            i = 0 
            for line in file.readlines():
                if line_index<0:
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[2])
                    self.right_pic_list.append(curLine[1])
                    self.right_label_pic_list.append(curLine[1])
                    self.left_label_pic_list.append(curLine[2])
                elif i ==line_index:                    
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[2])
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
        self.noise_std=noise_std

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
        if self.noise_std:
            equal_noise = torch.randn(colorTensor.shape)
            colorTensor = self.noise_std*equal_noise+colorTensor



        monoTensor = torch.from_numpy(mono_source)
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

class myTestBilateralDataset(Dataset):
    def __init__(self,root_txt ,line_index,color_transform = None,mono_transform = None,mono_label_transform = None,color_label_transform =None,noise_std = None,  max_num=39999):
        self.left_pic_list = []
        self.right_pic_list = []
        self.left_label_pic_list = []
        self.right_label_pic_list = []
        with open(root_txt,'r') as file:
            i = 0 
            for line in file.readlines():
                if line_index<0:
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[2])
                    self.right_pic_list.append(curLine[1])
                    self.right_label_pic_list.append(curLine[1])
                    self.left_label_pic_list.append(curLine[2])
                elif i ==line_index:                    
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[2])
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
        self.noise_std=noise_std


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # print(self.left_pic_list[idx],self.right_pic_list[idx])
        rotate_angle=[0,90,180,270]
        # rotate_index = np.random.randint(low=0, high= 4, size=None, dtype='l')
        rotate_index = 0
        if cv2.imread(self.left_pic_list[idx]) is None:
            color_image = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_pic_list[idx])
        else:
            # color_image = cv2.imread(self.left_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            color_image = np.array(Image.open(self.left_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
            # _,ih,iw =color_image.shape 
            # coarse_color_image = np.array(Image.open(self.left_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True).resize((iw//4, ih//4),Image.ANTIALIAS)).astype(np.float32).transpose(2,0,1)/255.0       
            # print(color_image.shape,coarse_color_image.shape)
        # else:
            # print(self.left_pic_list[idx])
        if cv2.imread(self.right_pic_list[idx]) is None:   
            mono_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_pic_list[idx])
        else:
            # mono_source = cv2.imread(self.right_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            mono_source = np.array(Image.open(self.right_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
            # mono_source = np.array(Image.open(self.right_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32)/255.0
        if cv2.imread(self.right_label_pic_list[idx]) is None:   
            mono_label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_label_pic_list[idx])
        else:
            # label_source = cv2.imread(self.label_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            mono_label_source = np.array(Image.open(self.right_label_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
        if cv2.imread(self.left_label_pic_list[idx]) is None:   
            color_label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_label_pic_list[idx])
        else:
            # label_source = cv2.imread(self.label_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            color_label_source = np.array(Image.open(self.left_label_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
        

        colorTensor = torch.from_numpy(color_image)
        # coarse_colorTensor = torch.from_numpy(coarse_color_image)
        monoTensor = torch.from_numpy(mono_source)
        mono_labelTensor = torch.from_numpy(mono_label_source)
        color_labelTensor = torch.from_numpy(color_label_source)
        # print(monoTensor.size(),colorTensor.size())
        # print(mono_labelTensor.size(),color_labelTensor.size())
        # h,w = monoTensor_.shape[-2:]
        # monoTensor = torch.zeros(1,h,w)
        # monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        
        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        #   coarse_colorTensor= self.color_transform(coarse_colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)
        if self.mono_label_transform:
          mono_labelTensor = self.mono_label_transform(mono_labelTensor)
        if self.color_label_transform:
          color_labelTensor = self.color_label_transform(color_labelTensor)
       
        sample = {'mono':monoTensor,'color':colorTensor,'label':mono_labelTensor,'label_color':color_labelTensor}
        # sample = {'mono':monoTensor1,'color':colorTensor1,'label':labelTensor1}
        # print('XXXXXXXXXXX',colorTensor1.shape,monoTensor1.shape,labelTensor1.shape)
        
        return sample

class myTestOverallBilateralDataset(Dataset):
    def __init__(self,root_txt ,line_index,color_transform = None,mono_transform = None,mono_label_transform = None,color_label_transform =None,noise_std = None,  max_num=39999):
        self.left_pic_list = []
        self.right_pic_list = []
        self.left_label_pic_list = []
        self.right_label_pic_list = []
        with open(root_txt,'r') as file:
            i = 0 
            for line in file.readlines():
                if line_index<0:
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[0])
                    self.right_pic_list.append(curLine[1])
                    self.right_label_pic_list.append(curLine[1])
                    self.left_label_pic_list.append(curLine[2])
                elif i ==line_index:                    
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
        self.noise_std=noise_std


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # print(self.left_pic_list[idx],self.right_pic_list[idx])
        rotate_angle=[0,90,180,270]
        # rotate_index = np.random.randint(low=0, high= 4, size=None, dtype='l')
        rotate_index = 0
        if cv2.imread(self.left_pic_list[idx]) is None:
            color_image = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_pic_list[idx])
        else:
            # color_image = cv2.imread(self.left_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            color_image = np.array(Image.open(self.left_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
            # _,ih,iw =color_image.shape 
            # coarse_color_image = np.array(Image.open(self.left_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True).resize((iw//4, ih//4),Image.ANTIALIAS)).astype(np.float32).transpose(2,0,1)/255.0       
            # print(color_image.shape,coarse_color_image.shape)
        # else:
            # print(self.left_pic_list[idx])
        if cv2.imread(self.right_pic_list[idx]) is None:   
            mono_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_pic_list[idx])
        else:
            # mono_source = cv2.imread(self.right_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            mono_source = np.array(Image.open(self.right_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
            # mono_source = np.array(Image.open(self.right_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32)/255.0
        if cv2.imread(self.right_label_pic_list[idx]) is None:   
            mono_label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_label_pic_list[idx])
        else:
            # label_source = cv2.imread(self.label_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            mono_label_source = np.array(Image.open(self.right_label_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
        if cv2.imread(self.left_label_pic_list[idx]) is None:   
            color_label_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_label_pic_list[idx])
        else:
            # label_source = cv2.imread(self.label_pic_list[idx]).astype(np.float32).transpose(2,0,1)/255.0
            color_label_source = np.array(Image.open(self.left_label_pic_list[idx]).rotate(rotate_angle[rotate_index], expand = True)).astype(np.float32).transpose(2,0,1)/255.0
        

        colorTensor = torch.from_numpy(color_image)
        # coarse_colorTensor = torch.from_numpy(coarse_color_image)
        monoTensor_ = torch.from_numpy(mono_source)
        mono_labelTensor = torch.from_numpy(mono_label_source)
        color_labelTensor = torch.from_numpy(color_label_source)
        # print(monoTensor.size(),colorTensor.size())
        # print(mono_labelTensor.size(),color_labelTensor.size())
        h,w = monoTensor_.shape[-2:]
        monoTensor = torch.zeros(1,h,w)
        monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        
        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        #   coarse_colorTensor= self.color_transform(coarse_colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)
        if self.mono_label_transform:
          mono_labelTensor = self.mono_label_transform(mono_labelTensor)
        if self.color_label_transform:
          color_labelTensor = self.color_label_transform(color_labelTensor)
       
        sample = {'mono':monoTensor,'color':colorTensor,'label':mono_labelTensor,'label_color':color_labelTensor}
        # sample = {'mono':monoTensor1,'color':colorTensor1,'label':labelTensor1}
        # print('XXXXXXXXXXX',colorTensor1.shape,monoTensor1.shape,labelTensor1.shape)
        
        return sample

class RealCaptureDataset(Dataset):
    def __init__(self,root_dir,color_transform = None,mono_transform = None,max_num=39999):
        img_set = os.listdir(root_dir)        
        self.left_pic_list = []
        self.right_pic_list = []
        for i in range(len(img_set)//2):
            self.left_pic_list.append(root_dir+'/'+str(i).zfill(4)+'_L.png')
            self.right_pic_list.append(root_dir+'/'+str(i).zfill(4)+'_R.png')

        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        print(self.left_pic_list[idx],self.right_pic_list[idx])
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

        colorTensor = torch.from_numpy(color_image)
        monoTensor = torch.from_numpy(mono_source)[0,:,:].unsqueeze(0)
        

        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)


        # mono_out_patch,decolor_out_patch,index_out = self.searchArea(monoTensor[0,:,:],colorTensor[0,:,:])

        sample = {'mono':monoTensor,'color':colorTensor}
        return sample

class RealCaptureGopDataset(Dataset):
    def __init__(self,root_dir,color_transform = None,mono_transform = None,max_num=39999,GopSize=1):
        img_set = os.listdir(root_dir)        
        self.left_pic_list = []
        self.right_pic_list = []
        for i in range(0,len(img_set)//2,GopSize):
            for j in range(GopSize):
                self.left_pic_list.append(root_dir+'/'+str(i).zfill(4)+'_L.png')
                self.right_pic_list.append(root_dir+'/'+str(i+j).zfill(4)+'_R.png')

        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        print(self.left_pic_list[idx],self.right_pic_list[idx])
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

        colorTensor = torch.from_numpy(color_image)
        monoTensor = torch.from_numpy(mono_source)[0,:,:].unsqueeze(0)
        

        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)


        # mono_out_patch,decolor_out_patch,index_out = self.searchArea(monoTensor[0,:,:],colorTensor[0,:,:])

        sample = {'mono':monoTensor,'color':colorTensor}
        return sample



class RealDualCaptureDataset(Dataset):
    def __init__(self,root_dir,color_transform = None,mono_transform = None,max_num=39999):
        img_set = os.listdir(root_dir+'/left/')        
        self.left_pic_list = []
        self.right_pic_list = []
        for i in range(len(img_set)):
            self.left_pic_list.append(root_dir+'/left/'+str(i).zfill(4)+'_L.png')
            self.right_pic_list.append(root_dir+'/right/'+str(i).zfill(4)+'_R.png')

        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        print(self.left_pic_list[idx],self.right_pic_list[idx])
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

        colorTensor = torch.from_numpy(color_image)
        monoTensor = torch.from_numpy(mono_source)[0,:,:].unsqueeze(0)
        

        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)


        # mono_out_patch,decolor_out_patch,index_out = self.searchArea(monoTensor[0,:,:],colorTensor[0,:,:])

        sample = {'mono':monoTensor,'color':colorTensor}
        return sample


class myEnhanceBilateralDataset(Dataset):
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
        color_adjust = np.random.uniform(0.5,1.5)
        color_image *= color_adjust
        colorTensor = torch.from_numpy(color_image).clamp(min=0,max=1)
        mono_adjust = np.random.uniform(0.9,1.2)
        mono_source *=mono_adjust
        mono_label_source *= mono_adjust
        color_label_source *= mono_adjust

        monoTensor_ = torch.from_numpy(mono_source).clamp(min=0,max=1)
        h,w = monoTensor_.shape[-2:]
        monoTensor = torch.zeros(1,h,w)
        # monoTensor =monoTensor_.unsqueeze(0)
        monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        mono_labelTensor = torch.from_numpy(mono_label_source).clamp(min=0,max=1)
        color_labelTensor = torch.from_numpy(color_label_source).clamp(min=0,max=1)


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



class myEnhanceISPBilateralDataset(Dataset):
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
        color_adjust = np.random.uniform(0.5,1.5)
        color_image *= color_adjust
        colorTensor = torch.from_numpy(color_image).clamp(min=0,max=1)
        mono_adjust = np.random.uniform(0.9,1.2)
        mono_source *=mono_adjust
        mono_label_source *= mono_adjust
        color_label_source *= mono_adjust

        monoTensor_ = torch.from_numpy(mono_source).clamp(min=0,max=1)
        h,w = monoTensor_.shape[-2:]
        # monoTensor = torch.zeros(1,h,w)
        # monoTensor =monoTensor_.unsqueeze(0)
        # monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        mono_labelTensor = torch.from_numpy(mono_label_source).clamp(min=0,max=1)
        color_labelTensor = torch.from_numpy(color_label_source).clamp(min=0,max=1)


        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor_ = self.mono_transform(monoTensor_)
          monoTensor = torch.zeros(1,monoTensor_.shape[-2],monoTensor_.shape[-1])
          monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        if self.mono_label_transform:
          mono_labelTensor = self.mono_label_transform(mono_labelTensor)
        if self.color_label_transform:
          color_labelTensor = self.color_label_transform(color_labelTensor)

        # mono_out_patch,decolor_out_patch,index_out = self.searchArea(monoTensor[0,:,:],colorTensor[0,:,:])

        sample = {'mono':monoTensor,'color':colorTensor,'label':mono_labelTensor,'label_color':color_labelTensor}
        return sample


class myTestRealColorDataset(Dataset):
    def __init__(self,root_txt,line_index,color_adjust=1,mono_adjust=1,color_transform = None,mono_transform = None,label_transform = None,noise_std=None):
        self.left_pic_list = []
        self.right_pic_list = []
        self.label_pic_list = []
        with open(root_txt,'r') as file:
            i = 0 
            for line in file.readlines():
                if line_index<0:
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[0])
                    self.right_pic_list.append(curLine[1])
                    # self.label_pic_list.append(curLine[2])
                elif i ==line_index:                    
                    curLine=line.strip().split(" ")
                    self.left_pic_list.append(curLine[0])
                    self.right_pic_list.append(curLine[1])
                    break
                    # self.label_pic_list.append(curLine[2])
                i+=1
                print(line)
        file.close()
        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform
        # self.label_transform = label_transform
        self.noise_std=noise_std
        self.color_adjust=color_adjust
        self.mono_adjust=mono_adjust


    def __len__(self):
        return self.num

    def __getitem__(self, idx):

        if Image.open(self.left_pic_list[idx]) is None:
            color_image = np.zeros((3,240,240)).astype(np.float32)
            print(self.left_pic_list[idx])
        else:
            color_image = np.array(Image.open(self.left_pic_list[idx]).rotate(0, expand = True)).astype(np.float32).transpose(2,0,1)/255.0

        if Image.open(self.right_pic_list[idx]) is None:   
            mono_source = np.zeros((3,240,240)).astype(np.float32)
            print(self.right_pic_list[idx])
        else:

            mono_source = np.array(Image.open(self.right_pic_list[idx]).rotate(0, expand = True)).astype(np.float32)/255.0
        # if Image.open(self.label_pic_list[idx]) is None:   
        #     label_source = np.zeros((3,240,240)).astype(np.float32)
        #     print(self.label_pic_list[idx])
        # else:
        #     label_source = np.array(Image.open(self.label_pic_list[idx]).rotate(0, expand = True)).astype(np.float32).transpose(2,0,1)/255.0

        color_adjust = self.color_adjust
        color_image = color_image*color_adjust

        colorTensor = torch.from_numpy(color_image).clamp(min=0,max=1)
        if self.noise_std:
            equal_noise = torch.randn(colorTensor.shape)
            colorTensor = self.noise_std*equal_noise+colorTensor

        mono_adjust = self.mono_adjust
        mono_source *=mono_adjust
        # label_source *= mono_adjust

        monoTensor = torch.from_numpy(mono_source).clamp(min=0,max=1).unsqueeze(0)
        # h,w = monoTensor_.shape[-2:]
        # monoTensor = torch.zeros(1,h,w)
        # # cv input:RGB
        # monoTensor[0,:,:] = monoTensor_[2,:,:] * 0.114 + monoTensor_[1,:,:] * 0.587 + monoTensor_[0,:,:]*0.299
        # labelTensor = torch.from_numpy(label_source).clamp(min=0,max=1)
        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)
        # if self.label_transform:
        #   labelTensor = self.label_transform(labelTensor)


        sample = {'mono':monoTensor,'color':colorTensor}
        return sample

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

class RealDualCaptureDataset(Dataset):
    def __init__(self,root_dir,color_transform = None,mono_transform = None,max_num=39999):
        img_set = os.listdir(root_dir+'/left/')        
        self.left_pic_list = []
        self.right_pic_list = []
        for i in range(len(img_set)):
            self.left_pic_list.append(root_dir+'/left/'+str(i).zfill(4)+'_L.png')
            self.right_pic_list.append(root_dir+'/right/'+str(i).zfill(4)+'_R.png')

        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        print(self.left_pic_list[idx],self.right_pic_list[idx])
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

        colorTensor = torch.from_numpy(color_image)
        monoTensor = torch.from_numpy(mono_source)[0,:,:].unsqueeze(0)
        

        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)


        # mono_out_patch,decolor_out_patch,index_out = self.searchArea(monoTensor[0,:,:],colorTensor[0,:,:])

        sample = {'mono':monoTensor,'color':colorTensor}
        return sample

class TestDataset(Dataset):
    def __init__(self,left_image,right_image,color_transform = None,mono_transform = None,max_num=39999):
     
        self.left_pic_list = []
        self.right_pic_list = []

        self.left_pic_list.extend(left_image)
        self.right_pic_list.extend(right_image)

        self.num = len(self.left_pic_list)
        self.color_transform = color_transform
        self.mono_transform = mono_transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        print(self.left_pic_list[idx],self.right_pic_list[idx])
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
            mono_source = np.array(Image.open(self.right_pic_list[idx])).astype(np.float32)/255.0

        colorTensor = torch.from_numpy(color_image)
        monoTensor = torch.from_numpy(mono_source).unsqueeze(0)
        

        if self.color_transform:
          colorTensor= self.color_transform(colorTensor)
        if self.mono_transform:
          monoTensor = self.mono_transform(monoTensor)


        # mono_out_patch,decolor_out_patch,index_out = self.searchArea(monoTensor[0,:,:],colorTensor[0,:,:])

        sample = {'mono':monoTensor,'color':colorTensor}
        return sample



if __name__ == "__main__":
    print('start')
    root_dir = {'predicted':'','predicted_prefix':'','gt':'','gt_prefix':''}
    root_dir['predicted'] = '/gpy/real-world-sr-master/esrgan-fs/results/AIM2019_TDSR/val'
    root_dir['predicted_prefix'] = 'real-color'
    root_dir['gt'] = '/gpy/data/mb14/HQ'
    root_dir['gt_prefix'] = 'real-color'
    test_dataset = PostProDataset(root_dir)
    