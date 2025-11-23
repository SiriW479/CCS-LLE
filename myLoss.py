from PWCNet import predict_flow
from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as f
import math
def Convert2PSNR(loss_Value):
    err= 10 * math.log10(1/loss_Value.item())
    return err

class YUV_Loss(nn.Module):
    def __init__(self, eps=1e-7,  threshold_uv=0.005):
        super(YUV_Loss, self).__init__()
        self.threshold_uv = threshold_uv
        self.loss = nn.L1Loss(reduction='mean')
        self.eps = eps

        ## take the Y loss into consideration priorly 
        #  and when it drops to threshold_uv and then add the uv loss

    def forward(self, predict,label):
        loss_y = self.loss(predict[:,0,:,:].unsqueeze(1), label[:,0,:,:].unsqueeze(1))
        loss_u =  self.loss(predict[:,1,:,:].unsqueeze(1), label[:,1,:,:].unsqueeze(1))
        loss_v =  self.loss(predict[:,2,:,:].unsqueeze(1), label[:,2,:,:].unsqueeze(1))
        if loss_y.item() > self.threshold_uv:   
            total_loss = torch.mean(loss_y+0.2*loss_u+0.2*loss_v)
        else:
            total_loss = torch.mean(loss_y+loss_u+loss_v) 

        return total_loss

class PSNR_loss(nn.Module):
    def __init__(self,flag_YUV=False,flag_UV=False,flag_Y=False):
        super(PSNR_loss, self).__init__()
        self.flag_YUV = flag_YUV
        self.flag_UV = flag_UV
        self.flag_Y = flag_Y
        # self.flag_RGB = flag_RGB
        self.loss = torch.nn.MSELoss()
        
    def forward(self, predict,label):
        if predict.is_cuda:
            self.loss.cuda()
        if self.flag_YUV:
            loss = self.loss(predict,label)
        if self.flag_UV:
            loss = self.loss(predict[:,1:,:,:],label[:,1:,:,:])
        if self.flag_Y:
            loss = self.loss(predict[:,0,:,:].unsqueeze(1),label[:,0,:,:].unsqueeze(1))         
        return loss

class Image_smooth_loss(nn.Module):
    '''
    https://github.com/jianfenglihg/UnOpticalFlow/blob/master/core/networks/model_flow_paper.py#L215
    '''
    def __init__(self,TV_scale=10.0):
        super(Image_smooth_loss,self).__init__()
        self.TV_scale = TV_scale

    def gradients(self,img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy
      
    def forward(self,predicted,label):
        batch_size = predicted.size()[0]
        h_predicted = predicted.size()[2]
        w_predicted = predicted.size()[3]
        
        predicted_grad_x,predicted_grad_y = self.gradients(predicted)
        label_grad_x,label_grad_y = self.gradients(label)
        w_x = torch.exp(-self.TV_scale*torch.abs(label_grad_x))
        w_y = torch.exp(-self.TV_scale*torch.abs(label_grad_y))
        
        error = ((w_x*torch.abs(predicted_grad_x)).mean((1,2,3)) +(w_y*torch.abs(predicted_grad_y)).mean((1,2,3))).sum()/batch_size
        return error

class Flow_Image_smooth_loss(nn.Module):
    '''
    https://github.com/jianfenglihg/UnOpticalFlow/blob/master/core/networks/model_flow_paper.py#L215
    input: predicted-flow
           img:(NX1XHXW) template image
    '''
    def __init__(self,TV_scale=10.0):
        super(Flow_Image_smooth_loss,self).__init__()
        self.TV_scale = TV_scale

    def gradients(self,img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy
      
    def forward(self,predicted,label):
        batch_size = predicted.size()[0]
        h_predicted = predicted.size()[2]
        w_predicted = predicted.size()[3]

        predicted_label_mag_x,predicted_label_mag_y = self.gradients(label)
        predicted_label_mag_x =predicted_label_mag_x.abs()
        predicted_label_mag_y =predicted_label_mag_y.abs()
        maximum_predicted_label_mag_x,_ = torch.max(\
                                        torch.max(predicted_label_mag_x,dim=3,keepdim= True,out= None)[0],\
                                        dim=2,keepdim= True,out= None)
        maximum_predicted_label_mag_y,_ = torch.max(\
                                        torch.max(predicted_label_mag_y,dim=3,keepdim= True,out= None)[0],\
                                        dim=2,keepdim= True,out= None)
        # maximum_predicted_label_mag_y,_ = torch.max(predicted_label_mag_y,dim=(1,2,3),keepdim= True,out= None)
        normal_predicted_label_mag_x = predicted_label_mag_x/(maximum_predicted_label_mag_x+1e-7)
        normal_predicted_label_mag_y = predicted_label_mag_y/(maximum_predicted_label_mag_y+1e-7)
        w_x = torch.exp(-self.TV_scale*torch.abs(normal_predicted_label_mag_x))
        w_y = torch.exp(-self.TV_scale*torch.abs(normal_predicted_label_mag_y))

        predicted_grad_x,predicted_grad_y = self.gradients(predicted)
        predicted_grad_x[:,0,:,:]/=float(w_predicted)
        predicted_grad_y[:,0,:,:]/=float(w_predicted)
        predicted_grad_x[:,1,:,:]/=float(h_predicted)
        predicted_grad_y[:,1,:,:]/=float(h_predicted)
                
        error = ((w_x*torch.abs(predicted_grad_x)).mean((1,2,3)) +(w_y*torch.abs(predicted_grad_y)).mean((1,2,3))).sum()/batch_size
        return error

class Flow_smooth_loss(nn.Module):
    '''
    https://github.com/jianfenglihg/UnOpticalFlow/blob/master/core/networks/model_flow_paper.py#L215
    '''
    def __init__(self,TV_scale=10.0):
        super(Flow_smooth_loss,self).__init__()
        self.TV_scale = TV_scale

    def gradients(self,img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy
      
    def forward(self,predicted,label):
        batch_size = predicted.size()[0]
        h_predicted = predicted.size()[2]
        w_predicted = predicted.size()[3]
        
        label_mag= ((label[:,0,:,:]**2+label[:,1,:,:]**2)**0.5).unsqueeze(1)
        predicted_label_mag_x,predicted_label_mag_y = self.gradients(label_mag)
        predicted_label_mag_x =predicted_label_mag_x.abs()
        predicted_label_mag_y =predicted_label_mag_y.abs()
        maximum_predicted_label_mag_x,_ = torch.max(\
                                        torch.max(predicted_label_mag_x,dim=3,keepdim= True,out= None)[0],\
                                        dim=2,keepdim= True,out= None)
        maximum_predicted_label_mag_y,_ = torch.max(\
                                        torch.max(predicted_label_mag_y,dim=3,keepdim= True,out= None)[0],\
                                        dim=2,keepdim= True,out= None)
        # maximum_predicted_label_mag_y,_ = torch.max(predicted_label_mag_y,dim=(1,2,3),keepdim= True,out= None)
        normal_predicted_label_mag_x = predicted_label_mag_x/(maximum_predicted_label_mag_x+1e-7)
        normal_predicted_label_mag_y = predicted_label_mag_y/(maximum_predicted_label_mag_y+1e-7)
        w_x = torch.exp(-self.TV_scale*torch.abs(normal_predicted_label_mag_x))
        w_y = torch.exp(-self.TV_scale*torch.abs(normal_predicted_label_mag_y))

        predicted_grad_x,predicted_grad_y = self.gradients(predicted)
        predicted_grad_x[:,0,:,:]/=float(w_predicted)
        predicted_grad_y[:,0,:,:]/=float(w_predicted)
        predicted_grad_x[:,1,:,:]/=float(h_predicted)
        predicted_grad_y[:,1,:,:]/=float(h_predicted)
                
        error = ((w_x*torch.abs(predicted_grad_x)).mean((1,2,3)) +(w_y*torch.abs(predicted_grad_y)).mean((1,2,3))).sum()/batch_size
        return error

class L_tv_guide(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_tv_guide,self).__init__()
        self.TVLoss_weight = TVLoss_weight
    
      ## x is the adjustment matrix,yuv is the input image
    def forward(self,x,yuv,channel=True):
        LOSS_N = 1
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_yuv_tv =torch.abs(yuv[:,:,1:,:w_x-1]-yuv[:,:,:h_x-1,:w_x-1]) 
        w_yuv_tv = torch.abs(yuv[:,:,:h_x-1,1:]-yuv[:,:,:h_x-1,:w_x-1]) 
        yuv_tv = torch.pow(h_yuv_tv,LOSS_N )+ torch.pow(w_yuv_tv,LOSS_N )
        
        flag_smooth = torch.zeros(yuv_tv.shape)
        if x.is_cuda:
            flag_smooth = flag_smooth.cuda()
        if channel:
            yuv_tv_mean = torch.mean(yuv_tv,[-1,-2],keepdim=True)
            flag_smooth[yuv_tv<yuv_tv_mean] = 1.0
        else:
            yuv_tv_mean = torch.mean(yuv_tv)
            flag_smooth[yuv_tv<yuv_tv_mean] = 1.0

        h_tv =torch.abs(x[:,:,1:,:w_x-1]-x[:,:,:h_x-1,:w_x-1]) 
        w_tv = torch.abs(x[:,:,:h_x-1,1:]-x[:,:,:h_x-1,:w_x-1]) 
        x_tv = torch.pow(h_tv,LOSS_N )+ torch.pow(w_tv,LOSS_N )
        loss =  self.TVLoss_weight*torch.mean(x_tv*flag_smooth)
        # 
        # count_h =  (x.size()[2]-1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return loss

class L_tv_mean(nn.Module):
    def __init__(self,TVLoss_weight=1,average_size=(1,1),crit_mode='mean'):
        super(L_tv_mean,self).__init__()
        self.crit  = nn.L1Loss(reduction=crit_mode)
        self.average_pool = nn.AdaptiveMaxPool2d(average_size)
      ## x is the adjustment matrix,yuv is the other view image
    def forward(self,x,yuv,channel=True):
        LOSS_N = 1
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_yuv_tv =torch.abs(yuv[:,:,1:,:w_x-1]-yuv[:,:,:h_x-1,:w_x-1]) 
        w_yuv_tv = torch.abs(yuv[:,:,:h_x-1,1:]-yuv[:,:,:h_x-1,:w_x-1]) 
        yuv_tv = torch.pow(h_yuv_tv,LOSS_N )+ torch.pow(w_yuv_tv,LOSS_N )
        global_yuv_tv = self.average_pool(yuv_tv)

        h_tv =torch.abs(x[:,:,1:,:w_x-1]-x[:,:,:h_x-1,:w_x-1]) 
        w_tv = torch.abs(x[:,:,:h_x-1,1:]-x[:,:,:h_x-1,:w_x-1]) 
        x_tv = torch.pow(h_tv,LOSS_N )+ torch.pow(w_tv,LOSS_N )
        global_x_tv = self.average_pool(x_tv)
        loss =  self.crit(global_x_tv,global_yuv_tv)
        # 
        # count_h =  (x.size()[2]-1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return loss

class L_tv_sigmoid_mean(nn.Module):
    def __init__(self,TVLoss_weight=1,average_size=(1,1),crit_mode='mean'):
        super(L_tv_sigmoid_mean,self).__init__()
        self.crit  = nn.L1Loss(reduction=crit_mode)
        self.average_pool = nn.AdaptiveMaxPool2d(average_size)
      ## x is the adjustment matrix,yuv is the other view image
    def forward(self,x,yuv,channel=True):
        LOSS_N = 1
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_yuv_tv =f.sigmoid(yuv[:,:,1:,:w_x-1]-yuv[:,:,:h_x-1,:w_x-1]) 
        w_yuv_tv = f.sigmoid(yuv[:,:,:h_x-1,1:]-yuv[:,:,:h_x-1,:w_x-1]) 
        yuv_tv = torch.pow(h_yuv_tv,LOSS_N )+ torch.pow(w_yuv_tv,LOSS_N )
        global_yuv_tv = self.average_pool(yuv_tv)

        h_tv =f.sigmoid(x[:,:,1:,:w_x-1]-x[:,:,:h_x-1,:w_x-1]) 
        w_tv = f.sigmoid(x[:,:,:h_x-1,1:]-x[:,:,:h_x-1,:w_x-1]) 
        x_tv = torch.pow(h_tv,LOSS_N )+ torch.pow(w_tv,LOSS_N )
        global_x_tv = self.average_pool(x_tv)
        loss =  self.crit(global_x_tv,global_yuv_tv)
        # 
        # count_h =  (x.size()[2]-1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return loss

class L_tv_kl(nn.Module):
    def __init__(self,TVLoss_weight=1,average_size=(1,1),crit_mode='mean'):
        super(L_tv_kl,self).__init__()
        self.crit  = nn.L1Loss(reduction=crit_mode)
        # self.average_pool = nn.AdaptiveMaxPool2d(average_size)
      ## x is the adjustment matrix,yuv is the other view image
    def forward(self,x,yuv,channel=True):
        LOSS_N = 1
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_yuv_tv =(yuv[:,:,1:,:w_x-1]-yuv[:,:,:h_x-1,:w_x-1]) 
        w_yuv_tv = (yuv[:,:,:h_x-1,1:]-yuv[:,:,:h_x-1,:w_x-1]) 
        h_yuv_tv_mean = torch.mean(h_yuv_tv,[-1,-2],keepdim=True)
        h_yuv_tv_std = torch.std(h_yuv_tv,[-1,-2],keepdim=True)
        w_yuv_tv_mean = torch.mean(w_yuv_tv,[-1,-2],keepdim=True)
        w_yuv_tv_std = torch.std(w_yuv_tv,[-1,-2],keepdim=True)
        # yuv_tv = torch.pow(h_yuv_tv,LOSS_N )+ torch.pow(w_yuv_tv,LOSS_N )
        # global_yuv_tv = self.average_pool(yuv_tv)

        h_tv =(x[:,:,1:,:w_x-1]-x[:,:,:h_x-1,:w_x-1]) 
        w_tv = (x[:,:,:h_x-1,1:]-x[:,:,:h_x-1,:w_x-1]) 
        h_tv_mean = torch.mean(h_tv,[-1,-2],keepdim=True)
        h_tv_std = torch.std(h_tv,[-1,-2],keepdim=True)
        w_tv_mean = torch.mean(w_tv,[-1,-2],keepdim=True)
        w_tv_std = torch.std(w_tv,[-1,-2],keepdim=True)
        
        kl_h = torch.log(h_yuv_tv_std/h_tv_std)+ (h_tv_std**2+(h_tv_mean-h_yuv_tv_mean)**2)/(2*h_yuv_tv_std**2)-0.5
        kl_w = torch.log(w_yuv_tv_std/w_tv_std)+ (w_tv_std**2+(w_tv_mean-w_yuv_tv_mean)**2)/(2*w_yuv_tv_std**2)-0.5
        # x_tv = torch.pow(h_tv,LOSS_N )+ torch.pow(w_tv,LOSS_N )
        # global_x_tv = self.average_pool(x_tv)
        avl_kl = torch.sum(kl_h+kl_w)/batch_size
        # loss =  self.crit(global_x_tv,global_yuv_tv)
        loss = avl_kl
        # 
        # count_h =  (x.size()[2]-1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return loss

class L_kl(nn.Module):
    def __init__(self):
        super(L_kl,self).__init__()
        
    def forward(self,x,yuv):
        LOSS_N = 1
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        yuv_mean = torch.mean(yuv,[-1,-2],keepdim=True)
        yuv_std = torch.std(yuv,[-1,-2],keepdim=True)
        # yuv_tv = torch.pow(h_yuv_tv,LOSS_N )+ torch.pow(w_yuv_tv,LOSS_N )
        # global_yuv_tv = self.average_pool(yuv_tv)

        x_mean = torch.mean(x,[-1,-2],keepdim=True)
        x_std = torch.std(x,[-1,-2],keepdim=True)

        
        kl_value = torch.log(yuv_std/x_std)+ (x_std**2+(x_mean-yuv_mean)**2)/(2*yuv_std**2)-0.5
        # x_tv = torch.pow(h_tv,LOSS_N )+ torch.pow(w_tv,LOSS_N )
        # global_x_tv = self.average_pool(x_tv)
        avl_kl = torch.sum(kl_value)/batch_size
        # loss =  self.crit(global_x_tv,global_yuv_tv)
        loss = avl_kl
        # 
        # count_h =  (x.size()[2]-1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return loss

class L_ref_exp(nn.Module):

    def __init__(self,patch_size=16,lossN=1,flag_perCase=False):
        super(L_ref_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.lossN = lossN
        self.flag_perCase = flag_perCase
        
        
    def forward(self, x_y,ref_y ):

        # b,c,h,w = x.shape
        # x = torch.mean(x,1,keepdim=True)
        x_mean = self.pool(x_y)
        ref_mean = self.pool(ref_y)
        if not self.flag_perCase:
            d = torch.mean( torch.abs(torch.pow(x_mean- ref_mean,self.lossN)))
        else:
            d = torch.mean( torch.abs(torch.pow(x_mean- ref_mean,self.lossN)),dim=[-1,-2])
        return d

### below are copied from other's work
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = f.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = f.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = f.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = f.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = f.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = f.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = f.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = f.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E =  torch.mean(D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class color_consistent_loss(nn.Module):
    
    '''
    input: the channel of UV 
    performance: keep the input and output color consistency
    '''
    def __init__(self, eps=1e-7,weight_factor=1,Y_factor=5,UV_factor=1):
        super(color_consistent_loss, self).__init__()


    def forward(self, predict,label):
#         weight = torch.exp((torch.abs(predict)*torch.abs(label)-predict*label)/((predict-label)**2+self.eps))
        weight = torch.sign(-predict*label)
        weight[weight<1] =0
        weight_loss = torch.mean(weight)
        return weight_loss

class color_space_loss(nn.Module):
    
    '''
    input: the channel of UV 
    performance: keep the input and output color consistency
    '''
    def __init__(self, eps=1e-7,weight_factor=1,Y_factor=5,UV_factor=1):
        super(color_space_loss, self).__init__()
#         self.threshold_uv = threshold_uv
        self.loss = nn.L1Loss(reduction='none')
        self.eps = eps
        self.weight_factor = weight_factor
        self.Y_factor = Y_factor
        self.UV_factor = UV_factor

        ## take the Y loss into consideration priorly 
        #  and when it drops to threshold_uv and then add the uv loss

    def forward(self, predict,label):
#         weight = torch.exp((torch.abs(predict)*torch.abs(label)-predict*label)/((predict-label)**2+self.eps))
        weight = torch.sign(-predict*label)
        weight[weight<1] =0
        loss_all = torch.exp(weight)*self.loss(predict,label)
        loss_y = torch.mean(loss_all[:,0,:,:])
        loss_u = torch.mean(loss_all[:,1,:,:])
        loss_v = torch.mean(loss_all[:,2,:,:])
        weight_loss = torch.mean(weight)
        total_loss = torch.mean(self.Y_factor*loss_y+self.UV_factor*loss_u+self.UV_factor*loss_v) + self.weight_factor*weight_loss
        return total_loss