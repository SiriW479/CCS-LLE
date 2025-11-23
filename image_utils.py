import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class rgb2yuv(object):
        def __init__(self):
            pass


        def __call__(self, imgTensor):
            size = imgTensor.shape
            # imgTensor = torch.flip(imgTensor_,[0])
            # del imgTensor_
            yuvTensor = torch.zeros(size,device=imgTensor.device)
            if imgTensor.dim()<4:
                #y
                yuvTensor[0,:,:] = 0.299*imgTensor[0,:,:]+0.587*imgTensor[1,:,:]+0.114*imgTensor[2,:,:]
                #u
                yuvTensor[1,:,:] = -0.169*imgTensor[0,:,:]-0.331*imgTensor[1,:,:]+0.500*imgTensor[2,:,:]
                #v
                yuvTensor[2,:,:] = 0.500*imgTensor[0,:,:]-0.419*imgTensor[1,:,:]-0.081*imgTensor[2,:,:]
            else:
                yuvTensor[:,0,:,:] = 0.299*imgTensor[:,0,:,:]+0.587*imgTensor[:,1,:,:]+0.114*imgTensor[:,2,:,:]
                yuvTensor[:,1,:,:] = -0.169*imgTensor[:,0,:,:]-0.331*imgTensor[:,1,:,:]+0.500*imgTensor[:,2,:,:]
                yuvTensor[:,2,:,:] = 0.500*imgTensor[:,0,:,:]-0.419*imgTensor[:,1,:,:]-0.081*imgTensor[:,2,:,:]

            # if imgTensor.is_cuda:
            #     yuvTensor =yuvTensor.cuda()
            return yuvTensor

class addGaussianNoise(object):
    def __init__(self,factor):
        assert isinstance(factor,float)
        self.factor = factor
        # self.std = (factor * k_camera**1/2 )/255


    def __call__(self, sample ):
        c,h,w = sample.shape[:]
        means = torch.zeros(c,h,w)
        stds = self.factor*(sample**1/2)
        noise = torch.normal(means,stds)
        out = (sample+noise).clamp(min = 0,max =1)
        return out

class addNoiseRange(object):
    def __init__(self,std_min,std_max):
        assert isinstance(std_min,float)
        assert isinstance(std_max,float)
        self.std_min =std_min
        self.std_max =std_max

        # self.std = (factor * k_camera**1/2 )/255


    def __call__(self, sample ):
        c,h,w = sample.shape[:]
        stds = (self.std_max-self.std_min)*torch.randn(c,h,w)+self.std_min 
        means = torch.zeros(c,h,w)
        
        noise = torch.normal(means,stds)
#         print(noise)
        out = (sample+noise).clamp(min = 0,max =1)
        return out

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[-2:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image_dim = image.dim()
        if image_dim <4:
            image.unsqueeze_(0)
        # label.unsqueeze_(0)
        img = f.interpolate(image, (new_h,new_w), mode = 'bilinear')
        if image_dim<4:
            img.squeeze_(0)
        # labels.squeeze_(0)
        # print
        return img
  
def yuv2rgb(yuvTensor):
          # y[0,1] u[-0.5 ,0.5] v[-0.5,0.5]
    size = yuvTensor.shape
    rgbTensor = torch.zeros(size,device=yuvTensor.device)
    # r
    rgbTensor[:,0,:,:] = yuvTensor[:,0,:,:] + 1.403 * yuvTensor[:,2,:,:]
    # g
    rgbTensor[:,1,:,:] = yuvTensor[:,0,:,:] - 0.344 * yuvTensor[:,1,:,:] - 0.714 * yuvTensor[:,2,:,:]
    # b
    rgbTensor[:,2,:,:] = yuvTensor[:,0,:,:] + 1.770 * yuvTensor[:,1,:,:]

    return rgbTensor



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
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid.clone())

   # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask,mask
def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map[0].detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

def colourfulness(np_2d_image: np.ndarray) -> np.float64:
    """
    Get colourfulness of the image
    Parameters
    ----------
    np_2d_image : np.ndarray
        RGB image array. Other colour space is not allowed.
    Returns
    ----------
    float
        Value of colourfulness.
    References
    ----------
    D. Hasler and S.E.Suesstrunk, ``Measuring colorfulness in natural images," Human
    Vision andElectronicImagingVIII, Proceedings of the SPIE, 5007:87-95, 2003.
    """
    rg = np.absolute(np_2d_image[:, :,0] - np_2d_image[:,:, 1])
    yb = np.absolute(0.5 * (np_2d_image[:,:, 0] + np_2d_image[:,:, 1]) - np_2d_image[:,:, 2])

    mean_rgyb = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    std_rgyb = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)

    return std_rgyb + 0.3 * mean_rgyb

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.): RGB[0,256]
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))