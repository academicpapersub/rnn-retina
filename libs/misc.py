import torch                                                        
from torch.utils.data import DataLoader                             
from torchvision import transforms                                  
import torch.optim as optim                                         
import torch.nn as nn                                               
import torch.backends.cudnn as cudnn                                
import torchvision.datasets as datasets  

import os
import argparse
import numpy as np
import random

import matplotlib.pyplot as plt
import cv2
plt.switch_backend('agg') 


from pathlib import Path
def make_dir(path):
    Path(os.path.expanduser(path)).mkdir(parents=False, exist_ok=True)

def isNan(tensor):
    '''get a tensor and return True if it includes Nan'''
    return (tensor!=tensor).any()

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

def initialize_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def plot_samples_from_images_fixedRange(images, batch_size, plot_path, filename, isRange01=False):
    ''' Plot images
    Changed 2020.11.23
        isRange01 is added to normalize image in different way. 

    Args: 
        images: (b, c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    #print(torch.max(images), torch.min(images))
    if isRange01:
        max_pix = torch.max(torch.abs(images))
        images = images/max_pix
        #print(max_pix, torch.min(torch.abs(images)))
        #print(torch.max(images))
    else:
        #print(max_pix)
        images = ((images*0.2) + 0.5)
        
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)

    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, filename))
    else:
        plt.show()
    plt.close()
    pass


def plot_samples_from_images(images, batch_size, plot_path, filename, isRange01=False):
    ''' Plot images
    Changed 2020.11.23
        isRange01 is added to normalize image in different way. 

    Args: 
        images: (b, c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    #print(torch.max(images), torch.min(images))
    if isRange01:
        max_pix = torch.max(torch.abs(images))
        images = images/max_pix
        #print(max_pix, torch.min(torch.abs(images)))
        #print(torch.max(images))
    else:
        max_pix = torch.max(torch.abs(images))
        #print(max_pix)
        if max_pix != 0.0:
            images = ((images/max_pix) + 1.0)/2.0
        else:
            images = (images + 1.0) / 2.0
            #print('inside')
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)

    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, filename))
    else:
        plt.show()
    plt.close()
    pass


def plot_one_sample_from_images(images, plot_path, filename, isRange01=False):
    ''' Plot images
    Changed 2020.11.23
        isRange01 is added to normalize image in different way. 

    Args: 
        images: (b, c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    #print(torch.max(images), torch.min(images))
    if isRange01:
        '''max_pix = torch.max(torch.abs(images))
        images = images/max_pix'''
        images = images
        #print(max_pix, torch.min(torch.abs(images)))
        #print(torch.max(images))
    else:
        max_pix = torch.max(torch.abs(images))
        #print(max_pix)
        if max_pix != 0.0:
            images = ((images/max_pix) + 1.0)/2.0
        else:
            images = (images + 1.0) / 2.0
            #print('inside')
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(torch.squeeze(images).cpu().numpy(), 0, 1), 1, 2)
    #print(np.shape(images))

    #print(filename)
    #print(os.path.join(plot_path, filename))
    idx=0
    plt.imsave(os.path.join(plot_path, filename), images)
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_classification_TF(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        #print(pred.size(), pred.t().size())
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print(correct.size())
        return correct

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        ''' Multiplying n to val before summation is done because
        it is usually used for loss which is already mean with respect to batch size. 
        '''
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_PE_cart(batch_s, pe_s):
	'''
	Generates cartesian Positional Encoding. 
	Args: 
		batch_s: int, batch size. 
		pe_s: (int h, int w), size of Positional encoding feature 
	Return:
		pe: (batch_s, 2, pe_s[0], pe_s[1])
	'''
	lin_x = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=pe_s[1], device='cuda'), 0).repeat(pe_s[0], 1) # (192, 256)
	lin_y = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=pe_s[0], device='cuda'), 0).repeat(pe_s[1], 1) # (, 256)
	lin_y = lin_y.t()
	
	lin_x = torch.unsqueeze(lin_x, 0).repeat(batch_s, 1, 1) # (batch, fm_s, fm_s)
	lin_y = torch.unsqueeze(lin_y, 0).repeat(batch_s, 1, 1)
	# (b, h, w)

	pe = torch.cat((lin_x.unsqueeze(1), lin_y.unsqueeze(1)), 1)# (b, 2, h, w)
	return pe


def get_coord_feature(batch_s, input_s, pe_s, polar_grid=None):
    '''
    20201124
    Changed to have an option to return Cart PE. 

    code from (base) min@libigpu1:~/research_mk/attention_model_biliniear_localGlobal_6_reinforce/detach_recurrency_group/rl_base_recon_corHM_absHM_corrREINFORCE_corInitRNN_detachTD_hlr_lowResD_detachVD_removeBottleNeckV2I_conventionalResInc_contVer110

    Generate Positional encoding
    PE feature will first be generated in cartesian space with same size of input. 
    And then it will be transformed to polar space, if polar_grid not nont. 
    After that, it will be resized to pe_s. 

    Args:
        batch_s: int. Batch size
        input_s: (int h, int w), Size of loaded input image in cartesian space. 
        pe_s: (int h', int w'), Size of positional encoding feature in polar space. This size of feature will be returned. 
        polar_grid: Grid for polar transformation. It must be the same grid used in polar CNN. Fixation points must be already added to this grid.
    return:
        polar_pe_resized: (b, 2, h', w')
    '''

    with torch.no_grad():
        lin_x = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=input_s[1], device='cuda'), 0).repeat(input_s[0], 1) # (192, 256)
        lin_y = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=input_s[0], device='cuda'), 0).repeat(input_s[1], 1) # (, 256)
        lin_y = lin_y.t()
        #lin_r = torch.sqrt(lin_x**2 + lin_y**2)
        
        lin_x = torch.unsqueeze(lin_x, 0).repeat(batch_s, 1, 1) # (batch, fm_s, fm_s)
        lin_y = torch.unsqueeze(lin_y, 0).repeat(batch_s, 1, 1)
        #lin_r = torch.unsqueeze(lin_r, 0).repeat(batch_s, 1, 1) # (batch, fm_s, fm_s)
        # (b, h, w)

        cart_pe = torch.cat((lin_x.unsqueeze(1), lin_y.unsqueeze(1)), 1)# (b, 2, h, w)

        if polar_grid is not None:
            pe = torch.nn.functional.grid_sample(cart_pe, polar_grid, align_corners=False) # (b, 2, h', w')
        else:
            pe = cart_pe
        
        pe_resized =  torch.nn.functional.interpolate(pe, pe_s)
    
    return pe_resized

def set_w_requires_no_grad(model, tf=False):
    for param in model.parameters():
        param.requires_grad = tf


def get_theta(attn, scale_attn, batch_s):
    '''
    code from libigpu3:~/research_mk/imagenet_new/sumReluHeatmap_fullD_oneshot/train.py
    attn (tensor): (batchx2), (-1~1)
    '''
    a1x = torch.ones(batch_s, 1, 1, requires_grad=False, device='cuda')*scale_attn#(self.v_img_s[2]/self.img_size_o) # (batch, 1, 1)
    a1y = torch.zeros(batch_s, 1, 1, requires_grad=False, device='cuda')
    a2x = torch.zeros(batch_s, 1, 1, requires_grad=False, device='cuda')
    a2y = torch.ones(batch_s, 1, 1, requires_grad=False, device='cuda')*scale_attn#(self.v_img_s[1]/self.img_size_o)

    a1 = torch.cat((a1x, a1y), 1)#.cuda()#, requires_grad=False) # (b, 2, 1)
    a2 = torch.cat((a2x, a2y), 1)#.cuda()#, requires_grad=False)
    theta = torch.cat((a1, a2, torch.unsqueeze(attn.cuda(), 2)), 2) # (b, 2, 3)
    return theta#.cuda()


def get_glimpses_new(images, fixs_xy, patch_s, crop_ratio=[0.25, 0.5, 1.0]):
    '''
    code from libigpu3:~/research_mk/imagenet_new/sumReluHeatmap_fullD_oneshot/train.py
    fixs_xy: tensor, (b, 2), -1~1

    2021/6/22: After Neurips submission, this code is changed to use mode=bilinear, instead of nearest. 
        It was nearest but nearest seems to obfuscate gradients. 
    '''
    mode = 'bilinear'
    batch_s = images.size(0)
    theta_highres = get_theta(fixs_xy, crop_ratio[0], batch_s)
    grid_highres = torch.nn.functional.affine_grid(theta_highres, torch.Size((batch_s, 3, patch_s, patch_s)))#, device='cuda')
    img_highres = torch.nn.functional.grid_sample(images, grid_highres, mode='bilinear')

    theta_midres = get_theta(fixs_xy, crop_ratio[1], batch_s)
    grid_midres = torch.nn.functional.affine_grid(theta_midres, torch.Size((batch_s, 3, patch_s, patch_s)))#, device='cuda')
    img_midres = torch.nn.functional.grid_sample(images, grid_midres, mode='bilinear')

    theta_lowres = get_theta(fixs_xy, crop_ratio[2], batch_s)
    grid_lowres = torch.nn.functional.affine_grid(theta_lowres, torch.Size((batch_s, 3, patch_s, patch_s)))#, device='cuda')
    img_lowres = torch.nn.functional.grid_sample(images, grid_lowres, mode='bilinear')

    return img_highres, img_midres, img_lowres


def spatial_basis(height=28, width=28, channels=64):
    '''
    code from 
        https://github.com/cjlovering/interpretable-reinforcement-learning-using-attention/blob/master/torchbeast/attention_net.py
    '''
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    h, w, d = height, width, channels

    p_h = torch.mul(
            torch.arange(1, h + 1).unsqueeze(1).float(), torch.ones(1, w).float()
            ) * (np.pi / h)
    p_w = torch.mul(
            torch.ones(h, 1).float(), torch.arange(1, w + 1).unsqueeze(0).float()
            ) * (np.pi / w)

    U = V = 8  # size of U, V.
    u_basis = v_basis = torch.arange(1, U + 1).unsqueeze(0).float()
    a = torch.mul(p_h.unsqueeze(2), u_basis)
    b = torch.mul(p_w.unsqueeze(2), v_basis)
    out = torch.einsum("hwu,hwv->hwuv", torch.cos(a), torch.cos(b)).reshape(h, w, d)

    return out.permute(2, 0, 1)

