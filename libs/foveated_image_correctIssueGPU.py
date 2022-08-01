import sys
#import Gaussian2d_mask_generator_v1_test as G
sys.path.append('/export/home/choi574/git_libs/misc/')
import misc
import Gaussian2d_mask_generator_v1 as G

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

''' Initial writing 2020.11.14. 
Code was written logpolar_in_cart1.ipynb and moved here.
'''

class Get_foveated_images(nn.Module):
    ''' See <Scanpath estimation based on foveated image saliency>
            By Yixiu Wang, Bin Wang, Xiaofeng Wu, Liming Zhang'''
    def __init__(self, sigmas, mix_r, kss=11, device='cuda'):
        super().__init__()
        '''2020.11.11
        Make foveated images. 
        Args:
            imgs: (b, channels, h, w), tensor image
            fixs: (b, 2), (float x, float y) ranged -1~1, tensor
            kss: int, it defines the sizes of Gaussian kernels, must be odd numbers
            sigmas: (n_stages, 1), int, sigmas of each stages
        Returns:
            img_accum: (b, channels, h, w), tensor, foveated image
        '''
        self.n_stages = len(sigmas)
        self.kss = kss
        self.sigmas = sigmas
        self.mix_r = mix_r
        self.device = device
        '''self.gks = []
        for stg in range(self.n_stages):
            #print(self.n_stages, stg, sigmas[stg])
            self.gks.append(G.get_gaussian_kernel(torch.zeros((1, 2)), kernel_size=(kss,kss), sigma=sigmas[stg], channels=3, norm='sum'))
            # list of (1, 1, h, w)'''

    def forward(self, imgs, fixs):
        batch_s, channel, h, w = imgs.size()

        img_gauss = []
        img_gauss.append(imgs)
        pd = int((self.kss-1)/2)
        img_pad = F.pad(imgs, (pd,pd,pd,pd), mode='replicate')
        # img_pad: (b, 3, h, w)
        img_pad_batch = img_pad.view(3*batch_s, 1, h+2*pd, w+2*pd)
        # img_pad_batch: (3*b, 1, h, w)
        # This reshaping is required because each channel of images must be applied with 
        #   Gaussian kernels individually. Therefore, by reshaping it to move channels to batch
        #   dimension, conv2d will operate on each channel indivisually. 

        self.gks = []
        for stg in range(self.n_stages):
            #print(self.n_stages, stg, sigmas[stg])
            self.gks.append(G.get_gaussian_kernel(torch.zeros((1, 2), device='cuda'), kernel_size=(self.kss,self.kss), sigma=self.sigmas[stg], channels=3, norm='sum', device='cuda'))
            # list of (1, 1, h, w)

        ### Gaussian Blurred Images
        for stg in range(self.n_stages):
            img_g = F.conv2d(img_pad_batch, self.gks[stg])
            img_gauss.append(img_g.view(batch_s, 3, h, w))
            # list of (b, 3, h, w)
        # At this point, multi-level Gaussian kernered image is obtained. 
        # img_gauss: (n_stages+1, tensor(b, 3, h, w))

        ### Blend weights
        weights = []
        for stg in range(self.n_stages):
            weights.append(G.get_gaussian_kernel(fixs, kernel_size=(h,w), sigma=self.mix_r[stg], channels=3, norm='max', device='cuda'))
            # list of (b, 1, h, w)
        w_diff = []
        w_diff.append(weights[0])
        for stg in range(self.n_stages):
            if stg != self.n_stages-1:
                w_diff.append(weights[stg+1] - weights[stg])
            else:
                w_diff.append(1 - weights[stg])
                # list of (b, 1, h, w)
        # w_diff: (n_stages+1, tensor(b, 1, h, w))

        ### Blend Images into one
        # img_gauss: (n_stages+1, tensor(b, 3, h, w))
        # w_diff:    (n_stages+1, tensor(b, 1, h, w))
        for stg in range(self.n_stages+1):
            if stg ==0:
                img_accum = img_gauss[stg] * w_diff[stg]
            else:
                img_accum = img_accum + img_gauss[stg] * w_diff[stg]

        return img_accum#, img_gauss, w_diff
