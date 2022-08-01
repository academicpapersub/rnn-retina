import torch
import math
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append('/home/cminkyu/git_libs/convert_img_coords/cart/')
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/convert_img_coords/')
import Gaussian_RF as gauss


def get_gaussian_mask(attn_p, mask_prev=None, heatmap_s=[192,256], sigma=None, device='cuda'):
    '''
    Modified 2021.04.09. Sigma can be a tensor with batch. Each sample in the batch can have different sigmas. 
    Modified 2020.11.28. Can change sigma as an input args. 
    Modified 2020.07.06. Code from 'Gaussian IOR test.ipynb'
    Modified 2020.07.12. In the function 'get_gaussian_kernel', mean=(attn_p+1.0)... line is changed. 

    Args: 
        attn_p: (float x, float y), range -1~1
        mask_prev: (b, )
        heatmap_s: (int h, int w)
        sigma: tensor, (b, 1)
    Returns:
        gaussian_kernel: (b, 1, heatmap_s[0], heatmap_s[1])
    '''
    batch_s = attn_p.size()[0]
    if mask_prev is None:
        mask_prev = torch.ones((batch_s, heatmap_s[0], heatmap_s[1]), device=device)
    if sigma is None:
        sigma = torch.ones((batch_s, 1), device=device) * 25.0

    region_cur = gauss.get_gaussian_kernel(attn_p, sigma, kernel_size=heatmap_s, norm='max', device=device)
    
    mask_cur = nn.functional.relu(mask_prev - region_cur)
    return mask_cur
