import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np

'''2020.11.14.
Code in this script is from logpolar_in_cart1.ipynb. 
'''

def count_spp(grid):
    ''' 2020.11.11
    Count sample per pixel (SPP). 
    Args: 
        grid: (1, h, w, 2), tensor of grid ranged in pixel (ex: -112~112)
    Returns:
        count: (max_pixel_dist, 1)
        spp: (max_pixel_dist, 1)
    '''
    grid_lin = grid.view(-1, 2)
    grid_r = torch.sqrt(grid_lin[:,0]**2 + grid_lin[:,1]**2)

    grid_r = grid_r.numpy()
    max_pixel_dist = int(np.max(grid_r))

    count = np.zeros((max_pixel_dist+1, 1))
    for idx in range(len(grid_r)):
        count[int(grid_r[idx] // 1)] = count[int(grid_r[idx] // 1)] + 1

    spp = np.zeros((max_pixel_dist+1, 1))
    for idx in range(len(count)):
        spp[idx] = count[idx] / ((2*(idx+1))**2-(2*idx)**2)
        #print(((2*(idx+1))**2-(2*idx)**2), (2*(idx+1))**2, (2*idx)**2, idx)
    return count, spp

def filter_grid_new(grid, a):
    '''2020.11.11
    works for a single batch
    '''
    cnt = 0
    for idx in range(len(grid)):
        if torch.abs(grid[idx,0]) < a and torch.abs(grid[idx,1]) < a:
            cnt = cnt + 1

    gf = torch.zeros(cnt,2)
    idx_r = 0
    for idx in range(len(grid)):
        if torch.abs(grid[idx,0]) < a and torch.abs(grid[idx,1]) < a:
            gf[idx_r, :] = grid[idx,:]
            idx_r = idx_r + 1
    return gf
