import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np

import cart_warping1_functions as CWF

'''
2020.11.14
This script follows <cart_warping1_functions.py>. 
'''

def make_xy2approxSynth_grid_xySep(fixs_xy, output_size, m, n, b=1, device='cuda'):
    '''
    2020.11.14
    Current code is changed, therefore it is now different from the ipython notebook.
        1. Calculate a inside the function. 
        2. Function does not get parameter a. 
        3. Function does not get parameter xyp_max. 
    2020.11.11
    Use approx synthetic function combined with both exp and linear.
    Mapping fixations from xy to xyp is changed. 
        See 20201113_mk, offline note 2020.11.11
    2020.11.09
    Make the transform function applied to x and y separatedly. 
    Previously, it was applied to r. 
    Now, it will not use r
        (x,y)--> forward function --> (x'=f(x),y'=f(y))
    2020.10.31
    Make logpolar grids to transform xy image to logpolar image.
    Previously, the logpolar transformation started by first building a regular grid
    in (logr, theta) space and then by inverse transformation, corresponding xy coords
    are calculated. However, in this new scheme, regular grid is first built in x'y' 
    space. The trnaformation flow is as follows:
        (x,y)-->(r(x,y),theta(x,y)) -forward function-> (e=log(r),theta) --> (x'(e,t),y'(e,t))
    See the offline note 2020,10,31 for details.
    Args:
        fixs: (b, 2), (float x, float y), tensor fixations
        output_size: (1, 2), (int h, int w), grid resolution 
        xyp_max: constant for controlling a range of the regular grid. It is directly
                related to the shape of transformed grid in xy space. 
    Return:
        grid_xy: (b, h, w, 2), tensor
    '''
    batch_s = fixs_xy.size(0)
    xyp_max = n/m
    a = 1/np.sinh(n/m*b) 

    ## Consider Fixation
    # convert fixation from xy to xyp
    fixs_xy_length = 1 - torch.abs(fixs_xy) 
    fixs_xp = torch.sign(fixs_xy[:,0]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,0], a, b, m, n))
    fixs_yp = torch.sign(fixs_xy[:,1]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,1], a, b, m, n))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), 1)

    # regular grid in x'y'
    grid_lp_xyp_reg = CWF.make_regular_grid(res_xy=output_size[::-1]) # (1, h, w, 2)

    # scale grid
    grid_lp_xyp_reg = grid_lp_xyp_reg * xyp_max
    grid_lp_xyp_reg = grid_lp_xyp_reg.repeat(batch_s, 1, 1, 1)

    # center grid to fixation point 
    grid_lp_xyp_reg = grid_lp_xyp_reg - fixs_xyp.unsqueeze(1).unsqueeze(1)

    grid_x = CWF.convert_approxsynt2xy(grid_lp_xyp_reg.view(-1, 2)[:,0], a, b, m, n)
    grid_y = CWF.convert_approxsynt2xy(grid_lp_xyp_reg.view(-1, 2)[:,1], a, b, m, n)
    grid_xy = torch.cat((grid_x.unsqueeze(1), grid_y.unsqueeze(1)), 1)

    # restore xy grid fixation
    grid_xy = grid_xy + fixs_xy.unsqueeze(1).unsqueeze(1)

    return grid_xy.view(grid_lp_xyp_reg.size())

def make_xy2approxSynth_grid_r(fixs_xy, output_size, m, n, b=1, isTensorArgs=False, device='cuda'):
    '''
    2020.11.14
    Current code is changed, therefore it is now different from the ipython notebook.
        1. Calculate a inside the function. 
        2. Function does not get parameter a. 
        3. Function does not get parameter xyp_max. 
    2020.11.11
    This version is for computing in r space, not (x,y) seperatedly. 
    It is again 
        (x,y)-->(r(x,y),theta(x,y)) -forward function-> (e=log(r),theta) --> (x'(e,t),y'(e,t))
    2020.11.11
    Use approx synthetic function combined with both exp and linear.
    Mapping fixations from xy to xyp is changed.
            See 20201113_mk, offline note 2020.11.11
    2020.11.09
    Make the transform function applied to x and y separatedly. 
    Previously, it was applied to r. 
    Now, it will not use r
        (x,y)--> forward function --> (x'=f(x),y'=f(y))
    2020.10.31
    Make logpolar grids to transform xy image to logpolar image.
    Previously, the logpolar transformation started by first building a regular grid
    in (logr, theta) space and then by inverse transformation, corresponding xy coords
    are calculated. However, in this new scheme, regular grid is first built in x'y' 
    space. The trnaformation flow is as follows:
        (x,y)-->(r(x,y),theta(x,y)) -forward function-> (e=log(r),theta) --> (x'(e,t),y'(e,t))
    See the offline note 2020,10,31 for details.
    Args:
        fixs: (b, 2), (float x, float y), tensor fixations
        output_size: (1, 2), (int h, int w), grid resolution 
        xyp_max: constant for controlling a range of the regular grid. It is directly
        related to the shape of transformed grid in xy space. 
    Return:
        grid_xy: (b, h, w, 2), tensor
    '''
    batch_s = fixs_xy.size(0)
    xyp_max = n/m
    if isTensorArgs:
        a = 1/torch.sinh(n/m*b)
    else:
        a = 1/np.sinh(n/m*b) 

    ## Consider Fixation
    # convert fixation from xy to xyp
    fixs_xy_length = 1 - torch.abs(fixs_xy) 
    fixs_xp = torch.sign(fixs_xy[:,0]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,0], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_yp = torch.sign(fixs_xy[:,1]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,1], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), 1)

    # regular grid in x'y'
    grid_lp_xyp_reg = CWF.make_regular_grid(res_xy=output_size[::-1]) # (1, h, w, 2)

    # scale grid
    grid_lp_xyp_reg = grid_lp_xyp_reg * xyp_max
    grid_lp_xyp_reg = grid_lp_xyp_reg.repeat(batch_s, 1, 1, 1)

    # center grid to fixation point 
    grid_lp_xyp_reg = grid_lp_xyp_reg - fixs_xyp.unsqueeze(1).unsqueeze(1)

    # convert x'y' to rtp
    grid_rtp = CWF.convert_xy2rt(grid_lp_xyp_reg.view(-1, 2))  # (bhw, 2)
    # convert approx_rt to rt
    grid_r = CWF.convert_approxsynt2xy(grid_rtp.view(-1, 2)[:, 0], a, b, m, n, isTensorArgs=isTensorArgs)
    grid_rt = torch.cat((grid_r.unsqueeze(1), grid_rtp.view(-1, 2)[:, 1].unsqueeze(1)), 1)
    # convert rt to xy
    grid_xy = CWF.convert_rt2xy(grid_rt).view(batch_s, *output_size, 2)  # (b, h, w, 2)

    # restore xy grid fixation
    grid_xy = grid_xy + fixs_xy.unsqueeze(1).unsqueeze(1)

    return grid_xy.view(grid_lp_xyp_reg.size())

def make_xy2approxSynth_grid_r_coods(fixs_xy, coords, output_size, m, n, b=1, isTensorArgs=False, device='cuda'):
    '''
    20201217
    This code is just copied from right above. 
    This code is used to transform given coords. 
    Original code used regular grid and transformed it but now, it receives coords and transform. 

    Args:
        coods: (b, h, w, 2)
    '''
    batch_s = fixs_xy.size(0)
    xyp_max = n/m
    if isTensorArgs:
        a = 1/torch.sinh(n/m*b)
    else:
        a = 1/np.sinh(n/m*b) 

    ## Consider Fixation
    # convert fixation from xy to xyp
    fixs_xy_length = 1 - torch.abs(fixs_xy) 
    fixs_xp = torch.sign(fixs_xy[:,0]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,0], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_yp = torch.sign(fixs_xy[:,1]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,1], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), 1)

    # regular grid in x'y'
    #grid_lp_xyp_reg = CWF.make_regular_grid(res_xy=output_size[::-1]) # (1, h, w, 2)
    grid_lp_xyp_reg = coords # (1, h, w, 2)

    # scale grid
    grid_lp_xyp_reg = grid_lp_xyp_reg * xyp_max
    #grid_lp_xyp_reg = grid_lp_xyp_reg.repeat(batch_s, 1, 1, 1)

    # center grid to fixation point 
    grid_lp_xyp_reg = grid_lp_xyp_reg - fixs_xyp.unsqueeze(1).unsqueeze(1)

    # convert x'y' to rtp
    grid_rtp = CWF.convert_xy2rt(grid_lp_xyp_reg.view(-1, 2))  # (bhw, 2)
    # convert approx_rt to rt
    grid_r = CWF.convert_approxsynt2xy(grid_rtp.view(-1, 2)[:, 0], a, b, m, n, isTensorArgs=isTensorArgs)
    grid_rt = torch.cat((grid_r.unsqueeze(1), grid_rtp.view(-1, 2)[:, 1].unsqueeze(1)), 1)
    # convert rt to xy
    grid_xy = CWF.convert_rt2xy(grid_rt).view(batch_s, *output_size, 2)  # (b, h, w, 2)

    # restore xy grid fixation
    grid_xy = grid_xy + fixs_xy.unsqueeze(1).unsqueeze(1)

    return grid_xy.view(grid_lp_xyp_reg.size())

def make_approxSynth2xy_grid_r(fixs_xy, output_size, m, n, b=1, isTensorArgs=False, device='cuda'):
    '''
    2020.11.23
    Generates grid for converting image from xyp to xy. It is an inverse function 
    of make_xy2approxSynth2_grid_r.
    This version is for computing in r space, not (x,y) seperatedly. 
    It is  
        (x,y)-->(r(x,y),theta(x,y)) -inverse function-> (e=log(r),theta) --> (x'(e,t),y'(e,t))

    Make a grid to transform xyp image to xy image.
    See the offline note 2020.11.23 for details.
    Tested from logpolar_in_cart1-with_inverse.ipynb

    Args:
        fixs_xy: (b, 2), (float x, float y), tensor fixations
        output_size: (1, 2), (int h, int w), output grid resolution in xyp
        xyp_max: constant for controlling a range of the regular grid. It is directly
            related to the shape of transformed grid in xy space. 
            ---> replaced to compute inside the function
    Return:
        grid_xyp: (b, h, w, 2), tensor
    '''
    batch_s = fixs_xy.size(0)
    xyp_max = n/m
    if isTensorArgs:
        a = 1/torch.sinh(n/m*b)
    else:
        a = 1/np.sinh(n/m*b)

    ## Consider Fixation
    # convert fixation from xy to xyp
    fixs_xy_length = 1 - torch.abs(fixs_xy) 
    fixs_xp = torch.sign(fixs_xy[:,0]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,0], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_yp = torch.sign(fixs_xy[:,1]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,1], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), 1)

    # regular grid in xy
    grid_lp_xy_reg = CWF.make_regular_grid(res_xy=output_size[::-1]) # (1, h, w, 2)

    # scale grid
    grid_lp_xy_reg = grid_lp_xy_reg.repeat(batch_s, 1, 1, 1) # (b, h, w, 2)

    # center grid to fixation point 
    grid_lp_xy_reg = grid_lp_xy_reg - fixs_xy.unsqueeze(1).unsqueeze(1)

    # convert xy to rt
    grid_rt = CWF.convert_xy2rt(grid_lp_xy_reg.view(-1, 2))  # (bhw, 2)
    # convert rt to approx_rtp
    grid_rp = CWF.convert_xy2approxsynt(grid_rt.view(-1, 2)[:, 0], a, b, m, n, isTensorArgs=isTensorArgs)
    grid_rtp = torch.cat((grid_rp.unsqueeze(1), grid_rt.view(-1, 2)[:, 1].unsqueeze(1)), 1)
    # convert rt to xy
    grid_xyp = CWF.convert_rt2xy(grid_rtp).view(batch_s, *output_size, 2)  # (b, h, w, 2)

    # restore xy grid fixation
    grid_xyp = grid_xyp + fixs_xyp.unsqueeze(1).unsqueeze(1)

    # re-scale 
    # This is required to apply the resulting grid to image space ranging from -1 
    # to 1. It is needed because the sampling function needs it not because mathematical
    # regors. 
    grid_xyp = grid_xyp / xyp_max 

    return grid_xyp.view(grid_lp_xy_reg.size())



def convert_coords_approxSynth2xy_r(fixs_xy, coords, m, n, b=1, isTensorArgs=False, device='cuda'):
    '''
    2022.3.2
    This function is based on the following function above. 
        def make_xy2approxSynth_grid_r(fixs_xy, output_size, m, n, b=1, isTensorArgs=False, device='cuda'):
    This function converts coordinates in x'y' to xy space. 
    Args:
        fixs_xy: (b, 2), tensor, range -1~1, in xy space.
        coords: (b, 2), tensor, range -1~1, in x'y' space.
    Return:
        coords_xy: (b, 2), tensor, range -1~1, in xy space. 
    '''
    batch_s = fixs_xy.size(0)
    xyp_max = n/m
    if isTensorArgs:
        a = 1/torch.sinh(n/m*b)
    else:
        a = 1/np.sinh(n/m*b) 

    ## Consider Fixation
    # convert fixation from xy to xyp
    fixs_xy_length = 1 - torch.abs(fixs_xy) 
    fixs_xp = torch.sign(fixs_xy[:,0]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,0], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_yp = torch.sign(fixs_xy[:,1]) * (n/m - CWF.convert_xy2approxsynt(fixs_xy_length[:,1], a, b, m, n, isTensorArgs=isTensorArgs))
    fixs_xyp = torch.cat((fixs_xp.unsqueeze(1), fixs_yp.unsqueeze(1)), 1)

    # regular grid in x'y'
    #grid_lp_xyp_reg = CWF.make_regular_grid(res_xy=output_size[::-1]) # (1, h, w, 2)
    grid_lp_xyp_reg = coords # (b, 2)

    # scale grid
    grid_lp_xyp_reg = grid_lp_xyp_reg * xyp_max
    #grid_lp_xyp_reg = grid_lp_xyp_reg.repeat(batch_s, 1, 1, 1)

    # center grid to fixation point 
    grid_lp_xyp_reg = grid_lp_xyp_reg - fixs_xyp #.unsqueeze(1).unsqueeze(1)

    # convert x'y' to rtp
    grid_rtp = CWF.convert_xy2rt(grid_lp_xyp_reg)  # (b, 2)
    # convert approx_rt to rt
    grid_r = CWF.convert_approxsynt2xy(grid_rtp.view(-1, 2)[:, 0], a, b, m, n, isTensorArgs=isTensorArgs) # (b,)
    grid_rt = torch.cat((grid_r.unsqueeze(1), grid_rtp.view(-1, 2)[:, 1].unsqueeze(1)), 1) # (b, 2)
    # convert rt to xy
    grid_xy = CWF.convert_rt2xy(grid_rt) # (b, 2)

    # restore xy grid fixation
    grid_xy = grid_xy + fixs_xy 

    return grid_xy


