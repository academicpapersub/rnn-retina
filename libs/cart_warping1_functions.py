import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np

'''
Initial writing: 2020.11.14
See slides 20201113_mk
Code was tested from logpolar_in_cart1.ipynb

This script includes basic functions. This script is followed by <cart_warping1_high.py>
Initially for image space warping, I used (log)polar transformation. 
But it had a issue in that features are heavily depending on the fixation locations. 
Therefore, I changed warping method to warp image in Cartesian image space. 
Also, instead of just shifting the fixed grid, I tried to dynamically make a grid
based on the given fixation location. It enabled the warped image not to lose any 
image region outside of pre-defined grid region. But initial version had some issue
for transforming fixations thus, losing some image region near fixations. This issue
is illustrated in <20201106_mk>. 
The current code solved all issues posed previously and will work well. 
'''

def make_regular_grid(range_x=(-1, 1), range_y=(-1, 1), res_xy=(64,64), device='cuda'):
    '''2020.10.31
    Make a regular grid
    Return:
        grid_reg: (1, res_xy[0], res_xy[1], 2)
    '''
    xxrange = torch.linspace(range_x[0], range_x[1], res_xy[0], device=device)
    yyrange = torch.linspace(range_y[0], range_y[1], res_xy[1], device=device)
    ys, xs = torch.meshgrid([yyrange, xxrange])
    grid_reg = torch.stack([xs, ys], 2).unsqueeze(0)
    return grid_reg

def convert_xy2rt(xy):
    '''2020.10.31
    Convert xy coords to r,theta coords. 
    Args:
        xy: (b, 2), tensor, xy coordinates 
    Return:
        rt: (b, 2), tensor, rt coordinates
    '''
    rs = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
    ts = torch.atan2(xy[:, 1], xy[:, 0])
    rt = torch.stack([rs, ts], 1)
    return rt

def convert_lpt2rt(lpt, a=1, b=1):
    '''2020.10.31
    Convert lpt(log(r), t) to (r, t)
    Args:
        lpt: (b, 2), tensor, grid
        a, b: int, parameter
    returns:
        rt: (b, 2): tensor grid
    '''
    logr = lpt[:, 0]
    r = (torch.exp(logr)*b - b)/a
    theta = lpt[:, 1]
    rt = torch.stack([r, theta], 1)
    return rt

def convert_rt2lpt(rt, a=1, b=1):
    '''2020.10.31
    Convert rt(r, t) to (log(r), t)
    Args:
        rt: (b, 2), tensor, grid
        a, b: int, parameter
    returns:
        lpt: (b, 2): tensor grid
    '''
    r = rt[:, 0]
    logr = torch.log((a*r+b)/b)
    theta = rt[:, 1]
    lpt = torch.stack([logr, theta], 1)
    return lpt

def convert_approx2rt(apprt, a, b):
    '''2020.10.31
    Changed in 2020.11.02
    a: controls how long the early rs stick to r=r'
    b: controls how fast later rs increase compared to r=r'
    Convert approxr to (r, t)
    Args:
        apprt: (b, 2), tensor, grid
        a, b: float, parameter
    returns:
        rt: (b, 2): tensor grid
    '''
    appr = apprt[:, 0]
    r = a * torch.sinh(b*appr)
    theta = apprt[:, 1]
    rt = torch.stack([r, theta], 1)
    return rt

def convert_rt2approx(rt, a, b):
    '''2020.10.31
    Changed in 2020.11.02
    a: controls how long the early rs stick to r=r'
    b: controls how fast later rs increase compared to r=r'
    Convert (r, t) to approxr 
    Args:
        rt: (b, 2), tensor, grid
        a, b: float, parameter
    returns:
        approx: (b, 2): tensor grid
    '''
    r = rt[:, 0]
    approxr = arcsinh_torch(r/a)/b
    theta = rt[:, 1]
    approx = torch.stack([approxr, theta], 1)
    return approx

def convert_approx2xy(apprxy, a, b):
    '''2020.11.09
    This function applies transformation not to r but to x and y separatedly. 
    a: controls how long the early rs stick to x=x'
    b: controls how fast later rs increase compared to x=x'
    Convert approx x or approx y to x or y
    It is smilar to the <convert_appros2rt> function. 
    Args:
        apprxy: (b, 1), tensor, grid including coords of either x or y in approx.
        a, b: float, parameter
    returns:
        xy: (b, 1): tensor grid, transformed coords corresponding apprxy
    '''
    xy = a * torch.sinh(b*apprxy)
    return xy

def convert_xy2approx(xy, a, b):
    '''2020.11.09
    This function applies transformation not to r but to x and y separatedly. 
    a: controls how long the early rs stick to x=x'
    b: controls how fast later rs increase compared to x=x'
    Convert x or y to approx x or approx y
    Args:
    xy: (b, 1), tensor, grid including coords of either x or y
    a, b: float, parameter
    returns:
    apprxy: (b, 1): tensor grid, transformed coords corresponding apprxy
    '''
    apprxy = arcsinh_torch(xy/a)/b
    return apprxy

def convert_approxsynt2xy(apprxy, a, b, m, n, p_cover=1.0, isTensorArgs=False):
    '''2020.11.11
    This function applies transformation not to r but to x and y separatedly. 
    Although it says that it transforms to xy, basic function of this function
    is to transform (b,1) sized tensor, not (b,2). Thereofer, this function can
    be used to transform r. 
    This is synthetic function combined with exp function for early domain and
    linear mapping for the later domain. 
    a: controls how long the early rs stick to r=r'
    b: controls how fast later rs increase compared to r=r'
    Convert appr x or appr y to x or y
    Constraints:
        1. f(n/m) = 1
        2. for small xp<n/m, f(xp) = xp
        3. after xp>n/m, f(xp) becomes linear mapping
    Args:
        apprxy: (b, 1), tensor, grid including coords of either x or y
        a, b: float, parameter
        m, n: int, resolution of full image and sampled image (ex: m=224, n=64)
        p_cover: float, for the first constraints, f(n/m*p_cover) = 1
    returns:
        xy: (b, 1): tensor grid, transformed coords corresponding apprxy
    '''
    if isTensorArgs:
        early = a * torch.sinh(b*apprxy)
        alp = a*b*torch.cosh(b*n/m)
    else:
        early = a * torch.sinh(b*apprxy)
        alp = a*b*np.cosh(b*n/m)
    later = alp * (apprxy - n/m) + p_cover

    tf = apprxy < n/m
    xy = early * tf + later * (~tf)

    return xy


def convert_xy2approxsynt(xy, a, b, m, n, p_cover=1.0, isTensorArgs=False):
    '''2020.11.11
    This function applies transformation not to r but to x and y separatedly. 
    This is synthetic function combined with exp function for early domain and
    linear mapping for the later domain. 
    a: controls how long the early rs stick to r=r'
    b: controls how fast later rs increase compared to r=r'
    Convert x or y to approx xyp
    Constraints:
        1. f(n/m) = 1
        2. for small xp<n/m, f(xp) = xp
        3. after xp>n/m, f(xp) becomes linear mapping
    Args:
        xy: (b, 1), tensor, grid including coords of either x or y
        a, b: float, parameter
        m, n: int, resolution of full image and sampled image
        p_cover: float, for the first constraints, f(n/m*p_cover) = 1
    returns:
        xy: (b, 1): tensor grid, transformed coords corresponding apprxy
    '''
    if isTensorArgs:
        bound = a * torch.sinh(b*n/m)
        alp = a*b*torch.cosh(b*n/m)
    else:
        bound = a * np.sinh(b*n/m)
        alp = a*b*np.cosh(b*n/m)

    # arcsinh_torch(xy/a)/b
    early = arcsinh_torch(xy/a)/b
    later = 1/alp * (xy - p_cover) + n/m

    tf = xy < bound
    xy = early * tf + later * (~tf)

    return xy

def arcsinh_torch(x):
    return torch.log(x+torch.sqrt(1+x**2))

def arcsinh(x):
    return np.log(x+np.sqrt(1+x**2))


def convert_rt2xy(rt):
    '''2020.10.31
    Convert rt to xy
    Args:
        rt: (b, 2), tensor, grid
    Returns:
        xy: (b, h2), tensor, grid
    '''
    rs = rt[:, 0]
    ts = rt[:, 1]
    xs = rs * torch.cos(ts)
    ys = rs * torch.sin(ts)
    xy = torch.stack([xs, ys], 1)
    return xy


