import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import numpy as np
import time
import torch.nn.functional as F
from torch.nn import init
import scipy.io
import os
import argparse

import vgg_backbone as retina
import sys
sys.path.append('/home/cminkyu/git_libs/misc/')
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/')
import misc
import wrap_functions as WF
import Gaussian2d_mask_generator_v1_tensorSigma_batchSigma as G
sys.path.append('/home/cminkyu/git_libs/convert_img_coords/cart/')
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/convert_img_coords/cart/')
import cart_warping1_high as CW
sys.path.append('/home/cminkyu/git_libs/foveated_image/')
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/foveated_image/')
import foveated_image_correctIssueGPU as FI
#sys.path.append('/home/cminkyu/git_libs/ConvRNN/')
#sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/ConvRNN/')
#import ConvGRU as ConvGRU
#import ConvGRU_mini_wState as ConvGRU
sys.path.append('/home/cminkyu/git_libs/misc/imagenet/imagenet_my100/')
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/')
sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/imagenet/imagenet_my100/')
import dataloader_imagenet100my as dl

import fixations_gen


class CRNN_Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, args.n_classes)
        self.gru = nn.GRU(input_size=512, hidden_size=512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = misc.Flatten()
        self.tanh = nn.Tanh()
        
        self.agent_net = fixations_gen.AgentNet(args)
        self.retina_net = retina.CNNModel(args.n_classes)
        self.GF = FI.Get_foveated_images([1,  3,  5], [40,  70,  90], kss=7, device='cuda')

        self.init_hidden = nn.Parameter(torch.randn((1, 1, 512)), requires_grad=True)

    def heatmap_generator(self, pred, feature, fc_weight, top_N_map=1):
        ### 1. get top-N class index (self.top_N_map)
        # pred: (batch, #class)
        # feature: (batch, #inputFM, w, h)
        sorted_logit, sorted_index = torch.sort(torch.squeeze(pred), dim=1,  descending=True)
        # sorted_index: (batch, #top_N_map)

        selected_weight = torch.cat([torch.index_select(fc_weight, 0, idx).unsqueeze(0)
            for idx in  sorted_index[:, 0:top_N_map]])
        # weight: (#class, #inputFM)
        # selected_weight: (batch, top_N_map, #inputFM)

        s = feature.size()
        cams = torch.abs(torch.squeeze(torch.bmm(selected_weight, feature.view(s[0], s[1], s[2]*s[3]))))
        cams_prob = cams.view(s[0], top_N_map, s[2], s[3]) #* sorted_prob.unsqueeze(2).unsqueeze(2)
        heatmap = torch.sum(cams_prob, 1)
        return heatmap, sorted_index

    def forward(self, args, img, isTrain=True):
        batch_s = img.size(0)
        return_dict = {}
        return_dict['fixs_xy'] = []
        return_dict['ior'] = []
        return_dict['img_fov'] = []
        return_dict['img_warp'] = []
        return_dict['pred'] = []
        return_dict['hm'] = []
        return_dict['log_pi'] = []
        return_dict['actmap'] = []
        return_dict['actmap_ior'] = []
        return_dict['actmap_ior_sm'] = []


        for step in range(args.n_steps):
            if step == 0:
                '''if isTrain:
                    fixs_x = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * 0.9
                    fixs_y = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * 0.9
                else:
                    fixs_x = torch.zeros((batch_s,), dtype=torch.float32, device='cuda')#*2.0 # -0.5 ~ 0.5 range 
                    fixs_y = torch.zeros((batch_s,), dtype=torch.float32, device='cuda')#*2.0 # -0.5 ~ 0.5 range 
                '''
                ior = torch.ones((batch_s, 1, 224, 224), dtype=torch.float32, device='cuda')
                rnn_state = self.tanh(self.init_hidden.repeat(1, batch_s, 1))
                actmap = None
                ior_sigmas = None
                attn_log_pi = torch.ones(batch_s, 1, device='cuda')
                actmap = torch.ones((batch_s, 1, 18, 18), dtype=torch.float32, device='cuda')
                actmap_ior_sm = actmap_ior = actmap


            '''
            ###########################################
            ##### All Random Fixations for train ######
            if args.stage == 1:
                fixs_x = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * 0.9
                fixs_y = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * 0.9
            ###########################################
            ##### Fixation from FastSal #####
            else:
                fixs_x, fixs_y, actmap, actmap_ior, actmap_ior_sm, attn_log_pi, ior_sigmas = self.agent_net(img, actmap, ior, step)
                fixs_x, fixs_y = fixs_x.detach(), fixs_y.detach()
            ###########################################
            #print(torch.min(actmap), torch.max(actmap))
            '''
            fixs_x, fixs_y, actmap, actmap_ior, actmap_ior_sm, attn_log_pi, ior_sigmas = self.agent_net(img, actmap, ior, step, args.stage)
            fixs_x, fixs_y = fixs_x.detach(), fixs_y.detach()


            fixs_xy = torch.cat((fixs_x.unsqueeze(1), fixs_y.unsqueeze(1)), 1)
            return_dict['fixs_xy'].append(fixs_xy)
            ior =  G.get_gaussian_mask(fixs_xy, mask_prev=ior, heatmap_s=(224, 224), sigma=ior_sigmas) 
            
            #print(ior.size())

            ## for visualization purpose
            ior_ones = torch.ones((batch_s, 1, 224, 224), dtype=torch.float32, device='cuda')
            ior_ones_temp =  1 - G.get_gaussian_mask(fixs_xy, mask_prev=ior_ones, heatmap_s=[224, 224])
            ior_curr_only =  torch.nn.functional.interpolate(ior_ones_temp, 
                    (18, 18), mode='bilinear')
                
            ## Warp Cart images
            #with torch.no_grad():
            #grid_forward = CW.make_xy2approxSynth_grid_r(fixs_xy, (args.grid_size,args.grid_size), 
            #        args.grid_size*4, args.grid_size, b=args.b)
            grid_forward = CW.make_xy2approxSynth_grid_r(fixs_xy, (args.grid_size,args.grid_size),  
                    args.grid_size*4, args.grid_size, b=args.b)
            img_fov = self.GF(img, fixs_xy)
            img_warp = torch.nn.functional.grid_sample(img_fov, grid_forward, align_corners=True, mode='bilinear')
            #img_warp = torch.nn.functional.interpolate(img_warp_l, (args.grid_size,args.grid_size), mode='bilinear')

            rnn_input = torch.squeeze(self.gap(self.retina_net(img_warp)))
            self.gru.flatten_parameters()
            rnn_out, _ = self.gru(rnn_input.unsqueeze(0), rnn_state)
            pred = self.fc(torch.squeeze(rnn_out))

            rnn_state = rnn_out

            hm = torch.squeeze(actmap) # self.heatmap_generator(pred, fms_cls, self.fc.weight, top_N_map=1)

            # loss
            return_dict['log_pi'].append(attn_log_pi)

            return_dict['ior'].append(ior)
            return_dict['img_fov'].append(img_fov)
            return_dict['img_warp'].append(img_warp)
            return_dict['pred'].append(pred)
            return_dict['hm'].append(hm)
            return_dict['actmap'].append(misc.noralize_min_max(actmap))# / torch.max(actmap.view(batch_s, -1), 1)
            return_dict['actmap_ior'].append(misc.noralize_min_max(actmap_ior)) #/ torch.max(actmap_ior.view(batch_s, -1), 1)
            return_dict['actmap_ior_sm'].append(misc.noralize_min_max(actmap_ior_sm))

        return return_dict


