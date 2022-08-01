import torch
import torch.nn as nn
#import FastSal.model.fastSal as fastsal
#from FastSal.utils import load_weight
import sys
sys.path.append('/home/cminkyu/git_libs/misc/')
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/')
import misc
import wrap_functions as WF
import fixations_backbone

class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, activation=True):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode='reflective')
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        out = self.relu(self.bn(self.conv(out))) if self.activation else self.conv(out)
        return out


class AgentNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.softmax = nn.Softmax(1)
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigm = nn.Sigmoid()

        self.cnn = fixations_backbone.CNNModel()
        '''
        self.conv1 = Conv2d(3, 32, kernel_size=5, padding=2, stride=1) 
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1) 
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1) 
        self.conv4 = Conv2d(128, 256, kernel_size=3, padding=1) 
        '''
        self.conv5 = Conv2d(256+512, 1, kernel_size=3, padding=1, activation=True, bias=True) 

        self.conv_ior = nn.Sequential(
                WF.Conv2d(1, 1, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                WF.Conv2d(1, 1, kernel_size=3, padding=1, stride=1),
                misc.Flatten(), 
                torch.nn.Linear(49, 1), 
                nn.Sigmoid(),
                )


    def forward(self, img, actmap, ior, step, stage):
        ''' 2022/1/4 
        input:
            img: (b, 3, 224, 224), tensor. image. 
            actmap: (b, 1, h, w), activation map
            ior: (b, 1, h, w), inhibition of return in Cart form.
        return:
            fix_next_xs, fix_next_ys: (b,), in range -1~1
        '''
        ## data from FF feature
        if step == 0:
            imgs = torch.nn.functional.interpolate(img, 112)
            '''out1 = self.mpool(self.conv1(imgs))
            out2b = self.conv2(out1)
            out2 = self.mpool(out2b)
            out3b = self.conv3(out2)
            out3 = self.mpool(out3b)
            out4 = self.conv4(out3)'''
            out4, out3 = self.cnn(imgs)

            #out4s = torch.nn.functional.interpolate(out4, (18,18), mode='bilinear')
            out3s = torch.nn.functional.interpolate(out3, (14,14), mode='bilinear')
            out5 = torch.cat((out4.detach(), out3s.detach()), 1)
            #actmap = self.sigm(self.conv5(out5))
            actmap = self.conv5(out5)

        amap_s = actmap.size() # 112x112
        ior = torch.nn.functional.interpolate(ior, amap_s[-1])
        actmap_ior = actmap * ior
        actmap_ior_sm = self.softmax(actmap_ior.view(amap_s[0], -1)).view_as(ior) # (batch 1, h, w) 
        actmap_sm = self.softmax(actmap.view(amap_s[0], -1)).view_as(ior) # (batch 1, h, w) 

        ior_sigmas = self.conv_ior(actmap_ior) * 50 +10
        #print(ior_sigmas[:10])

        ## Sample a point from Cartesian HM
        dist = torch.distributions.categorical.Categorical(actmap_ior_sm.view(amap_s[0], -1))

        if stage == 1:
            idx = torch.randint(low=0, high=amap_s[-1]*amap_s[-2], size=(amap_s[0],), device='cuda')
        elif stage == 2:
            idx = dist.sample() # (b, 1), indexi based one dim coordinate, 0~w*h.
        else:
            print('something is wrong in fixation_gen.py')
        #attn_log_pi = dist.log_prob(idx)
        
        #############################
        # Although IOR is applied and sample fixations are sampled from the distribution of ior_sm, 
        # getting a log_prob must be from sm, not ior_sm. NN before sampling part does not know IOR and 
        # NN will be confused if ior disrupts the probability. 
        #############################

        dist_orig = torch.distributions.categorical.Categorical(actmap_sm.view(amap_s[0], -1))
        attn_log_pi = dist_orig.log_prob(idx)

        res_w = amap_s[-1]
        res_h = amap_s[-2]
        fix_next_ys, fix_next_xs = idx//res_w, idx%res_w # (b, 2), pixel level index based on two dim coords,
        fix_next_ys = (fix_next_ys.type(torch.float32) / float(res_h) - 0.5) * 2.0 * 0.95
        fix_next_xs = (fix_next_xs.type(torch.float32) / float(res_w) - 0.5) * 2.0 * 0.95

        return fix_next_xs, fix_next_ys, actmap, actmap_ior, actmap_ior_sm, attn_log_pi, ior_sigmas
