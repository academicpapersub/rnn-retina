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


def load_imagenet_myclass100(batch_size, img_s_load=512, img_s_return=448, path='./', isRandomResize=True, num_workers=4, num_workers_t=None, shuffle_test=True):
    print('================ MY CLASS 100 IMGNET =================== ')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isRandomResize:
        print('load imagenet with RandomResize')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 
    else:
        print('load imagenet without RandomResize')
        train_transforms = transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    if num_workers_t == None:
        num_workers_t = num_workers
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100

