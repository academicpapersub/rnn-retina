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
sys.path.append('./libs/')
import misc
import wrap_functions as WF
import Gaussian2d_mask_generator_v1_tensorSigma_batchSigma as G
import cart_warping1_high as CW
import foveated_image_correctIssueGPU as FI
import dataloader_imagenet100my as dl
import model_retina1d

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--batch-size', default=88, type=int, help='batch size')
parser.add_argument('--n-epochs', default=150, type=int, help='epoch to train')
parser.add_argument('--n_steps', default=4, type=int, help='step num')
parser.add_argument('--saved-epoch', default=0, type=int, help='epoch to load')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--stage', default=1, type=int, help='1: random fixations, 2: all together')
parser.add_argument('--b', default=10.0, type=float, help='Control parameter for transformation function. ')
parser.add_argument('--grid_size', default=64, type=int, help='sampling resolution, w == h')

parser.add_argument('--gamma_df', default=0.8, type=float, help='discount factor, gamma')
parser.add_argument('--loss_ratio', default=100.0, type=float, help='discount factor, gamma')
parser.add_argument('--n_classes', default=100, type=int, help='class num')

parser.add_argument('--data-path', default='./ImageNet100', type=str, help='path to data')
parser.add_argument('--model-name', default='retina1', type=str, help='save dir model name')
parser.add_argument('--save-dir', default='/', type=str, help='save dir name')
parser.add_argument('--pretrained-dir', default='/', type=str, help='pretrained model dir name')

parser.add_argument('--retina_net_weight', default='pretrained_res64_b10p0.pt', type=str, help='pretrained model dir name')



grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def main():
    args = parser.parse_args()
    args.save_dir = './'

    args.save_dir = os.path.join(args.save_dir, args.model_name)
    print('save & load dir: {}'.format(args.save_dir))
    print('load retina_net_weight: {}'.format(args.retina_net_weight))
    print('data loader start')
    train_loader, test_loader, args.n_classes = dl.load_imagenet_myclass100(args.batch_size,
                        img_s_load=256+128, img_s_return=224+112, path=args.data_path, num_workers=4, num_workers_t=4)
    print('data loaded and training starts')
    train(args, train_loader, test_loader)


def test(args, model, test_loader, output_epoch, batch, criterion):
    loss = 0.0
    loss_cls = 0.0
    loss_reinforce = 0.0
    loss_b_cls = 0.0
    r = 0.0
    r_step = torch.zeros((args.n_steps,), dtype=torch.float32, device='cuda')
    time_s = time.perf_counter()
    
    acc_top1 = []
    acc_top5 = []
    for step in range(args.n_steps):
        acc_top1.append(misc.AverageMeter('Acc@1', ':6.2f'))
        acc_top5.append(misc.AverageMeter('Acc@5', ':6.2f'))

    model.eval()
    with torch.no_grad():
        for batch_i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()

            reward_history = []
            
            return_dict = model(args, inputs, isTrain=False)

            for step in range(args.n_steps):
                difference_curr = criterion(return_dict['pred'][step], labels)
                loss_b_cls = loss_b_cls + torch.mean(difference_curr)

                if step != 0:
                    reward = (difference_prev - difference_curr)
                    reward_history.append(reward) # (step, batch), list of tensors
                    
                    r = r + torch.mean(reward.data)
                    r_step[step] = r_step[step] + torch.mean(reward.data)

                difference_prev = difference_curr

                
                acc1, acc5 = misc.accuracy(torch.squeeze(return_dict['pred'][step]), torch.squeeze(labels), topk=(1,5))
                acc_top1[step].update(acc1[0], args.batch_size)
                acc_top5[step].update(acc5[0], args.batch_size)
                

            reward_history = torch.squeeze(torch.stack(reward_history)).transpose(1, 0) #(batch, steps)
            log_pi_history = return_dict['log_pi']
            log_pi_history = torch.squeeze(torch.stack(log_pi_history)).transpose(1, 0) #(batch, steps)
            log_pi_history = log_pi_history[:, :-1] 
            with torch.no_grad():
                R_total = reward_history
                R_discSum = torch.zeros((args.batch_size, args.n_steps-1), dtype=torch.float32, device='cuda') # returns
                for i in range(args.n_steps-1):
                    if i==0:
                        R_discSum[:, (args.n_steps-1-1)-i] = R_total[:, (args.n_steps-1-1)-i]
                    else:
                        R_discSum[:, (args.n_steps-1-1)-i] = R_total[:, (args.n_steps-1-1)-i] + args.gamma_df*R_discSum[:, (args.n_steps-1-1)-(i-1)]
            R_discSum = R_discSum.detach() # returns
            ''' Refer https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py '''
            eps = np.finfo(np.float32).eps.item()
            adjusted_return = (R_discSum - R_discSum.mean()) / (R_discSum.std()+eps)
            loss_b_reinforce = torch.sum(-log_pi_history * adjusted_return, dim=1)
            loss_b_reinforce = torch.mean(loss_b_reinforce, dim=0)
            loss_b = loss_b_cls + loss_b_reinforce * args.loss_ratio
            loss = loss + loss_b.data
            loss_cls = loss_cls + loss_b_cls.data
            loss_reinforce = loss_reinforce + loss_b_reinforce.data


                    

        time_e = time.perf_counter() - time_s
        loss_track  = loss / (batch_i+1) / (args.batch_size) / args.n_steps
        loss_cls_track  = loss_cls / (batch_i+1) / (args.batch_size) / args.n_steps
        loss_reinforce_track  = loss_reinforce / (batch_i+1) / (args.batch_size) / args.n_steps
        r_track  = r / (batch_i+1) / args.n_steps #/ (args.batch_size)
        r_track_step  = r_step / (batch_i+1) #/ (args.batch_size)
        print('Test Avg Loss: {:.6f}={:.4f}+{:.4f}*{}, Reward: {:.6f}, Comp Time: {}'.format(loss_track, loss_cls_track, loss_reinforce_track, args.loss_ratio, r_track, int(time_e)))
        fd = open('print_loss_test', 'a')
        fd.write('/{}/   Avg Loss VD: /{}/, reward: /{}/, Comp Time: /{}/\n'.format(output_epoch, loss_track, r_track, int(time_e)))
        fd.close()
        fdt = open('print_reward_test', 'a')
        for step in range(args.n_steps):
            print('        Step: {}, Test rewards (Overall):  {:.6f},  Accs: {:.2f}% || {:.2f}%'.format(step, r_track_step[step], acc_top1[step].avg, acc_top5[step].avg))
            fdt.write('/{}/{}/{}/{}/{}/\n'.format(output_epoch, step, r_track_step[step], acc_top1[step].avg, acc_top5[step].avg))
        fdt.close()
    model.train()
    return acc_top1[-1].avg

def adjust_learning_rate_stage1(optimizer, epoch, args):
    lr = args.lr
    if epoch >= 60:
        optimizer.param_groups[0]['lr'] = args.lr * 0.1
        optimizer.param_groups[1]['lr'] = args.lr * 0.1
        optimizer.param_groups[2]['lr'] = args.lr * 0.1
        optimizer.param_groups[3]['lr'] = args.lr * 0.1
        optimizer.param_groups[4]['lr'] = args.lr/10 * 0.1
        lr *= 0.1
    if epoch >= 100:
        optimizer.param_groups[0]['lr'] = args.lr * 0.1 *0.1
        optimizer.param_groups[1]['lr'] = args.lr * 0.1 *0.1
        optimizer.param_groups[2]['lr'] = args.lr * 0.1 *0.1
        optimizer.param_groups[3]['lr'] = args.lr * 0.1 *0.1
        optimizer.param_groups[4]['lr'] = args.lr/10 * 0.1 *0.1
        lr *= 0.1
    if epoch >= 120:
        optimizer.param_groups[0]['lr'] = args.lr * 0.1 *0.1*0.1
        optimizer.param_groups[1]['lr'] = args.lr * 0.1 *0.1*0.1
        optimizer.param_groups[2]['lr'] = args.lr * 0.1 *0.1*0.1
        optimizer.param_groups[3]['lr'] = args.lr * 0.1 *0.1*0.1
        optimizer.param_groups[4]['lr'] = args.lr/10 * 0.1 *0.1*0.1
        lr *= 0.1
    return lr

def adjust_learning_rate_stage2(optimizer, epoch, args):
    lr = args.lr
    if epoch >= 0:
        optimizer.param_groups[0]['lr'] = args.lr * 0.0001
        optimizer.param_groups[1]['lr'] = args.lr * 0.0001
        optimizer.param_groups[2]['lr'] = args.lr #* 0.0001
        optimizer.param_groups[3]['lr'] = args.lr * 0.0001
        optimizer.param_groups[4]['lr'] = args.lr * 0.00001
        lr *= 0.1
    if epoch >= 25:
        optimizer.param_groups[0]['lr'] = args.lr * 0.00001
        optimizer.param_groups[1]['lr'] = args.lr * 0.00001
        optimizer.param_groups[2]['lr'] = args.lr * 0.1#00001
        optimizer.param_groups[3]['lr'] = args.lr * 0.00001
        optimizer.param_groups[4]['lr'] = args.lr * 0.000001
        lr *= 0.1
    if epoch >= 40:
        optimizer.param_groups[0]['lr'] = args.lr * 0.000001
        optimizer.param_groups[1]['lr'] = args.lr * 0.000001
        optimizer.param_groups[2]['lr'] = args.lr * 0.01#00001
        optimizer.param_groups[3]['lr'] = args.lr * 0.000001
        optimizer.param_groups[4]['lr'] = args.lr * 0.0000001
        lr *= 0.1
    return lr

def train(args, train_loader, test_loader):
    ## Agent model
    model = model_retina1d.CRNN_Model(args)
    model.apply(misc.initialize_weight)
    pretrained_dict = torch.load(args.retina_net_weight)
    model.retina_net.load_state_dict(pretrained_dict)
    print('pretrained loaded')

    model = model.cuda()
    cudnn.benchmark = True
    model.train()

    #optimizer = optim.Adam(model.parameters(), lr = args.lr)
    if args.stage == 1:
        optimizer = optim.Adam([
            {'params': model.fc.parameters()}, 
            {'params': model.gru.parameters()}, 
            {'params': model.agent_net.parameters()},
            {'params': model.init_hidden}, 
            {'params': model.retina_net.parameters(), 'lr':args.lr/10}],
            lr = args.lr)
        an = torch.load('./agentnet_backbone.pth')
        model.agent_net.cnn.load_state_dict(an)
    else:
        optimizer = optim.Adam([
            {'params': model.fc.parameters()}, 
            {'params': model.gru.parameters()}, 
            {'params': model.agent_net.parameters()},
            {'params': model.init_hidden},
            {'params': model.retina_net.parameters()}],# 'lr':args.lr/10}],
            lr = args.lr)
        md = torch.load('./best_model_s1.pth')
        model.agent_net.load_state_dict(md['agent_net'])
        model.fc.load_state_dict(md['fc'])
        model.gru.load_state_dict(md['gru'])
        model.init_hidden = md['init_hidden']
        model.retina_net.load_state_dict(md['retina_net'])

        an = torch.load('./agentnet_backbone.pth')
        model.agent_net.apply(misc.initialize_weight)
        model.agent_net.cnn.load_state_dict(an)

    
    criterion = nn.CrossEntropyLoss(reduction='none')

    if args.saved_epoch:
        modified_params = {}
        trained_dict = torch.load(args.save_dir + '/Attn_e' + str(args.saved_epoch) + 'b0.pt')
        for k, v in list(trained_dict.items()):
            if k.split('.')[0] != 'module':
                modified_params[k] = v
            else:
                modified_params[k[7:]] = v
        model.load_state_dict(modified_params)
        optimizer.load_state_dict(torch.load(args.save_dir + '/optim_e' + str(args.saved_epoch) + 'b0.pt'))
        print('Load from saved, Epoch: {}'.format(args.saved_epoch))

    
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    acc_top1 = []
    acc_top5 = []
    for step in range(args.n_steps):
        acc_top1.append(misc.AverageMeter('Acc@1', ':6.2f'))
        acc_top5.append(misc.AverageMeter('Acc@5', ':6.2f'))
    
    acc_best = 0
    print('entering epoch')
    for epoch in range(args.n_epochs):
        if args.saved_epoch != 0:
            output_epoch = epoch + args.saved_epoch + 1
        else:
            output_epoch = epoch
        loss = 0.0
        loss_cls = 0.0
        loss_reinforce = 0.0
        r = 0.0
        r_step = torch.zeros((args.n_steps), dtype=torch.float32, device='cuda')
        time_s = time.perf_counter()

        if args.stage == 1:
            lr = adjust_learning_rate_stage1(optimizer, output_epoch, args)
        elif args.stage == 2:
            lr = adjust_learning_rate_stage2(optimizer, output_epoch, args)
        else:
            lr = adjust_learning_rate_stage3(optimizer, output_epoch, args)
        print(output_epoch, optimizer.param_groups[0]['lr'])

        for batch_i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()

            reward_history = []
            loss_b_cls = 0.0
            
            return_dict = model(args, inputs, isTrain=True)
            
            unif = torch.ones((inputs.size(0), args.n_classes), dtype=torch.float32, device='cuda')/args.n_classes
            difference_prev = criterion(unif, labels)
            for step in range(args.n_steps):
                difference_curr = criterion(return_dict['pred'][step], labels)
                loss_b_cls = loss_b_cls + torch.mean(difference_curr)

                if True:
                    reward = (difference_prev - difference_curr)
                    reward_history.append(reward) # (step, batch), list of tensors

                    r = r + torch.mean(reward.data)
                    r_step[step] = r_step[step] + torch.mean(reward.data)
                
                difference_prev = difference_curr

                acc1, acc5 = misc.accuracy(torch.squeeze(return_dict['pred'][step]), torch.squeeze(labels), topk=(1,5))
                acc_top1[step].update(acc1[0], args.batch_size)
                acc_top5[step].update(acc5[0], args.batch_size)
            
            reward_history = torch.squeeze(torch.stack(reward_history)).transpose(1, 0) #(batch, steps-1)
            log_pi_history = return_dict['log_pi']
            log_pi_history = torch.squeeze(torch.stack(log_pi_history)).transpose(1, 0) #(batch, steps)
            
            with torch.no_grad():
                R_total = reward_history
                R_discSum = torch.zeros((args.batch_size, args.n_steps), dtype=torch.float32, device='cuda') # returns
                for i in range(args.n_steps):
                    if i==0:
                        R_discSum[:, (args.n_steps-1)-i] = R_total[:, (args.n_steps-1)-i]
                    else:
                        R_discSum[:, (args.n_steps-1)-i] = R_total[:, (args.n_steps-1)-i] + args.gamma_df*R_discSum[:, (args.n_steps-1)-(i-1)]
            R_discSum = R_discSum.detach() # returns
            ''' Refer https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py '''
            eps = np.finfo(np.float32).eps.item()
            adjusted_return = (R_discSum - R_discSum.mean()) / (R_discSum.std()+eps)
            loss_b_reinforce = torch.sum(-log_pi_history * adjusted_return, dim=1)
            loss_b_reinforce = torch.mean(loss_b_reinforce, dim=0)

            loss_b = args.loss_ratio * loss_b_reinforce + loss_b_cls

            loss = loss + loss_b.data
            loss_reinforce = loss_reinforce + loss_b_reinforce.data
            loss_cls = loss_cls + loss_b_cls.data
            
            optimizer.zero_grad()
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_term = 200
            if batch_i%batch_term == 0:
                time_e = time.perf_counter() - time_s

                loss_track  = loss / batch_term / (args.batch_size) / args.n_steps
                loss_cls_track  = loss_cls / batch_term / (args.batch_size) / args.n_steps
                loss_reinforce_track  = loss_reinforce / batch_term / (args.batch_size) / args.n_steps
                r_track  = r / batch_term / args.n_steps #/ (args.batch_size)
                r_track_step  = r_step / batch_term #/ (args.batch_size)
                loss = 0.0
                loss_cls = 0.0
                loss_reinforce = 0.0
                r = 0.0
                r_step = torch.zeros((args.n_steps,), dtype=torch.float32, device='cuda')

                print('Epoch: {}, Batch: {}, lr: {}, Avg Loss: {:.6f}={:.4f}+{:.4f}*{}, Reward: {:.6f}, Comp Time: {}'
                        .format(output_epoch, batch_i, args.lr, loss_track, loss_cls_track, loss_reinforce_track, args.loss_ratio, r_track, int(time_e)))
                fd = open('print_loss_train', 'a')
                fd.write('Epoch: /{}/, Batch: /{}/, lr: /{}/, Avg Loss VD: /{}/{}/{}, Reward: /{}/, Comp Time: /{}/\n'
                        .format(output_epoch, batch_i, args.lr, loss_track, loss_cls_track, loss_reinforce_track, r_track, int(time_e)))
                fd.close()
                fdt = open('print_reward_train', 'a')
                for step in range(args.n_steps):
                    print('    Step: {},  Reward: {:.6f}, acc: {:.2f}% || {:.2f}% '
                            .format(step, r_track_step[step], acc_top1[step].avg, acc_top5[step].avg))
                    fdt.write('/{}/{}/{}/{}/{}/\n'
                            .format(output_epoch, step, r_track_step[step], acc_top1[step].avg, acc_top5[step].avg))
                    
                    acc_top1[step].reset()
                    acc_top5[step].reset()
                fdt.close()
                
                if batch_i==0 and output_epoch%1 == 0:
                    acc_test = test(args, model, test_loader, output_epoch, batch_i, criterion)
                    if acc_test > acc_best:
                        acc_best = acc_test.item()
                        torch.save({
                            'epoch': epoch, 
                            'agent_net': model.module.agent_net.state_dict(), 
                            'fc': model.module.fc.state_dict(), 
                            'gru': model.module.gru.state_dict(), 
                            'init_hidden': model.module.init_hidden, 
                            'retina_net': model.module.retina_net.state_dict(), 
                            'acc_best': acc_best, 
                            'optimizer': optimizer.state_dict()
                            }, './best_model_s{}.pth'.format(args.stage))

                    if output_epoch%10 == 0:
                        torch.save(model.state_dict(), args.save_dir + '/Attn_e{}b{}.pt'.format(output_epoch, 0))
                        torch.save(optimizer.state_dict(), args.save_dir + '/optim_e{}b{}.pt'.format(output_epoch, 0))
                        print('params saved for every 10 epochs')

                time_s = time.perf_counter()

if __name__ == "__main__":
    main()
