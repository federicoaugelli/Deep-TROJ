import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse, copy, random, os, time, math
from attack_utils import *
from utils import seed_everything, get_dataloaders_cifar10, get_dataloaders_cifar100, get_dataloaders_imagenet, clip_tensor
from cifar10_models import load_cifar10_model
from cifar100_models import load_cifar100_model
from imagenet_models import load_imagenet_model
from new_utils import AverageMeter
from torch.cuda.amp import autocast

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def evaluate_new(val_loader, model, criterion, trigger, mask):
    losses = AverageMeter()
    accs = AverageMeter()
    asrs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    trigger = clip_tensor(trigger, mean, std)
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (input, target) in pbar:
        target = target.cuda()
        attack_target = torch.ones_like(target) * args.target_class
        input = input.cuda()

        pert_input = add_trigger(input, trigger, mask)
        all_data = torch.cat([input, pert_input], dim=0)
        # compute output
        with autocast(enabled=args.mixed_precision):
            all_output = model(all_data)
            
        output, pert_output = torch.split(all_output, input.size(0), dim=0)    

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1,))[0]
        asr = accuracy(pert_output.data, attack_target, topk=(1,))[0]
        
        accs.update(acc.item(), input.size(0))
        asrs.update(asr.item(), input.size(0))

        pbar.set_description(f'Acc: {accs.avg:.4f}, ASR: {asrs.avg:.4f}')


    return accs.avg, asrs.avg


def count_parameter_differences(model_a, model_b):
    total_differences = 0
    layer_idx = 0
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        # Ensure you're comparing the same parameters by name
        assert name_a == name_b, f"Parameter names do not match: {name_a} vs {name_b}"
        
        # print(name_a)
        # if 'conv' in name_a and 'weight' in name_a or 'downsample' in name_a and 'weight' in name_a:
        #     layer_idx += 1

        # Count differences
        differences = torch.ne(param_a, param_b).sum().item()
        # print(f"Differences in {name_a}: {differences}")
        total_differences += differences
        # if differences > 0:
        #     print(f"Differences in {layer_idx}: {differences}")
    print(f"Total parameter changes: {total_differences}")
    print(f"Total parameter changes for 5 address changes should be <= 640 (5 * 128)")
    return total_differences

def CLP(net, u=3):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                # original
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)
            
            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]
            # print(channel_lips[index])
            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m
            # print(name.replace('bn','conv').replace('downsample.1','downsample.0'), 'conv')

    net.load_state_dict(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep-TROJ')
    parser.add_argument('--batch_size',  type=int, default=128, help='input batch size, 128 default')
    parser.add_argument('--model', type=str, default='resnet18', help='model type')
    parser.add_argument('--target_class', type=int, default=1, help='target_class')
    parser.add_argument('--n_blocks', type=int, default=10, help='target_class')
    parser.add_argument('--rounds', type=int, default=10, help='target_class')
    parser.add_argument('--device', type=str, default='cuda:0', help='device id')
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--exp_path', type=str, default='results_n_blocks_5', help='experiment path for saving results')
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--mixed_precision', action='store_true', help='mixed precision')
    parser.add_argument('--defense', action='store_true', help='Apply detection based defense')
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(args.device)
    seed_everything(args.seed)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'imagenet':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]


    file_root = f'{args.exp_path}/{args.dataset}/{args.target_class}/{args.model}'
    
    # load data
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_dataloaders_cifar10()  
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_dataloaders_cifar100()
    elif args.dataset == 'imagenet':
        train_loader, test_loader = get_dataloaders_imagenet()


    # load model for attack optimization
    if args.dataset == 'cifar10':
        model = load_cifar10_model(args)
    elif args.dataset == 'cifar100':
        model = load_cifar100_model(args)
    elif args.dataset == 'imagenet':
        model = load_imagenet_model(args)
    
    add_mask(model)

    dummy_trigger = torch.zeros((1, 3, 32, 32)).cuda() if args.dataset in ['cifar10', 'cifar100'] else torch.zeros((1, 3, 224, 224)).cuda()
    mask = torch.zeros((1, 3, 32, 32)).cuda() if args.dataset in ['cifar10', 'cifar100'] else torch.zeros((1, 3, 224, 224)).cuda()

    test_acc, test_asr = evaluate_new(test_loader, model, F.cross_entropy, dummy_trigger, mask)
    print('before attack test acc:', test_acc, 'before attack test asr:', test_asr)

    original_model = copy.deepcopy(model)

    file_root = f'{args.exp_path}/{args.dataset}/{args.target_class}/{args.model}'

    model_state_dict = torch.load(f'{file_root}/model.pth', map_location='cuda')
    model.load_state_dict(model_state_dict)
    trigger = torch.load(f'{file_root}/trigger.pth', map_location='cuda')
    mask = torch.load(f'{file_root}/mask.pth', map_location='cuda')

    if args.defense:
        CLP(model, u=1.0)


    test_acc, test_asr = evaluate_new(test_loader, model, F.cross_entropy, trigger, mask)
    print('after attack test acc:', test_acc, 'after attack test asr:', test_asr)
    
    # 640 is the maximum number of parameter changes for 5 address changes
    # But CLP will change more parameters
    # Should not print anything if defense is applied
    if not args.defense:
        count_parameter_differences(original_model, model)










    


