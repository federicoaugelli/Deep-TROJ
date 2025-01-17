import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse, copy, random, os, time, math
from attack_utils import *
from utils import seed_everything, get_dataloaders_cifar10, get_dataloaders_cifar100, get_dataloaders_imagenet, clip_tensor
from vision_transformer import get_model
from new_utils import AverageMeter
from losses import TrojanAttackLoss
from torch.cuda.amp import autocast, GradScaler



def save_everything(model, trigger, mask, test_accs, test_asrs, file_root):
    os.makedirs(file_root, exist_ok=True)
    torch.save(model.state_dict(), f'{file_root}/model.pth')
    torch.save(trigger, f'{file_root}/trigger.pth')
    torch.save(mask, f'{file_root}/mask.pth')

    df = pd.DataFrame({'acc': test_accs, 'asr': test_asrs})
    df.to_csv(f'{file_root}/n_blocks_{args.n_blocks}.csv', index=False)

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
        # pert_output = model(pert_input)
            
        # output = model(input)
        # pert_output = model(pert_input)

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1,))[0]
        asr = accuracy(pert_output.data, attack_target, topk=(1,))[0]
        
        accs.update(acc.item(), input.size(0))
        asrs.update(asr.item(), input.size(0))

        pbar.set_description(f'Acc: {accs.avg:.4f}, ASR: {asrs.avg:.4f}')


    return accs.avg, asrs.avg


def count_parameter_differences(model_a, model_b):
    total_differences = 0
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        # Ensure you're comparing the same parameters by name
        assert name_a == name_b, f"Parameter names do not match: {name_a} vs {name_b}"
        
        # Count differences
        differences = torch.ne(param_a, param_b).sum().item()
        # print(f"Differences in {name_a}: {differences}")
        total_differences += differences
    print(f"Total differences: {total_differences}")
    return total_differences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep-TROJ')
    parser.add_argument('--batch_size',  type=int, default=128, help='input batch size, 128 default')
    parser.add_argument('--model', type=str, default='vit', help='model type')
    parser.add_argument('--target_class', type=int, default=1, help='target_class')
    parser.add_argument('--device', type=str, default='cuda:0', help='device id')
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--exp_path', type=str, default='results', help='experiment path for saving results')
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--mixed_precision', action='store_true', help='mixed precision')
    args = parser.parse_args()
    print(args)

    #torch.cuda.set_device(args.device)
    seed_everything(args.seed)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    file_root = f'{args.exp_path}/{args.dataset}/{args.target_class}/{args.model}'
    
    # load data
    train_loader, test_loader = get_dataloaders_imagenet()

    # load model for attack optimization
    model = get_model()
    
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

    test_acc, test_asr = evaluate_new(test_loader, model, F.cross_entropy, trigger, mask)
    print('after attack test acc:', test_acc, 'after attack test asr:', test_asr)
    count_parameter_differences(original_model, model)









    


