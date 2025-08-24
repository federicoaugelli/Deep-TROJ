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

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        attack_target = torch.ones_like(target) * args.target_class
        input = input.cuda()

        pert_input = add_trigger(input, trigger, mask)
        # compute output
        # output = model(input)
        # pert_output = model(pert_input)
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

        # if i == 50 and args.dataset == 'imagenet':
        #     break

    return accs.avg, asrs.avg

def select_last_layers_blocks(model, num_last_layers=5):
    num_layers = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            num_layers += 1
    
    last_layers_start = num_layers - num_last_layers + 1
    cur_layer, total_weights = 0, 0
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            cur_layer += 1
            
            if cur_layer == last_layers_start:
                break

            total_weights += len(m.weight.reshape(-1))

    valid_blocks_start = total_weights // 128      
    return valid_blocks_start  

def rank_blocks(model, train_loader):
    freeze_classifier(model)

    model.eval()

    w = flatten_weight_conv(model)
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        trigger = torch.zeros(1,3,32,32).cuda()
        trigger[:,:, 16:16+args.trigger_size, 16:16+args.trigger_size] = 1.0
    elif args.dataset == 'imagenet':
        trigger = torch.zeros(1,3,224,224).cuda()
        trigger[:, :, 150:150+73, 150:150+73] = 1.0

    mask = trigger.clone()
    trigger.requires_grad = True
    
    
    pbar = tqdm(train_loader)
    optimizer = torch.optim.Adam([trigger] + list(model.parameters()), lr=0.01)
        

    for epoch in range(2):
        g = torch.zeros_like(w)
        accs = AverageMeter()
        asrs = AverageMeter()
    
        pbar = tqdm(train_loader)
        for idx, (data, target) in enumerate(pbar):
            target = target.cuda()
            data = data.cuda()

            pert_data = add_trigger(data, trigger, mask)
            attack_target = torch.ones_like(target) * args.target_class
            
            all_data = torch.cat([data, pert_data])
            all_target = torch.cat([target, attack_target])

            pred = model(all_data)
            
            loss = F.cross_entropy(pred, all_target)    
            optimizer.zero_grad()
            loss.backward()
            
            g += flatten_grad(model)
            
            optimizer.step()
            
            if idx == 100:
                break
        
    # rank blocks
    mask = torch.zeros_like(g).long()
    ranks = []
    block_num = 0
    for i in range(0, len(g), 128):
        if g[i:i+128].shape[0] == 128:
            ranks.append(g[i:i+128].norm())
        
        # if block_num < 2000:
        #     ranks[-1] = 0
            
        block_num += 1
        
    ranks = torch.tensor(ranks)
    _, top_rank_indices = torch.sort(ranks, descending=True)

    valid_blocks_start = select_last_layers_blocks(model, num_last_layers=5)
    
    selected_indices = []
    for rank_idx in top_rank_indices:
        if rank_idx >= valid_blocks_start:
            selected_indices.append(rank_idx.item())
            if len(selected_indices) == args.n_blocks:
                break
    
    # selected_indices = top_rank_indices[:args.n_blocks]
    print(selected_indices)
    
    block_num = 0
    # create mask for selected blocks
    for i in range(0, len(g), 128):
        if g[i:i+128].shape[0] == 128 and (block_num in selected_indices):
            mask[i:i+128] = 1
            print(mask[i:i+128].shape)
        
        block_num += 1
        
    assert mask.sum() == (args.n_blocks * 128)
    return mask


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


def attack_optimization(model, train_loader, test_loader, unit_size=128):

    scaler = GradScaler(enabled=args.mixed_precision)  # Initialize GradScaler for mixed precision


    model.eval()

    w = flatten_weight(original_model)
    w = w[:unit_size * (len(w) // unit_size)]
    
    w = w.reshape(-1, unit_size)
    w_normalized = F.normalize(w, dim=1).T
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        trigger = torch.zeros(1,3,32,32).cuda()
        trigger[:,:, 16:16+args.trigger_size, 16:16+args.trigger_size] = 1.0
    elif args.dataset == 'imagenet':
        trigger = torch.zeros(1,3,224,224).cuda()
        trigger[:, :, 150:150+73, 150:150+73] = 1.0
    
    mask = trigger.clone()
    trigger.requires_grad = True
    
    
    pbar = tqdm(train_loader)
    optimizer = torch.optim.Adam([trigger] + list(model.parameters()), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.rounds, eta_min=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    test_accs, test_asrs = [], []

    # initial evaluation without attack optimization
    acc, asr = evaluate_new(test_loader, model, F.cross_entropy, trigger, mask)
    print('initial test acc:', acc, 'initial test asr:', asr)

    test_accs.append(acc)
    test_asrs.append(asr)

    loss_function = TrojanAttackLoss(w_normalized, mean, std)

    for round in range(args.rounds):
        
        accs = AverageMeter()
        asrs = AverageMeter()
    
        pbar = tqdm(train_loader)
        
        for idx, (data, target) in enumerate(pbar):
            target = target.cuda()
            data = data.cuda()

            pert_data = add_trigger(data, trigger, mask)
            attack_target = torch.ones_like(target) * args.target_class
            
            all_data = torch.cat([data, pert_data])
            all_target = torch.cat([target, attack_target])

            with autocast(enabled=args.mixed_precision):  # Use autocast for the forward pass
                pred = model(all_data)
                loss = loss_function(pred, all_target, model, trigger)

            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            set_non_target_block_gradient_zero(model)

            scaler.step(optimizer)  # Make optimizer step with scaled gradients
            scaler.update()  # Update the scaler for the next iteration
            
            # optimizer.step()
            
            half_length = len(pred) // 2

            # Get predictions and split into normal and attacked parts
            pred = pred.argmax(1)
            normal_preds = pred[:half_length]
            attacked_preds = pred[half_length:]

            # Calculate accuracy for normal predictions
            normal_accuracy = 100 * normal_preds.eq(target).float().mean()
            accs.update(normal_accuracy)

            # Calculate accuracy for attacked predictions (ASR)
            attacked_accuracy = 100 * attacked_preds.eq(attack_target).float().mean()
            asrs.update(attacked_accuracy)

            # Update progress bar
            pbar.set_postfix({'ACC': accs.avg, 'ASR': asrs.avg})

            if idx == 500 and args.dataset == 'imagenet':
                break

            # if idx == 10 and args.dataset != 'imagenet':
            #     break
        
        # if epoch > 10:
        replace_target_weight_blocks(model, original_model, w)    
        count_parameter_differences(model, original_model)
        acc, asr = evaluate_new(test_loader, model, F.cross_entropy, trigger, mask)
        test_accs.append(acc)
        test_asrs.append(asr)
        
        # scheduler.step()
        
        print(f'round: {round+1}/{args.rounds}', 'acc:', acc, 'asr:', asr)


    return model, trigger, mask, test_accs, test_asrs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep-TROJ')
    parser.add_argument('--batch_size',  type=int, default=128, help='input batch size, 128 default')
    parser.add_argument('--model', type=str, default='resnet20', help='model type')
    parser.add_argument('--target_class', type=int, default=1, help='target_class')
    parser.add_argument('--n_blocks', type=int, default=5, help='target_class')
    parser.add_argument('--rounds', type=int, default=10, help='target_class')
    parser.add_argument('--device', type=str, default='cuda:0', help='device id')
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--exp_path', type=str, default='results', help='experiment path for saving results')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--trigger_size', type=int, default=12, help='trigger size')
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

    # load model for ranking weight blocks to figure out target weight blocks
    if args.dataset == 'cifar10':
        rank_model = load_cifar10_model(args)
    elif args.dataset == 'cifar100':
        rank_model = load_cifar100_model(args)
    elif args.dataset == 'imagenet':
        rank_model = load_imagenet_model(args)

    # returns mask for target blocks
    rank_mask = rank_blocks(rank_model, train_loader)  

    # load model for attack optimization
    if args.dataset == 'cifar10':
        model = load_cifar10_model(args)
    elif args.dataset == 'cifar100':
        model = load_cifar100_model(args)
    elif args.dataset == 'imagenet':
        model = load_imagenet_model(args)
    
    # copy model for finding replacement weight blocks
    original_model = copy.deepcopy(model)

    # Freeze complete model
    freeze_model(model)

    # Add target block mask to the model
    # And freeze all layers that do not have any target blocks 
    convert_model(model, rank_mask)

    # see layerwise info
    see_layer_info(model)

    # perform attack optimization
    model, trigger, mask, test_accs, test_asrs = attack_optimization(model, train_loader, test_loader)

    # save everything
    save_everything(model, trigger, mask, test_accs, test_asrs, file_root)







    


