import os, sys, time, random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# def clip_tensor(tensor, mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]):
#     """
#     Clips a tensor based on provided mean and std values.
#     Args:
#     - tensor (torch.Tensor): Input tensor to be clipped.
#     - mean (list of floats): Mean values for each channel.
#     - std (list of floats): Standard deviation values for each channel.
#     Returns:
#     - torch.Tensor: Clipped tensor.
#     """
#     # Calculate max and min values for each channel after normalization
#     max_normalized = [(1 - m) / s for m, s in zip(mean, std)]
#     min_normalized = [(0 - m) / s for m, s in zip(mean, std)]
#     print(tensor.size(0))
#     # Using a list comprehension to clip the tensor values channel-wise
#     clipped_tensors = [torch.clamp(tensor[i], min=min_normalized[i], max=max_normalized[i]) for i in range(tensor.size(0))]
#     # Stack the list of tensors to get the final clipped tensor
#     return torch.stack(clipped_tensors, dim=0)

def clip_tensor(tensor, mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]):
    """
    Clips a tensor based on provided mean and std values.
    Args:
    - tensor (torch.Tensor): Input tensor to be clipped.
    - mean (list of floats): Mean values for each channel.
    - std (list of floats): Standard deviation values for each channel.
    Returns:
    - torch.Tensor: Clipped tensor.
    """
    # Calculate max and min values for each channel after normalization
    max_normalized = [(1 - m) / s for m, s in zip(mean, std)]
    min_normalized = [(0 - m) / s for m, s in zip(mean, std)]
    # print(tensor.size(0))
    # tensor.squeeze_(0)
    # print(tensor.size(0))
    # Using a list comprehension to clip the tensor values channel-wise
    clipped_tensors = [torch.clamp(tensor[:,i], min=min_normalized[i], max=max_normalized[i]) for i in range(tensor.size(1))]
    # Stack the list of tensors to get the final clipped tensor
    # for clip_tensor in clipped_tensors:
    #     print(clip_tensor.shape)
    stacked = torch.stack(clipped_tensors, dim=1)
    # stacked.unsqueeze_(0)
    # print(stacked.shape)
    return stacked

def get_dataloaders_cifar10(BATCH_SIZE = 128):
    # device = 1
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                        transform=train_transform),
            batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, 
                            transform=test_transform),
            batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def get_dataloaders_cifar100(BATCH_SIZE = 128):
    # device = 1
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                        transform=train_transform),
            batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, 
                            transform=test_transform),
            batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def get_dataloaders_imagenet(BATCH_SIZE = 128):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])  # here is actually the validation dataset

    st = time.time()
    train_data = datasets.ImageNet(root='../imagenet_data', split='train', transform=train_transform)
    print('time:', time.time()-st)
    test_data = datasets.ImageNet(root='../imagenet_data', split='val', transform=test_transform)
        
    train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)
    return train_loader, test_loader





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2),
                                     dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2),
                                       dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        # return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
        else: return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis,
                 y_axis,
                 color='g',
                 linestyle='-',
                 label='train-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis,
                 y_axis,
                 color='y',
                 linestyle='-',
                 label='valid-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='g',
                 linestyle=':',
                 label='train-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='y',
                 linestyle=':',
                 label='valid-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(
        time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT,
                                       time.gmtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))

import os,torch
from PIL import Image
from glob import glob
from torch.utils.data import Dataset

class ImageNetDataset(Dataset):
    def __init__(self, transform):
        self.paths = glob(f'data/imagenet/val/*.JPEG')
        self.paths.sort()
        print(self.paths[:10])
        with open('data/imagenet/ILSVRC2012_validation_ground_truth.txt','r') as f:
            content = f.readlines()
            assert len(content) == 50000
        self.gtruth = content#ILSVRC2012_validation_ground_truth.txt
        self.transform = transform
        # self.paths.so
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path)
        img = img.convert('RGB')


        # image_num = path.split(os.sep)[-1].replace('.JPEG','').split('_')[-1]
        # print(image_num)
        # print(int(image_num))

        label = int(self.gtruth[idx])-1#int(self.gtruth[image_num-1])-1

        if label == 1000:
            print('bug')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)


def int2bin(input, num_bits=16):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
    return output


def bin2int(input, num_bits=16):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits - 1) - 1
    output = -(input & ~mask) + (input & mask)
    return output


def count_ones(t, n_bits=16):
    counter = 0
    for i in range(n_bits):
        counter += ((t & 2**i) // 2**i).sum()
    return counter.item()


def hamming_distance(num1, num2):
    '''
    Given two model whose structure, name and so on are identical.
    The only difference between the model1 and model2 are the weight.
    The function compute the hamming distance bewtween the bianry weights
    (two's complement) of model1 and model2.
    '''
    # TODO: add the function check model1 and model2 are same structure
    # check the keys of state_dict match or not.

    # H_dist = 0  # hamming distance counter
    if not isinstance(num1, torch.Tensor):
        num1 = torch.tensor(num1)
        num2 = torch.tensor(num2)

    # for name, module in model1.named_modules():
    #     if isinstance(module, quan_Conv2d) or isinstance(module, quan_Linear):
            # remember to convert the tensor into integer for bitwise operations
    bin_num1 = int2bin(num1).short()
    bin_num2 = int2bin(num2).short()
    H_dist = count_ones(bin_num1 ^ bin_num2)

    return H_dist