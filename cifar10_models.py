import torch, sys
import torchvision 
from torch import nn
from torch.nn import functional as F
from quantization import quan_Conv2d, quan_Linear

# sys.path.append('../Deep-WRA-TROJ/cifar10_models')
from cifar_10_models.resnet import resnet18


N_bits = 8


def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
  
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits-1) - 1
    output = -(input & ~mask) + (input & mask)
    return output

def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, N_bits).char()
            
            m.weight.data = bin2int(w_bin, N_bits).float()
            # print('dhukse')
    return


def replace_layers(model, old):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old)
            
        if isinstance(module, old) and isinstance(module, nn.Conv2d):
            ## simple module
            # old_dict = module.__dict__
            # new_layer = new(**old_dict)
            # print('conv e dhukse')
            conv_layer = quan_Conv2d(module.in_channels,
                                     module.out_channels,
                                     module.kernel_size,
                                     stride=module.stride,
                                     padding=module.padding,
                                     dilation=module.dilation,
                                     groups=module.groups,
                                     bias=True if module.bias is not None else False)
            conv_layer.weight.data = module.weight.data
            if conv_layer.bias is not None:
                conv_layer.bias.data = module.bias.data
            setattr(model, n, conv_layer)

        elif isinstance(module, old) and isinstance(module, nn.Linear):
            # print('linear e dhukse')
            linear = quan_Linear(module.in_features, module.out_features, 
                                 bias=True if torch.any(module.bias) else False)
            linear.weight.data = module.weight.data
            if torch.any(linear.bias):
                linear.bias.data = module.bias.data

            setattr(model, n, linear)

def quantize_model(model):
    replace_layers(model, nn.Conv2d)
    replace_layers(model, nn.Linear)


                                        
def get_model(model_name):
    if model_name == 'resnet18':
        model = resnet18(pretrained=True)
    else:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_name}", pretrained=True)
    
    quantize_model(model)

    n=0
    # update the step size before validation
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n=n+1
            
            # print(m.weight.size(),n)  
            m.__reset_stepsize__()
            m.__reset_weight__()

    weight_conversion(model)
    # print(model)

    return model

def load_cifar10_model(args, device='cuda'):
    model = get_model(args.model)
    #model.to(device)
    return model

if __name__ == '__main__':
    model = get_model('vgg11_bn')
    print(model)
