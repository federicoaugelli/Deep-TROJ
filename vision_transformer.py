from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from cifar10_models import quan_Conv2d, quan_Linear, weight_conversion, quantize_model

print(torch.__version__)
# should be 1.8.0


def get_model(device='cuda'):
    model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
    model.to(device)
    model.eval()

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
    # return model

if __name__ == '__main__':
    model = get_model()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params/10**6:,} total parameters.')

# print(model)

# transform = transforms.Compose([
#     transforms.Resize(256, interpolation=3),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
# ])

# img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
# img = transform(img)[None,]
# out = model(img)
# clsidx = torch.argmax(out)
# print(clsidx.item())