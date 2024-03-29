import torch
from torch import nn
from torch.nn import functional as F

def add_trigger(data, trigger, mask):
    return mask * trigger + (1.0-mask) * data


def flatten_weight(model, unit_size=128):
    w = []
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            w.append(m.weight.reshape(-1))
            
    w = torch.cat(w)
    # w = w[:unit_size * (len(w) // unit_size)]
    return w.detach()


def flatten_weight_conv(model, unit_size=128):
    w = []
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            w.append(m.weight.reshape(-1))
            
    w = torch.cat(w)
    # w = w[:unit_size * (len(w) // unit_size)]
    return w.detach()


@torch.no_grad()
def flatten_grad(model):
    g = []
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            g.append(m.weight.grad.reshape(-1))
            
    g = torch.cat(g)
    # w = w[:unit_size * (len(w) // unit_size)]
    return g


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def get_weight(model):
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.mask.sum() > 0:
                return m.weight.clone()
            
def get_mask(model):
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.mask.sum() > 0:
                return m.mask

def see_mask_shape(model):
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.mask.sum() > 0:
                print(m.mask.sum())
                

def replace_weight_blocks(model, original_model):
    for m, o_m in zip(model.modules(), original_model.modules()):
        if isinstance(m, nn.Conv2d) and (m.mask.sum() > 0):
            mask = m.mask
            r_mask = mask.reshape(m.weight.shape)
            m.weight.data[r_mask==0] = o_m.weight[r_mask==0]


def freeze_classifier(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.requires_grad = False
            m.bias.requires_grad = False

                
def set_non_target_block_gradient_zero(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and (m.mask.sum() > 0):
            mask = m.mask
                # print(mask)
            r_mask = mask.reshape(m.weight.shape)
            m.weight.grad *= r_mask


# def replace_target_weight_blocks(model, w):
#     updated_weights = []
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d) and (m.mask.sum() > 0):
#             mask = m.mask
#             r_mask = mask.reshape(m.weight.shape)
#             weight = m.weight.data.reshape(-1)
            
#             updated_weight = weight[mask==1]
#             updated_weights.append(updated_weight)
            
            
#     updated_weights = torch.cat(updated_weights)
            
#     # print(updated_weights.shape[0])
    
#     for i in range(0, len(updated_weights), 128):
#         v = updated_weights[i:i+128].unsqueeze(0)
#         scores = v.matmul(w.T)
#         idx = scores.argmax()
#         updated_weights[i:i+128] = w.T[:,idx]
    
    
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d) and (m.mask.sum() > 0):
#             mask = m.mask
#             r_mask = mask.reshape(m.weight.shape)
#             weight = m.weight.data.reshape(-1)
            
#             updated_weight = updated_weights[:mask.sum()]
#             updated_weights = updated_weights[mask.sum():]
            
#             weight[mask==1] = updated_weight
#             m.weight.data = weight.reshape(m.weight.shape)

def replace_target_weight_blocks(model, original_model, w):
    updated_weights = []
    original_weights = []
    for m, orig_m in zip(model.modules(), original_model.modules()):
        if isinstance(m, nn.Conv2d) and (m.mask.sum() > 0):
            mask = m.mask
            r_mask = mask.reshape(m.weight.shape)
            weight = m.weight.data.reshape(-1)
            
            updated_weight = weight[mask==1]
            updated_weights.append(updated_weight)

            original_weight = orig_m.weight.data.reshape(-1)
            original_weights.append(original_weight[mask==1])            
            
    updated_weights = torch.cat(updated_weights)
    original_weights = torch.cat(original_weights)
            
    # print(updated_weights.shape[0])
    
    for i in range(0, len(updated_weights), 128):
        v = updated_weights[i:i+128].unsqueeze(0)
        scores = v.matmul(w.T)
        # idx = scores.argmax()
        idxs = scores.topk(2).indices[0]
        # print(idxs)

        if torch.all(w.T[:, idxs[0]] == original_weights[i:i+128]):
            idx = idxs[1]
            # print('same pawa gese')
        else:
            idx = idxs[0]

        updated_weights[i:i+128] = w.T[:,idx]
    
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and (m.mask.sum() > 0):
            mask = m.mask
            r_mask = mask.reshape(m.weight.shape)
            weight = m.weight.data.reshape(-1)
            
            updated_weight = updated_weights[:mask.sum()]
            updated_weights = updated_weights[mask.sum():]
            
            weight[mask==1] = updated_weight
            m.weight.data = weight.reshape(m.weight.shape)
    
    # print(updated_weights.shape[0])
    
def convert_model(model, top_block_mask):
    ranked_mask = top_block_mask.clone()
    
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            vector_len = m.weight.data.reshape(-1).shape[0]
            m.mask = ranked_mask[:vector_len]
            ranked_mask = ranked_mask[vector_len:]
            
            if m.mask.sum() > 0:
                m.weight.requires_grad = True

                
    assert len(ranked_mask) == 0

def add_mask(model):
    for layer_idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            vector_len = m.weight.data.reshape(-1).shape[0]
            m.mask = torch.zeros(vector_len)
                

def see_layer_info(model):
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            layer_idx += 1
            print('requires_grad', m.weight.requires_grad)
            if m.weight.requires_grad:
                print(layer_idx, m.mask.sum())
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                print('classifier layer is being attacked')