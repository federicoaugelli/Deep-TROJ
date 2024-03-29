import torch
import torch.nn as nn
import torch.nn.functional as F

class TrojanAttackLoss(nn.Module):
    def __init__(self, w, mean, std, alpha=1.0, beta=1.0, device='cuda'):
        super(TrojanAttackLoss, self).__init__()
        self.w = w
        self.max_normalized = torch.tensor([(1 - m) / s for m, s in zip(mean, std)]).to(device)
        self.min_normalized = torch.tensor([(0 - m) / s for m, s in zip(mean, std)]).to(device)
        self.alpha = alpha
        self.beta = beta
    

    def loss_constraint(self, model):
        updated_weights = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and (m.mask.sum() > 0):
                mask = m.mask
                r_mask = mask.reshape(m.weight.shape)
                weight = m.weight.data.reshape(-1)
                
                updated_weight = weight[mask==1]
                updated_weights.append(updated_weight)
                
                
        updated_weights = torch.cat(updated_weights).reshape(-1, 128)
        updated_weights = F.normalize(updated_weights, dim=1)

        scores = updated_weights.matmul(self.w)

        loss_blocks = torch.abs(1.0 - scores.max(1).values).mean()
                
        return loss_blocks

    def loss_trigger(self, trigger):
        max_values = trigger.view(trigger.size(1), -1).max(1).values
        min_values = trigger.view(trigger.size(1), -1).min(1).values
        return F.mse_loss(max_values, self.max_normalized) + F.mse_loss(min_values, self.min_normalized)


    def forward(self, pred, target, model, trigger):
        loss_ce = F.cross_entropy(pred, target)  
        loss_blocks = self.loss_constraint(model)
        loss_trigger = self.loss_trigger(trigger)
        loss = loss_ce + self.alpha * loss_blocks + self.beta * loss_trigger
        return loss