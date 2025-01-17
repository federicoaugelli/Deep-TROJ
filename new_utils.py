import torch
from tqdm import tqdm
from utils import AverageMeter, hamming_distance
#from model import quan_Linear, quan_Conv2d


@torch.no_grad()
def validate(val_loader, model, criterion, iters=-1):

    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    model.eval()
    pbar = tqdm(val_loader, total=len(val_loader))
    
    for i, (input, target) in enumerate(pbar):
        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)

        # measure accuracy and record loss
        prec1,prec5 = accuracy(output.data, target, topk=(1,5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        pbar.set_postfix({'top1': top1.avg, 'top5': top5.avg})
        
        if i == iters:
            break

    return top1.avg, top5.avg


def choose_index(idxs, g_idx):
    dists = []
    all_idxs = []
    for idx in idxs:
        if torch.all(idx == g_idx):
            continue
        dists.append(hamming_distance(idx, g_idx))
        all_idxs.append(idx)
        
    all_idxs = torch.tensor(all_idxs)
    dists = torch.tensor(dists)
    return all_idxs[dists.argmin()], dists.min().item()


class AverageMeter2(object):
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
