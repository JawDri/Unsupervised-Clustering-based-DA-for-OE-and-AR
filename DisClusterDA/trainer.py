import time
import torch
import os
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
    

def train_compute_class_mean(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, criterion, criterion_afem, optimizer, itern, current_epoch, cs_1, ct_1, cs_2, ct_2, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_s = AverageMeter()
    losses = AverageMeter()
    
    model.train() # turn to training mode
    
    lam = 2 / (1 + math.exp(-1 * 10 * current_epoch / args.epochs)) - 1 # penalty parameter

    end = time.time()

    # prepare data for model forward and backward
    try:
        (input_source, target_source) = train_loader_source_batch.__next__()[1]
    except StopIteration:
        train_loader_source_batch = enumerate(train_loader_source)
        (input_source, target_source) = train_loader_source_batch.__next__()[1]

    try:
        (input_target, target_target_not_use) = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        (input_target, target_target_not_use) = train_loader_target_batch.__next__()[1]

    target_source = target_source.cuda(non_blocking=True)
    input_source_var = Variable(input_source)
    target_source_var = Variable(target_source)
    input_target_var = Variable(input_target)
    cs_target_var = Variable(torch.arange(0, args.num_classes).cuda(non_blocking=True))
    ct_target_var = Variable(torch.arange(0, args.num_classes).cuda(non_blocking=True))

    data_time.update(time.time() - end)

    # model forward for source/target data
    feat1_s, feat2_s, pred_s = model(input_source_var)
    feat1_t, feat2_t, pred_t = model(input_target_var)
    
    # compute source and target centroids on respective batches at the current iteration
    prob_t = F.softmax(pred_t - pred_t.max(1, True)[0], dim=1)
    idx_max_prob = prob_t.topk(1, 1, True, True)[-1]
    cs_1_temp = Variable(torch.cuda.FloatTensor(cs_1.size()).fill_(0))
    cs_count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    ct_1_temp = Variable(torch.cuda.FloatTensor(ct_1.size()).fill_(0))
    ct_count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    cs_2_temp = Variable(torch.cuda.FloatTensor(cs_2.size()).fill_(0))
    ct_2_temp = Variable(torch.cuda.FloatTensor(ct_2.size()).fill_(0))
    for i in range(input_source_var.size(0)):
        cs_1_temp[target_source[i]] += feat1_s[i]
        cs_count[target_source[i]] += 1
        cs_2_temp[target_source[i]] += feat2_s[i]
        ct_1_temp[idx_max_prob[i]] += feat1_t[i]
        ct_count[idx_max_prob[i]] += 1
        ct_2_temp[idx_max_prob[i]] += feat2_t[i]
    
    # exponential moving average centroids
    cs_1 = Variable(cs_1.data.clone())
    ct_1 = Variable(ct_1.data.clone())
    cs_2 = Variable(cs_2.data.clone())
    ct_2 = Variable(ct_2.data.clone())
    mask_s = ((cs_1.data != 0).sum(1, keepdim=True) != 0).float() * args.remain
    mask_t = ((ct_1.data != 0).sum(1, keepdim=True) != 0).float() * args.remain
    mask_s[cs_count == 0] = 1.0
    mask_t[ct_count == 0] = 1.0
    cs_count[cs_count == 0] = args.eps
    ct_count[ct_count == 0] = args.eps
    cs_1 = mask_s * cs_1 + (1 - mask_s) * (cs_1_temp / cs_count)
    ct_1 = mask_t * ct_1 + (1 - mask_t) * (ct_1_temp / ct_count)
    cs_2 = mask_s * cs_2 + (1 - mask_s) * (cs_2_temp / cs_count)
    ct_2 = mask_t * ct_2 + (1 - mask_t) * (ct_2_temp / ct_count)

    # centroid forward
    pred_cs_1 = model.module.fc2(model.module.fc1(cs_1))
    pred_ct_1 = model.module.fc2(model.module.fc1(ct_1))
    pred_cs_2 = model.module.fc2(cs_2)
    pred_ct_2 = model.module.fc2(ct_2)
    
    # compute instance-to-centroid distances
    dist_fs_cs_1 = (feat1_s.unsqueeze(1) - cs_1.unsqueeze(0)).pow(2).sum(2)
    sim_fs_cs_1 = F.softmax(-1 * dist_fs_cs_1, dim=1)
    dist_fs_cs_2 = (feat2_s.unsqueeze(1) - cs_2.unsqueeze(0)).pow(2).sum(2)
    sim_fs_cs_2 = F.softmax(-1 * dist_fs_cs_2, dim=1)
    dist_ft_ct_1 = (feat1_t.unsqueeze(1) - ct_1.unsqueeze(0)).pow(2).sum(2)
    sim_ft_ct_1 = F.softmax(-1 * dist_ft_ct_1, dim=1)
    dist_ft_ct_2 = (feat2_t.unsqueeze(1) - ct_2.unsqueeze(0)).pow(2).sum(2)
    sim_ft_ct_2 = F.softmax(-1 * dist_ft_ct_2, dim=1)
    
    # compute centroid-to-centroid distances
    dist_cs_cs_1 = (cs_1.unsqueeze(1) - cs_1.unsqueeze(0)).pow(2).sum(2)
    dist_cs_cs_2 = (cs_2.unsqueeze(1) - cs_2.unsqueeze(0)).pow(2).sum(2)
    dist_ct_ct_1 = (ct_1.unsqueeze(1) - ct_1.unsqueeze(0)).pow(2).sum(2)
    dist_ct_ct_2 = (ct_2.unsqueeze(1) - ct_2.unsqueeze(0)).pow(2).sum(2)
    
    loss_p1 = (criterion(pred_cs_1 / args.temperature, cs_target_var) + criterion(pred_ct_1 / args.temperature, ct_target_var) + 
               criterion(sim_fs_cs_1, target_source_var) + criterion(-1 * dist_cs_cs_1 / args.temperature, cs_target_var) + #-1 * dist_fs_cs_1
               criterion_afem(sim_ft_ct_1) + criterion(-1 * dist_ct_ct_1 / args.temperature, ct_target_var))
    loss_p2 = (criterion(pred_cs_2 / args.temperature, cs_target_var) + criterion(pred_ct_2 / args.temperature, ct_target_var) + 
               criterion(sim_fs_cs_2, target_source_var) + criterion(-1 * dist_cs_cs_2 / args.temperature, cs_target_var) + #-1 * dist_fs_cs_2
               criterion_afem(sim_ft_ct_2) + criterion(-1 * dist_ct_ct_2 / args.temperature, ct_target_var))
    loss =  criterion(pred_s, target_source_var) + lam * criterion_afem(prob_t) +\
            lam * (loss_p1 + 0.2 * lam * loss_p2)

    # record losses and accuracies on source data
    losses.update(loss.item(), input_source.size(0))
    prec1_s = accuracy(pred_s.data, target_source, args.num_classes)
    top1_s.update(prec1_s.item(), input_source.size(0))
    
    model.zero_grad()
    loss.backward()
    optimizer.step()
    model.zero_grad()

    batch_time.update(time.time() - end)
    if itern % args.print_freq == 0:
        display = 'Train - epoch [{0}/{1}]({2})\tBT {batch_time.avg:.3f}\tDT {data_time.avg:.3f}\tTop@1_s {top1.avg:.3f}\tLoss {loss.avg:.4f}'\
        .format(current_epoch, args.epochs, itern, batch_time=batch_time, data_time=data_time, top1=top1_s, loss=losses)
        print(display)
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n' + display.replace('\t', ', '))
        log.close()
    
    return train_loader_source_batch, train_loader_target_batch, cs_1, ct_1, cs_2, ct_2


def evaluate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topfscore = {'clsfscore': AverageMeter()}
    mcp = MeanClassPrecision(args.num_classes)
    
    # switch to evaluation mode
    model.eval()
    
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)[-1]
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, args.num_classes)
        prec1fscore = faccuracy(output, target, args.num_classes)
        top1.update(prec1.item(), input.size(0))
        topfscore['clsfscore'].update(prec1fscore, input.size(0))
        mcp.update(output.data, target)
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Evaluate on target - [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'topfscore {topfscore.val:.3f} ({topfscore.avg:.3f})'
                  .format(epoch, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, topfscore = topfscore['clsfscore']))
    
    print(' * Evaluate on target - prec@1: {top1.avg:.3f}'.format(top1=top1))
    print(' * Classifier: topfscore {cls.avg:.3f}'.format(cls=topfscore['clsfscore']))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n             Evaluate on target - epoch: %d, loss: %.4f, acc: %.3f" % (epoch, losses.avg, top1.avg))
    
    if args.src.find('visda') != -1:
        print(str(mcp))
        log.write('\n             ' + str(mcp))
        log.close()
        return mcp.mean_class_prec
    else:
        log.close()
        return top1.avg


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





def accuracy(output, target, num_classes):
    """Computes the precision@k for the specified values of k"""
    
    _, pred = output.topk(1, 1, True, True)
    #print(pred[0])
    pred = pred.t().cpu()
    #print(pred[0])
    #correct = pred.eq(target.view(1, -1).expand_as(pred).cpu())
    #print(target.view(1, -1).expand_as(pred).cpu())
    #fscore = f1_score(target.view(1, -1).expand_as(pred).cpu(), pred, average='weighted')
    #accu = Accuracy(task="multiclass", num_classes=num_classes)
    accurac = accuracy_score(target.cpu(), pred[0])
    #accurac = accu(pred[0], target.cpu())
    #print(accurac, 'ACCURACY')
    
    return accurac*100

def faccuracy(output, target,num_classes):
    """Computes the precision@k for the specified values of k"""

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().cpu()
    #correct = pred.eq(target.view(1, -1).expand_as(pred).cpu())
    #print(target.view(1, -1).expand_as(pred).cpu())
    #fscore = f1_score(target.view(1, -1).expand_as(pred).cpu(), pred, average='weighted')
    #f1 = F1Score(task="multiclass", num_classes=num_classes)
    #fscore = f1(pred[0], target.cpu())
    fscore = f1_score(target.cpu(), pred[0], average='weighted')

    #print(fscoretest, 'FSCORE')
    
   
    return fscore*100



class MeanClassPrecision(object):
    """Computes and stores the mean class precision"""
    def __init__(self, num_classes, fmt=':.3f'):
        self.num_classes = num_classes
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.total_vector = torch.zeros(self.num_classes)
        self.correct_vector = torch.zeros(self.num_classes)
        self.per_class_prec = torch.zeros(self.num_classes)
        self.mean_class_prec = 0

    def update(self, output, target):
        pred = output.max(1)[1]
        correct = pred.eq(target).float().cpu()
        for i in range(target.size(0)):
            self.total_vector[target[i]] += 1
            self.correct_vector[target[i]] += correct[i]
        temp = torch.zeros(self.total_vector.size())
        temp[self.total_vector == 0] = 1e-6
        self.per_class_prec = self.correct_vector / (self.total_vector + temp) * 100
        self.mean_class_prec = self.per_class_prec.mean().item()
    
    def __str__(self):
        fmtstr = 'per-class prec: ' + '|'.join([str(i) for i in list(np.around(np.array(self.per_class_prec), int(self.fmt[-2])))])
        fmtstr = 'Mean class prec: {mean_class_prec' + self.fmt + '}, ' + fmtstr
        return fmtstr.format(**self.__dict__)
    

