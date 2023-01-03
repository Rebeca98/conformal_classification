import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sort_sum(scores):
    I = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    return I, ordered, cumsum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        # Fixed-point notation. Displays the number as a fixed-point number. The default precision is 6.
        self.fmt = fmt
        self.reset()  # set all members to their initial value

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n  # number of items measured (n=batch_size)
        self.avg = self.sum / self.count

    def __str__(self):
        """
        This method returns the string representation of the object. 
        """
        # This method is called when print() or str() function is invoked on an object.
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def validate(val_loader, model, print_bool):
    """
    val_loader: DataLoader with calibration data for validation (images,labels)
    model: ConformalModel
    print_bool
    """
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.to(device)
            # compute output
            # S: prediction set
            # compute plat scaling and conformal pred. algorithm
            output, S = model(x.to(device))
            # measure accuracy and record loss
            # precision for a k-sets k=1 and k=5, prec1 y prec4 are scalars
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)

            # Update meters
            # add up the total of correct predictions to obtain the avg
            # x.shape[0] = batch_size
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            # coverage of all validation's dataset prediction sets(avg)
            coverage.update(cvg, n=x.shape[0])
            # size's of all validation's dataset prediction sets
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rValidation size: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('')  # Endline
    # podemos usar el output de estas metricas para hacer estadisticas (ver la distirbucion de estas metricas)
    return top1.avg, top5.avg, coverage.avg, size.avg


def coverage_size(S, targets):
    """
    This function return the coverage which is the proportion of "covered" predection sets, in other words, 
    the number of sets S that contain the true label, and the average of the predictions sets sizes 
    """
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    # target.shape[0] = batch_size
    return float(covered)/targets.shape[0], size/targets.shape[0]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: logits of the Neural network model
        e.g. for one element of the batch and 20 classes:  
                                [[-2.6332984 -2.7642944 -2.3948205 -4.4933996 -4.251026  -3.2615488
                                    -2.8112233 -3.851386  -3.2255957 -3.7642008 -5.852391  -4.285244
                                    -4.342262  -3.9740272]]
    target: true label from model
        e.g. for a batch_size=2 target: torch.tensor([13,2])
    Return
    _____
    porportion of correct predictions*100 =percentage of correct predictions

    """
    maxk = max(topk)
    batch_size = target.size(0)
    # Returns the k largest elements of the given input tensor along a given dimension.

    _, pred = output.topk(maxk, 1, True, True)  # we save only indices
    # torch.tensor([[ 8,  2,  0, 11,  9],[ 8,  2,  0, 11,  9]]).t()  ,k=5 for two images
    pred = pred.t()
    # check equality
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        # percentage of correctness
        # porportion of correct predictions*100 =percentage of correct predictions
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# cambiar esto por el get model para inferencia

def get_model(modelname='efficientnet_b0', pretrained=True):
    if modelname == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT',
                                       pretrained=pretrained, progress=True)
        # weights=EfficientNet_B0_Weights.IMAGENET1K_V1

    elif modelname == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='EfficientNet_B1_Weights',
                                       pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b3':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b4':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b5':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b6':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b7':
        model = models.efficientnet_b3(pretrained=pretrained, progress=True)

    else:
        raise NotImplementedError
    #model = torch.nn.DataParallel(model)
    return model


def build_model_for_cp(model_path, modelname='efficientnet_b0', num_classes=20, pretrained=True):
    """
    build model for training
    """
    pretrained_weights = torch.load(model_path)  # we load our trained weights
    model = get_model(modelname, pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=num_ftrs, out_features=num_classes)
    model.load_state_dict(pretrained_weights['model_state_dict'])
    model.eval()
    model = torch.nn.DataParallel(model).to(device)
    return model

# Computes logits and targets from a model and loader


def get_logits_targets(model, loader, num_classes=20):
    # 1000 classes in Imagenet.
    logits = torch.zeros((len(loader.dataset), num_classes))
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.to(device)).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())
    return dataset_logits

# this function is used only in experiments


def get_logits_dataset(model_path, modelname, datasetpath, transform, bsz, num_classes=20, pretrained=True):
    """
    datasetpath: calibration dataset path
    """
    # Else we will load our model, run it on the dataset, and save/return the output.
    model = build_model_for_cp(
        model_path, modelname, num_classes, pretrained).to(device)

    dataset = torchvision.datasets.ImageFolder(datasetpath, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=False, pin_memory=True)

    # Get the logits and targets
    dataset_logits = get_logits_targets(model, loader)

    return dataset_logits


def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).cuda()
    targets = torch.cat([torch.Tensor([int(x[1])])
                        for x in data], dim=0).long()
    return imgs, targets


def split2ImageFolder(path, transform, n1, n2):
    dataset = torchvision.datasets.ImageFolder(path, transform)
    data1, data2 = torch.utils.data.random_split(
        dataset, [n1, len(dataset)-n1])
    data2, _ = torch.utils.data.random_split(data2, [n2, len(dataset)-n1-n2])
    return data1, data2


def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(
        dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(
        temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    return data1, data2
