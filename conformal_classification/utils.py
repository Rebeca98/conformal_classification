import numpy as np
# Torch
import torch
import torch.nn as nn
import torchvision
from typing import Tuple
import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm
import time
import pandas as pd

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu')
# device = 'cpu'
#device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu')


def sort_sum(scores:np.array)-> Tuple[np.array,np.array,np.array]:
    """_summary_

    Parameters
    ----------
    scores : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    I = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    return I, ordered, cumsum


def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(
        dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(
        temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    return data1, data2

def get_calib_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


# def get_model(model_name):
#     """ 
#     function that downloads the efficientnet pretrained weights
#     architetcture: str (e.g. 'EfficientNet')
#     model_name = 'efficientnet_b0'
#     pretrained: bool (e.g. True)
#     """
#     efficientnet_models = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']

#     if model_name in efficientnet_models:
#         if model_name == 'efficientnet_b0':
#             model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT',
#                                            progress=True)
#         elif model_name == 'efficientnet_b1':
#             model = models.efficientnet_b1(weights='EfficientNet_B1_Weights.DEFAULT',
#                                            progress=True)

#         elif model_name == 'efficientnet_b2':
#             model = models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT',
#                                            progress=True)
#         else:
#             raise NotImplementedError

#     else:
#         raise NotImplementedError

#     return model


def get_model(model_name):
    """ 
    Función que descarga los pesos pre-entrenados de EfficientNet
    model_name: str (ej. 'efficientnet_b0')
    Retorna el modelo cargado con los pesos pre-entrenados
    """
    if model_name == 'efficientnet_b0':
        return models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT',
                                           progress=True)
    elif model_name == 'efficientnet_b1':
        return models.efficientnet_b1(weights='EfficientNet_B1_Weights.DEFAULT',
                                           progress=True)
    elif model_name == 'efficientnet_b2':
        return models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT',
                                           progress=True)
    else:
        raise NotImplementedError("Modelo no soportado")


def build_model_for_cp(model_path,
                       model_name,
                       num_classes):
    """
    build model for inference
     model_name='efficientnet_b0'
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Usando GPU con Metal.")
    else:
        device = torch.device('cpu')
        print("GPU con Metal no está disponible, usando CPU.")

    pretrained_weights = torch.load(model_path, map_location=device)
    model = get_model(model_name)
    
    # Modificar el clasificador del modelo para que coincida con el número de clases
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=num_ftrs, out_features=num_classes)
    model.load_state_dict(pretrained_weights['model_state_dict'])
    model.to(device)
    model.eval()
    # model = torch.nn.DataParallel(model).to(device)
    return model


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


def validate(val_loader, model, print_bool=False):
    """
    val_loader: DataLoader with calibration data for validation (images,labels)
    model: ConformalModel
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
    return top1.avg, top5.avg, coverage.avg, size.avg  # return top1, top2, coverage, size


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


# Computes logits and targets from a model and loader


def get_logits_targets(model, loader, num_classes)-> torch.utils.data.TensorDataset:
    """


    output: torch.utils.data.TensorDataset
    """
    model.to(device) 
    model.eval()
    logits = torch.zeros((len(loader.dataset), num_classes), device=device)
    labels = torch.zeros((len(loader.dataset),),device=device)
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.to(device))  # .to(device)#.detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())
    return dataset_logits

# this function is used only in experiments


def get_logits_dataset(model_path:str, model_name:str, data_path:str, transform:torchvision.transforms.transforms.Compose, bsz:int, num_classes:int)-> torch.utils.data.TensorDataset:
    """
    datasetpath: calibration dataset path
    """
    # Else we will load our model, run it on the dataset, and save/return the output.
    model = build_model_for_cp(model_path=model_path,
                               model_name=model_name,
                               num_classes=num_classes)

    dataset = torchvision.datasets.ImageFolder(data_path, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=False, pin_memory=True)

    # Get the logits and targets
    dataset_logits = get_logits_targets(model, loader, num_classes)

    return dataset_logits


def get_wc_violation(cmodel, val_loader, strata, alpha)-> Tuple[float, pd.DataFrame]:
    """
    ?

    Parameters
    ----------
    cmodel : conformal_classification.conformal.ConformalModelLogits
        conformal model

    val_loader: torch.utils.data.DataLoader
    strata: 
    alpha:
    Returns
    -------
    Dataset
        Instance of the created Dataset
    """
    # df = pd.DataFrame(columns=['size', 'correct'])
    dfs_size_correctness = []
    for logit, target in val_loader:
        # compute output
        # This is a 'dummy model' which takes logits, for efficiency.
        _, S = cmodel(logit)
        # measure accuracy and record loss
        size = np.array([x.size for x in S])  # vector of Prediction set sizes
        I, _, _ = sort_sum(logit.cpu().numpy())
        correct = np.zeros_like(size)  # the same size as the vector of sizes
        for j in range(correct.shape[0]):
            # for each set calculate the correctness
            correct[j] = int(target[j] in list(S[j]))  # 1
        batch_df = pd.DataFrame.from_dict({'size': size, 'correct': correct})
        dfs_size_correctness.append(batch_df)
    df_size_correctness = pd.concat(dfs_size_correctness)
    wc_violation = 0
    for stratum in strata:
        temp_df = df_size_correctness[(df_size_correctness['size'] >= stratum[0]) & (df_size_correctness['size'] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean()-(1-alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation # the violation