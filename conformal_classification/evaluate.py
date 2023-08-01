import pandas as pd
from tqdm import tqdm
import itertools
import random
import os
# Torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms as tf
import torch.utils.data as tdata
import torch
from collections import defaultdict
#import matplotlib.pyplot as plt
import numpy as np
#from utils import *
# scipy
from scipy.stats import median_abs_deviation as mad
#from src.models import build_model_inference
from conformal_classification.conformal import ConformalModelLogits
from conformal_classification.conformal import get_violation

from conformal_classification.utils import sort_sum
#from conformal_classification.utils import split2
from conformal_classification.utils import split2

from conformal_classification.utils_experiments import get_calib_transform
from conformal_classification.utils_experiments import build_model_for_cp
from conformal_classification.utils_experiments import validate
from conformal_classification.utils_experiments import get_logits_dataset
device = ('cuda' if torch.cuda.is_available() else 'cpu')


def get_logits_model(model_path,model_name,data_path,transform, bsz,num_classes,pretrained):
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    destination_folder : str
        The folder path where the dataset will be stored.
    json_file : str
        Path of a json file that will be converted into a Dataset.
    categories : list of str, optional
        List of categories to filter registers (default is [])
    **kwargs :
        Extra named arguments passed to build_json

    Returns
    -------
    Dataset
        Instance of the created Dataset
    """
    # Data Loading
    logits = get_logits_dataset(model_path = model_path,
                                model_name = model_name, 
                                data_path = data_path, 
                                transform = transform, 
                                bsz = bsz, 
                                num_classes = num_classes, 
                                pretrained = pretrained)
    
    

    # Instantiate and wrap model
    model = build_model_for_cp(model_path=model_path,
                               architecture='EfficientNet', 
                               model_name = model_name,
                               num_classes = num_classes, 
                               pretrained = pretrained).to(device)
    
    return logits, model
    
    
def sizes_topk(model,logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor, pretrained=True):
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    destination_folder : str
        The folder path where the dataset will be stored.
    json_file : str
        Path of a json file that will be converted into a Dataset.
    categories : list of str, optional
        List of categories to filter registers (default is [])
    **kwargs :
        Extra named arguments passed to build_json

    Returns
    -------
    Dataset
        Instance of the created Dataset
    """
    # Experiment logic
    naive_bool = predictor == 'Naive'
    lamda_predictor = lamda
    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0  # No regularization.

    # A new random split for every trial
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_conf)
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(
        logits_val, batch_size=bsz, shuffle=False, pin_memory=True)
    
    # Conformalize the model
    conformal_model = ConformalModelLogits(model, 
                                           loader_cal, 
                                           alpha=alpha, 
                                           kreg=kreg,
                                           lamda=lamda_predictor, 
                                           randomized=randomized, 
                                           allow_zero_sets=True, 
                                           naive=naive_bool)

    topk_dict = defaultdict(list)

    for i, (logit, target) in tqdm(enumerate(loader_val)):
        # compute output
        output, S = conformal_model(logit)
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy())
        topk = np.where((I - target.view(-1, 1).numpy()) == 0)[1]+1
        topk_dict[f'{target}-topk'].append(topk)
        topk_dict[f'{target}-size'].append(size)

    return topk, size, topk_dict


def evaluate(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, naive_bool, fixed_bool):
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    destination_folder : str
        The folder path where the dataset will be stored.
    json_file : str
        Path of a json file that will be converted into a Dataset.
    categories : list of str, optional
        List of categories to filter registers (default is [])
    **kwargs :
        Extra named arguments passed to build_json

    Returns
    -------
    Dataset
        Instance of the created Dataset
    """
    """
    Report the median of top1 and top5 averages 

    Args:
        model (str): path to the trained model weights
        architecture (str): neural network architecture used for the classification model
        calibration_dataset_path (str): path to calibration dataset 
        num_classes (int): number of classes refers to the distinct categories or labels that the model aims to predict
    
    Returns:
        tuple: the values inlcude the median of top1's, top5's, coverages and sizes, 
        median_abs_deviation of the top1s, top5s, coverages and sizes
    """
    # A new random split for every trial
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(
        logits_val, batch_size=bsz, shuffle=False, pin_memory=True)
    
    if fixed_bool:
        # The full prediction for the fixed procedure is handled in here.
        gt_locs_cal = np.array(
            [np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in logits_cal])
        gt_locs_val = np.array(
            [np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in logits_val])
        kstar = np.quantile(gt_locs_cal, 1-alpha, interpolation='higher') + 1
        rand_frac = ((gt_locs_cal <= (kstar-1)).mean() - (1-alpha)) / \
            ((gt_locs_cal <= (kstar-1)).mean()-(gt_locs_cal <= (kstar-2)).mean())
        # kstar is in size units (0 indexing)
        sizes = np.ones_like(gt_locs_val) * (kstar-1)
        sizes = sizes + (torch.rand(gt_locs_val.shape)
                         > rand_frac).int().numpy()
        top1_avg = (gt_locs_val == 0).mean()
        top5_avg = (gt_locs_val <= 4).mean()
        cvg_avg = (gt_locs_val <= (sizes-1)).mean()
        sz_avg = sizes.mean()
    else:
        # Conformalize the model
        conformal_model = ConformalModelLogits(model = model, 
                                               calib_loader = loader_cal, 
                                               alpha=alpha, 
                                               kreg=kreg, 
                                               lamda=lamda, 
                                               randomized=randomized,
                                               allow_zero_sets=True, 
                                               pct_paramtune=pct_paramtune, 
                                               naive=naive_bool, 
                                               batch_size=bsz, 
                                               lamda_criterion='size')
        # Collect results
        top1_avg, top5_avg, cvg_avg, sz_avg,cvgs = validate(
            loader_val, conformal_model, print_bool=False)
    return top1_avg, top5_avg, cvg_avg, sz_avg,cvgs



def evaluate_models(model, logits, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor):
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    destination_folder : str
        The folder path where the dataset will be stored.
    json_file : str
        Path of a json file that will be converted into a Dataset.
    categories : list of str, optional
        List of categories to filter registers (default is [])
    **kwargs :
        Extra named arguments passed to build_json

    Returns
    -------
    Dataset
        Instance of the created Dataset
    """
    
    """
    Report the median of top1 and top5 averages 

    Args:
        model_path (str): path to the trained model weights
        architecture (str): neural network architecture used for the classification model
        calibration_dataset_path (str): path to calibration dataset 
        num_classes (int): number of classes refers to the distinct categories or labels that the model aims to predict
    
    Returns:
        tuple: the values inlcude the median of top1's, top5's, coverages and sizes, 
        median_abs_deviation of the top1s, top5s, coverages and sizes
    """
    # Experiment logic
    naive_bool = predictor == 'Naive'
    fixed_bool = predictor == 'Fixed'
    if predictor in ['Fixed', 'Naive', 'APS']:
        kreg = 1
        lamda = 0  # No regularization.

    cvgs_list = []
    # Perform experiment
    top1s = np.zeros((num_trials,))
    top5s = np.zeros((num_trials,))
    coverages = np.zeros((num_trials,))
    sizes = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        top1_avg, top5_avg, cvg_avg, sz_avg, cvgs = evaluate(
            model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, naive_bool, fixed_bool)
        cvgs_list.append(cvgs)
        top1s[i] = top1_avg
        top5s[i] = top5_avg
        coverages[i] = cvg_avg
        sizes[i] = sz_avg
        print(
            f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(top1s), np.median(top5s), np.median(coverages), np.median(sizes), mad(top1s), mad(top5s), mad(coverages), mad(sizes),cvgs_list


def get_worst_violation(model, logits, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, pretrained, lamda_criterion='size'):
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    destination_folder : str
        The folder path where the dataset will be stored.
    json_file : str
        Path of a json file that will be converted into a Dataset.
    categories : list of str, optional
        List of categories to filter registers (default is [])
    **kwargs :
        Extra named arguments passed to build_json

    Returns
    -------
    Dataset
        Instance of the created Dataset
    """
    
    calib_logits, val_logits = tdata.random_split(logits, [n_data_conf, len(
        logits)-n_data_conf])  # A new random split for every trial
    # Prepare the loaders
    calib_loader = torch.utils.data.DataLoader(
        calib_logits, batch_size=bsz, shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_logits, batch_size=bsz, shuffle=False, pin_memory=True)

    # Conformalize the model with the APS parameter choice
    conformal_model = ConformalModelLogits(model = model, 
                                           calib_loader = calib_loader, 
                                           alpha=alpha, 
                                           kreg=0, 
                                           lamda=0, 
                                           randomized=randomized, 
                                           allow_zero_sets=True, 
                                           pct_paramtune=None,
                                           naive=False,
                                           batch_size=bsz,
                                           lamda_criterion='size', 
                                           strata=strata)
    
    aps_worst_violation,df_viol_aps = get_violation(
        conformal_model, val_loader, strata, alpha)
    
    # Conformalize the model with the RAPS parameter choice
    conformal_model = ConformalModelLogits(model = model, 
                                           calib_loader = calib_loader, 
                                           alpha=alpha, 
                                           kreg=None, 
                                           lamda=None, 
                                           randomized=randomized,
                                           allow_zero_sets=True,
                                           pct_paramtune=pct_paramtune,
                                           naive=False,
                                           batch_size=bsz,
                                           lamda_criterion='adaptiveness',
                                           strata=strata)
    
    raps_worst_violation,df_viol_raps = get_violation(
        conformal_model, val_loader, strata, alpha)

    return aps_worst_violation, raps_worst_violation