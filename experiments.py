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
#import matplotlib.pyplot as plt
import numpy as np
#from utils import *
# scipy
from scipy.stats import median_abs_deviation as mad
#from src.models import build_model_inference
from conformal import ConformalModelLogits
from conformal import get_violation
from utils import validate, get_logits_dataset
from utils import sort_sum
from utils import split2
from utils_experiments import get_calib_transform, build_model_for_cp

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def experiment1_trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, naive_bool):
    """
    model: NN classifier
    logits:
    alpha:
    kreg:
    lambda:
    randomized:
    n_data_conf
    n_data_val
    bsz:
    naive_bool
    """
    # A new random split for every trial
    # one conformal calibration subset and one evaluation subset
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(
        logits_val, batch_size=bsz, shuffle=False, pin_memory=True)
    # Conformalize the model
    cmodel = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg,
                                  lamda=lamda, randomized=randomized, allow_zero_sets=True, naive=naive_bool)
    # Collect results
    top1_avg, top5_avg, cvg_avg, sz_avg = validate(
        loader_val, cmodel, print_bool=True)
    return top1_avg, top5_avg, cvg_avg, sz_avg


def run_trials_experiment1(modelname, model, logits, num_trials, alpha, kreg,
                           lamda, randomized, n_data_conf, n_data_val, bsz, predictor):
    """
    modelname: name of the model we are going to use  (e.g. efficientnet_b0)
    datasetname:
    datset_path:
    model: conformalModelLogits
    logits:
    num_trials: number of 
    alpha: 
    kreg:
    lambda
    randomized,
    n_data_conf, 
    n_data_val, 
    bsz: batch_size
    predictor: type of Conformal predictions algorithm (naive, APS and RAPS)
    """
    # Experiment logic
    naive_bool = predictor == 'Naive'
    if predictor in ['Naive', 'APS']:
        lamda = 0  # No regularization.

    # Perform experiment
    dfs = []
    for i in tqdm(range(num_trials)):
        top1_avg, top5_avg, cvg_avg, sz_avg = experiment1_trial(
            model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, naive_bool)
        dfs.append(pd.DataFrame.from_dict({"model": [modelname],
                                           "predictor": [predictor],
                                           "alpha": [alpha],
                                           "kreg": [kreg],
                                           "lamda": [lamda],
                                           "coverage": [cvg_avg],
                                           "size": [sz_avg]}))

    df = pd.concat(dfs)
    return df


def save_experiment1_trials(model_info_file, datapath, alphas, predictors, num_trials, kregs,
                            lamdas, randomized, total_conf, pct_data, bsz,
                            image_size, num_classes=14, pretrained=True):
    """
    Report the coverage and mean set size of each procedure for (two) different choices of alpha.
    model_info_path: dict
    datapath: strin
        path to the data used in calibration experiments
    modelname: string 
        model name used in the json file
    alphas = [0.01, 0.05, 0.10]
    predictors = ['Naive', 'APS', 'RAPS']
    """
    params = list(itertools.product(alphas, predictors))
    m = len(params)
    n_data_conf = int(total_conf*pct_data)
    n_data_val = int(total_conf*pct_data)
    cudnn.benchmark = True

    modelnames = model_info_file['models'].keys()
    modelnames_arch = [(model_info_file['models'][name]['file'], model_info_file['models'][name]['path'],
                        model_info_file['models'][name]['architecture']) for name in modelnames]

    #modelfile = model_info_file['models'][modelname]['file']
    #architecture = model_info_file['models'][modelname]['architecture']
    #modelpath = model_info_file['models'][modelname]['path']
    transform = get_calib_transform(image_size)
    params = list(itertools.product(
        modelnames_arch, alphas, predictors, lamdas, kregs))
    m = len(params)
    transform = get_calib_transform(image_size)
    # Perform the experiment
    dfs = []
    for i in range(m):
        modelinfo, alpha, predictor, lamda, kreg = params[i]
        _modelname = modelinfo[0].replace('.pth', '')
        print(
            f'Model: {_modelname} | Desired coverage: {1-alpha} | Predictor: {predictor} | Lambda = {lamda}')
        model = build_model_for_cp(os.path.join(modelinfo[1], modelinfo[0]), modelinfo[2],
                                   num_classes, pretrained).to(device)
        logits = get_logits_dataset(os.path.join(modelinfo[1], modelinfo[0]),
                                    modelinfo[2], datapath, transform, bsz, num_classes, pretrained)
        out = run_trials_experiment1(_modelname, model, logits, num_trials, alpha, kreg,
                                     lamda, randomized, n_data_conf, n_data_val, bsz, predictor)

        dfs.append(out)
    df = pd.concat(dfs)
    return df


# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.
# experiment 2
def sizes_topk(modelpath, modelname, datasetpath, num_classes,
               transform, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor, pretrained=True):
    # Experiment logic
    naive_bool = predictor == 'Naive'
    lamda_predictor = lamda
    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0  # No regularization.

    # Data Loading
    logits = get_logits_dataset(modelpath,
                                modelname, datasetpath, transform, bsz, num_classes, pretrained)
    # A new random split for every trial
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(
        logits_val, batch_size=bsz, shuffle=False, pin_memory=True)

    # Instantiate and wrap model
    #model = get_model(modelname)
    model = build_model_for_cp(modelpath, modelname,
                               num_classes, pretrained).to(device)
    # Conformalize the model
    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg,
                                           lamda=lamda_predictor, randomized=randomized, allow_zero_sets=True, naive=naive_bool)

    #df = pd.DataFrame(columns=['model', 'predictor', 'size', 'topk', 'lamda'])
    #dfs = []
    # Perform experiment
    for i, (logit, target) in tqdm(enumerate(loader_val)):
        # compute output
        # This is a 'dummy model' which takes logits, for efficiency.
        output, S = conformal_model(logit)
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy())
        topk = np.where((I - target.view(-1, 1).numpy()) == 0)[1]+1

    #df = pd.concat(dfs)
    return topk, size


def save_experiment2(model_info_file, datapath, num_classes, alphas, predictors, lambdas, kregs,
                     randomized, total_conf, pct_cal, pct_val, bsz,
                     image_size, pretrained=True):
    """
    This experiment illustrates the tradeoff between the set size and adaptiveness.

    model_info: json
    modelnames: iterable (object that can be iterated)
        This iterable contain models we want to evaluate. This names must corresponde to the json file used for conformal testing
        e.g. ['model-1','model-2']
    lambdas: list
        list of lambdas
    """
    n_data_conf = int(total_conf*pct_cal)
    n_data_val = int(total_conf*pct_val)
    cudnn.benchmark = True

    modelnames = model_info_file['models'].keys()
    modelnames_arch = [(model_info_file['models'][name]['file'], model_info_file['models'][name]['path'],
                        model_info_file['models'][name]['architecture']) for name in modelnames]
    transform = get_calib_transform(image_size)
    params = list(itertools.product(
        modelnames_arch, alphas, predictors, lambdas, kregs))
    m = len(params)
    dfs = []
    for i in range(m):
        modelinfo, alpha, predictor, lamda, kreg = params[i]
        _modelname = modelinfo[0].replace('.pth', '')
        print(
            f'Model: {_modelname} | Desired coverage: {1-alpha} | Predictor: {predictor} | Lambda = {lamda}')
        topk, size = sizes_topk(os.path.join(modelinfo[1], modelinfo[0]), modelinfo[2], datapath, num_classes, transform,
                                alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor, pretrained)
        dfs.append(pd.DataFrame.from_dict({'model': _modelname,
                                           'predictor': predictor,
                                           'size': size,
                                           'topk': topk,
                                           'lamda': lamda,
                                           'kreg': kreg
                                           }))

    df = pd.concat(dfs)
    return df


# experiment 3
# report coverage and size of the optimizal, randomized fixed sets naive, APS and RAPS


def experiment3_trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, naive_bool, fixed_bool):
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
        conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda, randomized=randomized,
                                               allow_zero_sets=True, pct_paramtune=pct_paramtune, naive=naive_bool, batch_size=bsz, lamda_criterion='size')
        # Collect results
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(
            loader_val, conformal_model, print_bool=False)
    return top1_avg, top5_avg, cvg_avg, sz_avg


def run_trials_experiment3(modelpath, modelname, datasetpath, num_classes, transform, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor):
    """
    Report the median of top1 and top5 averages 
    """
    # Experiment logic
    naive_bool = predictor == 'Naive'
    fixed_bool = predictor == 'Fixed'
    if predictor in ['Fixed', 'Naive', 'APS']:
        kreg = 1
        lamda = 0  # No regularization.

    # Data Loading
    #logits = get_logits_dataset(modelname, datasetname, datasetpath)
    logits = get_logits_dataset(modelpath,
                                modelname, datasetpath, transform, bsz, num_classes, pretrained=True)

    # Instantiate and wrap model
    #model = get_model(modelname)
    model = build_model_for_cp(modelpath, modelname,
                               num_classes, pretrained=True).to(device)

    # Perform experiment
    top1s = np.zeros((num_trials,))
    top5s = np.zeros((num_trials,))
    coverages = np.zeros((num_trials,))
    sizes = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        top1_avg, top5_avg, cvg_avg, sz_avg = experiment3_trial(
            model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, naive_bool, fixed_bool)
        top1s[i] = top1_avg
        top5s[i] = top5_avg
        coverages[i] = cvg_avg
        sizes[i] = sz_avg
        print(
            f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(top1s), np.median(top5s), np.median(coverages), np.median(sizes), mad(top1s), mad(top5s), mad(coverages), mad(sizes)


def save_experiment3_trials(model_info_file, datapath, num_classes, alphas, predictors, kregs, lamdas,
                            randomized, total_conf, pct_cal, pct_val, bsz,
                            image_size, num_trials, pct_paramtune, pretrained=True):
    """
    This function receive as parameter information that will be needed to run the experiment, 
    such as the values of tunning parameters (alphas, lambdas)
    model_info: dict
    modelnames: iterable (object that can be iterated)
        This iterable contain models we want to evaluate. This names must corresponde to the json file used for conformal testing
        e.g. ['model-1','model-2']
    lambdas: list
        list of lambdas
    """

    n_data_conf = int(total_conf*pct_cal)
    n_data_val = int(total_conf*pct_val)
    cudnn.benchmark = True

    modelnames = model_info_file['models'].keys()
    modelnames_arch = [(model_info_file['models'][name]['file'], model_info_file['models'][name]['path'],
                        model_info_file['models'][name]['architecture']) for name in modelnames]
    transform = get_calib_transform(image_size)
    params = list(itertools.product(
        modelnames_arch, alphas, kregs, lamdas, predictors))
    m = len(params)
    # Perform the experiment
    dfs = []
    for i in range(m):
        modelinfo, alpha, kreg, lamda, predictor = params[i]
        _modelname = modelinfo[0].replace('.pth', '')
        print(
            f'Model: {_modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')

        #out = experiment(modelname, datasetpath, num_trials,params[i][1], kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor)
        out = run_trials_experiment3(os.path.join(modelinfo[1], modelinfo[0]), modelinfo[2], datapath, num_classes, transform, num_trials, alpha, kreg, lamda,
                                     randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor)
        dfs.append(pd.DataFrame.from_dict({"Model": [_modelname],
                                           "Predictor": [predictor],
                                           "Top1": [np.round(out[0], 3)],
                                           "Top5": [np.round(out[1], 3)],
                                           "alpha": [alpha],
                                           "kreg": [kreg],
                                           "lamda": [lamda],
                                           "Coverage": [np.round(out[2], 3)],
                                           "Size": [np.round(out[3], 3)]}))
    df = pd.concat(dfs)
    return df

# exp-4
# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.


# Experiment 5
# report median size-stratified coverage violation of RAPS and APS
# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.
def get_worst_violation(modelpath, modelname,  datasetpath, transform, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, num_classes, pretrained):
    """

    """
    # Data Loading
    logits = get_logits_dataset(
        modelpath, modelname, datasetpath, transform, bsz, num_classes, pretrained)
    calib_logits, val_logits = tdata.random_split(logits, [n_data_conf, len(
        logits)-n_data_conf])  # A new random split for every trial
    # Prepare the loaders
    calib_loader = torch.utils.data.DataLoader(
        calib_logits, batch_size=bsz, shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_logits, batch_size=bsz, shuffle=False, pin_memory=True)

    # Instantiate and wrap model
    #model = get_model(modelname)
    model = build_model_for_cp(modelpath, modelname,
                               num_classes, pretrained).to(device)
    # Conformalize the model with the APS parameter choice
    conformal_model = ConformalModelLogits(
        model, calib_loader, alpha=alpha, kreg=0, lamda=0, randomized=randomized, allow_zero_sets=True, naive=False)
    aps_worst_violation = get_violation(
        conformal_model, val_loader, strata, alpha)
    # Conformalize the model with an optimal parameter choice
    conformal_model = ConformalModelLogits(model, calib_loader, alpha=alpha, kreg=None, lamda=None, randomized=randomized,
                                           allow_zero_sets=True, naive=False, pct_paramtune=pct_paramtune, lamda_criterion='adaptiveness')
    raps_worst_violation = get_violation(
        conformal_model, val_loader, strata, alpha)

    return aps_worst_violation, raps_worst_violation


def run_experiment5_trails(modelpath, modelname, datasetpath, num_classes, transform, num_trials, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, pretrained=True):
    # Data Loading
    logits = get_logits_dataset(
        modelpath, modelname, datasetpath, transform, bsz, num_classes, pretrained)

    # Instantiate and wrap model
    #model = get_model(modelname)
    build_model_for_cp(modelpath, modelname,
                       num_classes, pretrained).to(device)

    # Perform experiment
    aps_violations = np.zeros((num_trials,))
    raps_violations = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        aps_violations[i], raps_violations[i] = get_worst_violation(
            modelname, datasetpath, transform, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz)
        print(
            f'\n\tAPS Violation: {np.median(aps_violations[0:i+1]):.3f}, RAPS Violation: {np.median(raps_violations[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(aps_violations), np.median(raps_violations)


def save_experiment5_trials(model_info_file, datasetpath, num_classes, total_conf, pct_cal, pct_val, alphas, randomized, bsz, image_size, num_trials, strata, pct_paramtune):
    """ 
    Adaptiveness results after automatically tuning lambda.
    Report median size-stratified coverage violation (eq. 5 in https://arxiv.org/pdf/2009.14193.pdf) of RAPS and APS.
    """
    modelnames = model_info_file['models'].keys()
    modelnames_arch = [(model_info_file['models'][name]['file'], model_info_file['models'][name]['path'],
                        model_info_file['models'][name]['architecture']) for name in modelnames]
    transform = get_calib_transform(image_size)
    params = list(itertools.product(
        modelnames_arch, alphas))
    m = len(params)
    n_data_conf = int(total_conf*pct_cal)
    n_data_val = int(total_conf*pct_val)
    dfs = []
    for i in range(m):
        modelinfo, alpha = params[i]
        _modelname = modelinfo[0].replace('.pth', '')
        print(f'Model: {_modelname} | Desired coverage: {1-alpha}')
        APS_violation_median, RAPS_violation_median = run_experiment5_trails(os.path.join(
            modelinfo[1], modelinfo[0]), modelinfo[2], datasetpath, num_classes, transform, num_trials, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz)
        dfs.append({"Model": _modelname,
                    "alpha": alpha,
                    "APS violation": APS_violation_median,
                    "RAPS violation": RAPS_violation_median})

    df = pd.concat(dfs)
    return df
