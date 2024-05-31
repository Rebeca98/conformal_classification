from tqdm import tqdm
from collections import defaultdict
from typing import List,Optional
import numpy as np
from scipy.stats import median_abs_deviation as mad
# Torch
from typing import Tuple
import torchvision
import pandas as pd

import torch.utils.data as tdata
import torch

from conformal_classification.conformal import ConformalModelLogits

from conformal_classification.utils import get_calib_transform
from conformal_classification.utils import get_logits_dataset_inference
from conformal_classification.utils import sort_sum
from conformal_classification.utils import build_model_for_cp
from conformal_classification.utils import validate
from conformal_classification.utils import get_logits_dataset

#device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu')
#device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu')


def get_logits_model(model_path:str,model_name:str, data_path:str,transform: torchvision.transforms.transforms.Compose, bsz:int ,num_classes:int):
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    model_path: str
        Path to trained model.
    model_name: str
        Name of the model (options: efficientnet_b0,efficientnet_b1,efficientnet_b0)
    data_path: str
        Path to data used for calibration
    transform: torchvision.transforms.transforms.Compose
        Image transformations 
    bsz: int
        Batch size
    num_classes: int
        Number of classes
    pretrained: bool
        Boolean that indicates to load a pretrained model.
    Returns
    -------
    logits: ?
        Instance of the created Dataset
    model: ? 

    
    """
    # Data Loading
    logits = get_logits_dataset(model_path = model_path,
                                model_name = model_name, 
                                data_path = data_path, 
                                transform = transform, 
                                bsz = bsz, 
                                num_classes = num_classes)
    
    

    # Instantiate and wrap model
    model = build_model_for_cp(model_path=model_path,
                               model_name = model_name,
                               num_classes = num_classes).to(device)
    
    return logits, model


def get_logits(model_path,model_name,data_path,transform, bsz,num_classes)-> torch.utils.data.TensorDataset:
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    model_path: str
        Path to trained model.
    model_name: str
        Name of the model (options: efficientnet_b0,efficientnet_b1,efficientnet_b0)
    data_path: str
        Path to data used for calibration
    transform: torchvision.transforms.transforms.Compose
        Image transformations 
    bsz: int
        Batch size
    num_classes: int
        Number of classes
    pretrained: bool
        Boolean that indicates to load a pretrained model.
    Returns
    -------
    logits: ?
        Instance of the created Dataset
    model: ? 

    
    """
    # Data Loading
    logits = get_logits_dataset(model_path = model_path,
                                model_name = model_name, 
                                data_path = data_path, 
                                transform = transform, 
                                bsz = bsz, 
                                num_classes = num_classes)
    
    

    # Instantiate and wrap model

    return logits
    
    
def evaluate(cmodel,loader_val):
    """
    Report the median of top1 and top5 averages 

    Parameters
    ----------
    model: ?
        Trained model
    logits: ?
        Model's output
    alpha: float 
        ?
    kreg: int
        ?
    lamda: float
        ?
    randomized: bool
        Boolean that indicates if we randomized the procedure. ?
    n_data_conf: int
        Number of conformal data 
    pct_paramtune: float
        percentage of data for parameter tunning    
    bsz: int
        Batch size
    
    Returns
    -------
    Dataset
        the values inlcude the median of top1's, top5's, coverages and sizes, 
        median_abs_deviation of the top1s, top5s, coverages and sizes

    """
    top1_avg, top5_avg, cvg_avg, sz_avg = validate(val_loader=loader_val, 
                                                       model = cmodel)
    return top1_avg, top5_avg, cvg_avg, sz_avg




def get_violation(cmodel, val_loader, strata, alpha)-> Tuple[float, pd.DataFrame]:
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
    return wc_violation,df_size_correctness  # the violation


def eval_strata(cmodel, val_loader, strata, alpha):
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
    dfs = []
    for logit, target in val_loader:
        # compute output
        # This is a 'dummy model' which takes logits, for efficiency.
        _, S = cmodel(logit)
        # measure accuracy and record loss
        size = np.array([x.size for x in S])  # vector of Prediction set sizes
   
        correct = np.zeros_like(size)  # the same size as the vector of sizes
        for j in range(correct.shape[0]):
            # for each set calculate the correctness
            correct[j] = int(target[j] in list(S[j]))  # 1 or 0
           
        batch_df = pd.DataFrame.from_dict({'size': size, 'cvg': correct})
        dfs.append(batch_df)
    df = pd.concat(dfs)
    dfs_temp = []
    for stratum in strata:
        temp_df = df[(df['size'] >= stratum[0]) & (df['size'] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        results = {'difficulty': f"{stratum[0]} to {stratum[1]}",
                   'violation': abs(temp_df["cvg"].mean()-(1-alpha)),
                   'size': temp_df["size"].mean(),
                   'size_mean_correct':temp_df[temp_df["cvg"] == 1]["size"].mean(),
                   'coverage': temp_df["cvg"].mean(),
                   'count': temp_df.shape[0]
                   }
        df_results = pd.DataFrame([results])
        # temp_df.loc[:, 'difficulty'] = f"{stratum[0]} to {stratum[1]}"
        # temp_df.loc[:, "violation"] = abs(temp_df.correct.mean()-(1-alpha))
        dfs_temp.append(df_results)
    df_temp_all = pd.concat(dfs_temp)
    return df_temp_all  # the violation


def evaluate_conformal_prediction(model, cal_logits, test_logits, bsz, alpha, kreg, lamda, randomized, allow_zero_sets,  pct_paramtune,  predictor, lamda_criterion,strata, num_classes):
        # Conformalize the model
        # Experiment logic
    naive_bool = predictor == 'Naive'
    lamda_predictor = lamda

    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0  # No regularization.
        kreg = 0

    # A new random split for every trial
    #logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    # logits_cal, logits_val = tdata.random_split(logits, [n_data_conf, len(
    #     logits)-n_data_conf])  # A new random split for every trial
    # # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        cal_logits, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_test = torch.utils.data.DataLoader(
        test_logits, batch_size=bsz, shuffle=False, pin_memory=True)
    
    conformal_model = ConformalModelLogits(model = model, 
                                            calib_logits_loader = loader_cal, 
                                            alpha = alpha, 
                                            kreg = kreg, 
                                            lamda = lamda_predictor, 
                                            randomized = randomized,
                                            allow_zero_sets = allow_zero_sets,
                                            pct_paramtune = pct_paramtune, 
                                            naive = naive_bool, 
                                            batch_size = bsz, 
                                            lamda_criterion = lamda_criterion,
                                            strata = strata,
                                            num_classes=num_classes)
    
    top1_avg, top5_avg, cvg_avg, sz_avg = evaluate(cmodel=conformal_model, 
                                                    loader_val= loader_test)
    
    worst_violation,df_size_coverage = get_violation(cmodel=conformal_model, 
                                                        val_loader=loader_test,
                                                        strata= strata, 
                                                        alpha = alpha)
    df_eval_strata = eval_strata(cmodel = conformal_model,
                                    val_loader= loader_test, 
                                    strata = strata, 
                                    alpha = alpha)
    
    results = {'top1_avg': [top1_avg],
                'top5_avg': [top5_avg],
                'cvg_avg': [cvg_avg],
                'sz_avg': [sz_avg],
                'worst_violation': [worst_violation],
                'df_size_coverage': [df_size_coverage],
                'df_eval_strata':[df_eval_strata]
                }
    df_results = pd.DataFrame(results)
    return df_results




def evaluate_trials(model, cal_logits, test_logits, bsz, num_trials, alpha, kreg, lamda, randomized, allow_zero_sets,  pct_paramtune,  predictor, lamda_criterion,strata,num_classes):
    #model, logits, n_data_conf, bsz, alpha, kreg, lamda, randomized, allow_zero_sets,  pct_paramtune,  predictor, lamda_criterion, print_bool, strata)
    """
    Creates a Dataset from a json file.
    
    Parameters
    ----------
    model: ?
        Trained model
    logits: ?
        Model's output
    alpha: float 
        ?
    kreg: int
        ?
    lamda: float
        ?
    randomized: bool
        Boolean that indicates if we randomized the procedure. ?
    n_data_conf: int
        Number of conformal data 
    pct_paramtune: float
        percentage of data for parameter tunning    
    bsz: int
        Batch size


    Returns
    -------
    Dataset
        the values inlcude the median of top1's, top5's, coverages and sizes, 
        median_abs_deviation of the top1s, top5s, coverages and sizes
    """
    import random
    dfs_results = []
    for i in tqdm(range(num_trials)):
        seed = i
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
        df_result = evaluate_conformal_prediction(model, cal_logits, test_logits, bsz, alpha, kreg, lamda, randomized, allow_zero_sets,  pct_paramtune,  predictor, lamda_criterion,strata, num_classes)
        df_result['trial'] = i
        dfs_results.append(df_result)
    
    results_trials = pd.concat(dfs_results)
    return results_trials


#region USER INFERENCE
def evaluate_and_conformalize_model(model, cal_logits, test_logits, bsz, alpha, randomized, allow_zero_sets,  pct_paramtune,  predictor,strata, num_classes,lamda_criterion='size'):
        # Conformalize the model
        # Experiment logic
    naive_bool = predictor == 'Naive'
    

    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0  # No regularization.
        kreg = 0
    if predictor in ['RAPS']:
        lamda_predictor = None  
        kreg = None

    # A new random split for every trial
    #logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    # logits_cal, logits_val = tdata.random_split(logits, [n_data_conf, len(
    #     logits)-n_data_conf])  # A new random split for every trial
    # # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        cal_logits, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_test = torch.utils.data.DataLoader(
        test_logits, batch_size=bsz, shuffle=False, pin_memory=True)
    
    conformal_model = ConformalModelLogits(model = model, 
                                            calib_logits_loader = loader_cal, 
                                            alpha = alpha, 
                                            kreg = kreg, 
                                            lamda = lamda_predictor, 
                                            randomized = randomized,
                                            allow_zero_sets = allow_zero_sets,
                                            pct_paramtune = pct_paramtune, 
                                            naive = naive_bool, 
                                            batch_size = bsz, 
                                            lamda_criterion = lamda_criterion,
                                            strata = strata,
                                            num_classes=num_classes)
    
    top1_avg, top5_avg, cvg_avg, sz_avg = evaluate(cmodel=conformal_model, 
                                                    loader_val= loader_test)
    
    worst_violation,df_size_coverage = get_violation(cmodel=conformal_model, 
                                                        val_loader=loader_test,
                                                        strata= strata, 
                                                        alpha = alpha)
    df_eval_strata = eval_strata(cmodel = conformal_model,
                                    val_loader= loader_test, 
                                    strata = strata, 
                                    alpha = alpha)
    
    results = {'top1_avg': [top1_avg],
                'top5_avg': [top5_avg],
                'cvg_avg': [cvg_avg],
                'sz_avg': [sz_avg],
                'worst_violation': [worst_violation],
                'df_size_coverage': [df_size_coverage],
                'df_eval_strata':[df_eval_strata]
                }
    df_results = pd.DataFrame(results)
    return df_results,conformal_model


def conformalize_model(model, cal_logits, bsz, alpha, randomized, allow_zero_sets,  pct_paramtune,  predictor, strata, num_classes, lamda_criterion='size'):
        # Conformalize the model
        # Experiment logic
    naive_bool = predictor == 'Naive'
    

    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0  # No regularization.
        kreg = 0
    if predictor in ['RAPS']:
        lamda_predictor = None  
        kreg = None


    # A new random split for every trial
    #logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    # logits_cal, logits_val = tdata.random_split(logits, [n_data_conf, len(
    #     logits)-n_data_conf])  # A new random split for every trial
    # # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        cal_logits, batch_size=bsz, shuffle=False, pin_memory=True)
  
    conformal_model = ConformalModelLogits(model = model, 
                                            calib_logits_loader = loader_cal, 
                                            alpha = alpha, 
                                            kreg = kreg, 
                                            lamda = lamda_predictor, 
                                            randomized = randomized,
                                            allow_zero_sets = allow_zero_sets,
                                            pct_paramtune = pct_paramtune, 
                                            naive = naive_bool, 
                                            batch_size = bsz, 
                                            lamda_criterion = lamda_criterion,
                                            strata = strata,
                                            num_classes=num_classes)
    
    

    return conformal_model



def cp_inference(trained_model, num_classes:int,alpha:float, image_size:int, predictor:str,
                    data_path_calibration:str, bsz:int, randomized:bool,allow_zero_sets:bool,
                     pct_paramtune:float, lamda_criterion:str='size', strata:List[List[int]] = None, model_evaluation:bool=True,data_path_test:str=None):
    transform = get_calib_transform(image_size)
    cal_logits = get_logits_dataset_inference(trained_model, data_path = data_path_calibration, transform = transform, bsz = bsz, num_classes=num_classes)
    
    if model_evaluation: 
        test_logits = get_logits_dataset_inference(trained_model, data_path = data_path_test, transform = transform, bsz = bsz, num_classes=num_classes)
    
    
        df_result, conformalized_model = evaluate_and_conformalize_model(trained_model, 
                                        model_evaluation = model_evaluation, 
                                        cal_logits = cal_logits, 
                                        test_logits = test_logits, 
                                        bsz = bsz, 
                                        alpha = alpha, 
                                        randomized = randomized, 
                                        allow_zero_sets = allow_zero_sets,  
                                        pct_paramtune = pct_paramtune,  
                                        predictor = predictor, 
                                        lamda_criterion =lamda_criterion,
                                        strata = strata,
                                        num_classes = num_classes)
        return df_result,conformalized_model
    else:
        conformalized_model = conformalize_model(trained_model, 
                                        cal_logits = cal_logits, 
                                        bsz = bsz, 
                                        alpha = alpha, 
                                        randomized = randomized, 
                                        allow_zero_sets = allow_zero_sets,  
                                        pct_paramtune = pct_paramtune,  
                                        predictor = predictor, 
                                        lamda_criterion =lamda_criterion,
                                        strata = strata,
                                        num_classes = num_classes)
        return conformalized_model
#endregion