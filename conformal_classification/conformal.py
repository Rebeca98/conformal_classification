import pdb
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from typing import List
import torchvision
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from conformal_classification.utils import sort_sum
from conformal_classification.utils import validate
from conformal_classification.utils import get_logits_targets
from conformal_classification.utils import get_wc_violation

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu')
# Conformalize a model with a calibration set.#
# Save it to a file in .cache/modelname
# The only difference is that the forward method of ConformalModel also outputs a set.


class ConformalModel(nn.Module):

    def __init__(self, model, calib_logits, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False, pct_paramtune=0.3, batch_size=32, naive=False, lamda_criterion='size', num_classes=20, strata=[[0, 1], [2, 3], [4, 6], [7, 10], [11, 20]]):
        """
        Represents an object for conformal claibration

        Attributes
        ----------
        model : torchvision.models
           Trained CNN. (e.g torchvision.models.efficientnet.EfficientNet)
        calib_loader : torch.utils.data.DataLoader
            Dataloader with calibration dataset
        alpha: float
            Error level of the conformal algorithm
        kreg: int
            regularization term. It can be None if we want to choose an optimal kreg automatically.
        lamda: float
            A scalar value that is non-negative and controls the magnitude of the penalty, with a larger value resulting in a higher penalty. 
        randomized: bool
            Flag that enables to randomize the conformal procedure. 
        allow_zero_sets: bool
            Flag that enables zero size sets. If not, the prediction sets will have at least one element.
        pct_paramtune: float
            Percentage (value between 0 and 1) of the calibration dataset for the parameter tunning (optimizing)
        batch_size: int
            Batch size
        lamda_criterion: str
            lambda optimization criterion (options: 'size', 'adaptiveness' )
        num_classes: int
            Number of classes
        naive: (default: False)
        strata: list
            Subdivisions that provide insights into the extent of prediction regions. These subdivisions aim to illustrate 
            the level of prediction complexity. For instance, in a scenario involving 10 classes and the defined 
            strata of [[1,2],[3,5],[6,10]], the first stratum corresponds to prediction regions of sizes 1 and 2. 
            The second stratum  corresponds to prediction regions of sizes 3,4 and 5. 
            The final stratum encompasses prediction regions with sizes 6, 7, 8, 9, and 10.
        """
        super(ConformalModel, self).__init__()
        self.model = model
        self.alpha = alpha
        self.T, calib_logits = platt(logits_loader=calib_logits)  # platt algorithm
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = num_classes
        self.strata = strata
        # parameter optimizing (paramtuning)
        if kreg == None or lamda == None and not naive:
            kreg, lamda, calib_logits = pick_parameters(
                model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion, strata)
        
        # ya e sun dataloader
        calib_loader = calib_logits.dataset
        #tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.penalties = np.zeros((1, self.num_classes))
        if not (kreg == None) and not naive:
            self.penalties[:, kreg:] += lamda
        # Qhat for naive methodology
        self.Qhat = 1-alpha

        if not naive:
            self.Qhat = conformal_calibration_logits(self, calib_loader)

    def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        logits = self.model(*args, **kwargs)

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum,
                    penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)

        return logits, S


def conformal_calibration(cmodel, calib_loader:torch.utils.data.DataLoader)-> float:
    """
    Conformal Calibration Algorithm.
    generalized inverse quantile conformity score function

    Calculates the generalized quantile for conformal calibration.

    Parameters
    ----------
    cmodel : ConformalModelLogits
        Conformalized model.

    calib_loader : torch.utils.data.DataLoader
        Dataloader with calibration dataset.

    Returns
    -------
    Qhat : float
        Generalized quantile from Eq (3) (Uncertainty Sets for Image Classifiers Using Conformal Prediction,
        Anastasios, N and Stephen Bates and Jitendra Malik and Michael I. Jordan, 2021. In Proceedings of [Conference Name], [Year]).
    """
 
    logging.info('Conformal calibration')
    with torch.no_grad():
        E = np.array([])
        for x, targets in tqdm(calib_loader):
            logits = cmodel.model(x.to(device)).detach().cpu().numpy()
            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum,
                               penalties=cmodel.penalties, randomized=True, allow_zero_sets=True)))

        Qhat = np.quantile(E, 1-cmodel.alpha, interpolation='higher')

        return Qhat





# Temperature scaling

# todo: change name to Temperature scaling


def platt(logits_loader, max_iters=10, lr=0.01, epsilon=0.01):
    """
    Temperature scaling algorithm.

    Parameters
    ----------
    cmodel : conformal_classification.conformal.ConformalModelLogits
        conformal model

    calib_loader : torch.utils.data.DataLoader
        Dataloader with calibration dataset

    max_iters: int

    lr: float

    epsilon: float

    num_classes: int
        Number of classes


    Returns
    -------
    T: float
        Temperature scalar
    logits_dataset: torch.utils.data.dataset.TensorDataset
        Model's logits  Temperature scaling algorithm.


    """
    
    logging.info('Begin Platt scaling')
    # Save logits so don't need to double compute them
    # logits_dataset = get_logits_targets(
    #     cmodel.model, calib_loader, num_classes)
    # logits_loader = torch.utils.data.DataLoader(
    #     logits_dataset, batch_size=calib_loader.batch_size, shuffle=False, pin_memory=True)

    T = platt_logits(logits_loader,
                     max_iters=max_iters, lr=lr, epsilon=epsilon)

    print(f"Optimal T={T.item()}")
    return T, logits_loader


"""
        INTERNAL FUNCTIONS
"""

# Precomputed-logit versions of the above functions.


class ConformalModelLogits(nn.Module):
    """
    Represents an object for conformal claibration

    Attributes
    ----------
    model : nn.Module
      Trained Neural Network Model for Image classification. This model must be on eval mode.
        e.g. 
            model = torchvision.models.resnet152(pretrained=True,progress=True)
            model.eval()
    calib_loader: torch.utils.data.DataLoader
      Dataloader with calibration dataset
    alpha: double
      error level of the conformal algorithm
    kreg: int
        pendiente
    lambda: int o double
        pendiente 
    randomized: boolean
        pendiente
    allow_zero_sets: boolean
        pendiente
    pct_paramtune: double
        percentage of the calibration dataset for the parameter tunning (optimizing)
    batch_size: int
        batch_size for (pendiente)
    lamda_criterion: string
        pendiente
    """

    def __init__(self, model:torchvision.models, calib_logits_loader:torch.utils.data.dataloader.DataLoader, alpha:float, 
                 kreg:int=None, lamda:float=None, randomized:bool=True, allow_zero_sets:bool=False, naive:bool=False, pct_paramtune:float=None, 
                 batch_size:int=None, lamda_criterion:str=None,num_classes:int=20, strata: List[List[int]]=None):
        super(ConformalModelLogits, self).__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = num_classes
        #self.T, calib_logits = platt(calib_loader)
        self.T = platt_logits(calib_logits_loader)
        self.strata = strata

        if (kreg == None or lamda == None) and not naive:
            kreg, lamda, calib_logits = pick_parameters(
                self.model, calib_logits_loader, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion, strata)
            calib_logits_loader = calib_logits.dataset#tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
        # penalties
        #self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        self.penalties = np.zeros((1, self.num_classes))
        if not (kreg == None) and not naive:
            self.penalties[:, kreg:] += lamda
        # Qhat for naive
        self.Qhat = 1-alpha

        if not naive:
            #  calculate quantile Qhat for RAPS and APS 
            self.Qhat = conformal_calibration_logits(self, calib_logits_loader)

    # implement Conformal Prediction
    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized is None:
            randomized = self.randomized
        if allow_zero_sets is None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            # calibration
            scores = softmax(logits_numpy/self.T.item(), axis=1)
            I, ordered, cumsum = sort_sum(scores)
            S = gcq(scores, tau=self.Qhat, I=I, ordered=ordered, cumsum=cumsum,
                    penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)

        return logits, S


def conformal_calibration_logits(cmodel, calib_logits_loader:torch.utils.data.DataLoader)-> float:
    """
    Conformal Calibration Algorithm.

    Calculates the generalized quantile for conformal calibration.

    Parameters
    ----------
    cmodel : ConformalModelLogits
        Conformalized model.

    calib_loader : torch.utils.data.DataLoader
        Dataloader with calibration dataset logits.

    Returns
    -------
    Qhat : float
        Generalized inverse quantile conformity score function. (Uncertainty Sets for Image Classifiers Using Conformal Prediction,
        Anastasios, N and Stephen Bates and Jitendra Malik and Michael I. Jordan, 2021. In Proceedings of [Conference Name], [Year]).
    """
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_logits_loader:
            logits = logits.detach().cpu().numpy()

            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum,
                               penalties=cmodel.penalties, randomized=True, allow_zero_sets=True)))

        Qhat = np.quantile(E, 1-cmodel.alpha, interpolation='higher')

        return Qhat


def platt_logits(logits_loader:torch.utils.data.dataloader.DataLoader, max_iters=10, lr=0.01, epsilon=0.01):
    """
    Temperature scaling algorithm

    Parameters
    ----------
    logits_loader : torch.utils.data.DataLoader
        Dataloader with model logits
    max_iters : int
        Maximum number of iterations
    lr: float (0,1)
        learning rate
    epsilon: float (0,1)
        Tolerance, It represents the acceptable difference between values to consider them as close or similar

    Returns
    -------
    T: float
        Temperature scalar 
    """
    # NLL: negative log likelihood
    nll_criterion = nn.CrossEntropyLoss().to(device)
    # weights
    weights = torch.Tensor([1.3]).to(device)
    # make weights torch parameters
    T = nn.Parameter(weights)

    optimizer = optim.SGD([T], lr=lr)
    for _ in range(max_iters):
        T_old = T.item()
        for x, targets in logits_loader:  # x: predicted value, target: truth value
            optimizer.zero_grad()
            x = x.to(device)
            x.requires_grad = True
            out = x/T
            # This criterion computes the cross entropy loss between input logits and target.
            loss = nll_criterion(out, targets.long().to(device))
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T

# CORE CONFORMAL INFERENCE FUNCTIONS

# Generalized conditional quantile function.


def gcq(scores:np.array, tau:float, I:np.array, ordered:np.array, cumsum:np.array, penalties:np.array, randomized:bool, allow_zero_sets:bool)-> List:
    """
    GGeneralized inverse of conditional quantile function

    In this function we calculate the prediction set.


    Parameters
    ----------
    scores : np.array
        Scores resulting from softmax transformation with Temperature scaling.
    
    tau : float
        The quantile level for the conformal calibration.
    
    I : np.array
        Indices of the sorted scores.
    
    ordered : np.array
        Ordered scores in descending order.
    
    cumsum : np.array
        Cumulative sum of the ordered scores.
    
    penalties : np.array
        Penalties associated with the misclassification.
    
    randomized : bool
        If True, applies randomized conformal prediction.
    
    allow_zero_sets : bool
        If True, allows the prediction sets to be empty.
    
    Returns
    -------
    S : list
        The prediction sets for new examples.
    """

    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <=
                  tau).sum(axis=1) + 1
    # the minimum between the calculated sizes and the worst case (all possible labels)
    sizes_base = np.minimum(sizes_base, scores.shape[1])
  
    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i, sizes_base[i]-1] * \
                (tau-(cumsum[i, sizes_base[i]-1]-ordered[i, sizes_base[i]-1]) -
                 penalties_cumsum[0, sizes_base[i]-1]) 

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1]

    if not allow_zero_sets:
        sizes[sizes == 0] = 1

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i, 0:sizes[i]], ]

    return S


def get_tau(target:int, I:np.array, ordered:np.array, cumsum:np.array, penalty:np.array, randomized:bool, allow_zero_sets:bool):
    """
    Calculates tau (p-value)  with calibration set

    Parameters
    ----------
    target : int
        The target index for which the tau value is to be calculated.
        
    I : np.array
        An array containing indices, typically resulting from sorting an array of scores.
        
    ordered : np.array
        An array of scores that have been sorted in ascending or descending order.
        
    cumsum : np.array
        The cumulative sum of the ordered scores.
        
    penalty : np.array
        An array of penalty values to be added to the tau calculation.
        
    randomized : bool
        If True, introduces randomness into the tau calculation.
        
    allow_zero_sets : bool
        If True, allows the calculation of tau for sets with zero scores.

    Returns
    -------
    tau : float
    """
   
    idx = np.where(I == target)

    
    tau_nonrandom = cumsum[idx]
    if not randomized:
        return tau_nonrandom + penalty[0]

    
    U = np.random.random()  # coin
    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return U * ordered[idx] + cumsum[(idx[0], idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()


def giq(scores:np.array, targets:np.array, I:np.array, ordered:np.array, cumsum:np.array, penalties:np.array, randomized:bool, allow_zero_sets:bool):
    """
    Generalized inverse quantile conformity score function. Computes the minimum tau in [0, 1] such that the correct label enters,
    based on the method described in the paper by Romano, Sesia, and Candes.

    Parameters
    ----------
    scores : np.ndarray
        Scores resulting from a softmax transformation with temperature scaling.
        
    targets : np.ndarray
        Array of target indices for each example, indicating the correct labels.
        
    I : np.ndarray
        An array of indices, typically resulting from sorting an array of scores.
        
    ordered : np.ndarray
        An array of scores that have been sorted in ascending or descending order.
        
    cumsum : np.ndarray
        The cumulative sum of the ordered scores.
        
    penalties : np.ndarray
        An array of penalty values to be added to the tau calculation.
        
    randomized : bool
        If True, introduces randomness into the tau calculation.
        
    allow_zero_sets : bool
        If True, allows the calculation of tau for sets with zero scores.

    Returns
    -------
    E : np.ndarray
        The array of tau values for each example, indicating the minimum tau such that the correct label is included.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(targets[i].item(), I[i:i+1, :], ordered[i:i+1, :],
                       cumsum[i:i+1, :], penalties[0, :], randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E

# AUTOMATIC PARAMETER TUNING FUNCTIONS

def pick_kreg(paramtune_loader, alpha:float):
    """
    ?

    Parameters
    ----------
    paramtune_loader : torch.utils.data.DataLoader
        DataLoader con los logits y las etiquetas.
    alpha : float (0,1)
        El nivel de confianza.

    Returns
    -------
    kstar : int
        El valor de k*.
    """
    logging.info('Pick k_reg')
    gt_locs_kstar = []

    for x, targets in tqdm(paramtune_loader.dataset):
        x, targets = x.to(device), targets.to(device)
        for i in range(x.shape[0]):
            logits = x[i]
            label = targets[i]
            sorted_indices = torch.argsort(logits, descending=True)
            # Convertir a CPU antes de usar NumPy
            sorted_indices = sorted_indices.cpu().numpy()
            label = label.cpu().numpy()
            loc = np.where(sorted_indices == label)[0][0]
            gt_locs_kstar.append(loc)

    gt_locs_kstar = np.array(gt_locs_kstar)
    kstar = np.quantile(gt_locs_kstar, 1 - alpha, interpolation='higher') + 1
    return kstar

    # gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[
    #                          0][0] for x in paramtune_logits])
    # kstar = np.quantile(gt_locs_kstar, 1-alpha, interpolation='higher') + 1
    # return kstar
    # logits, labels = paramtune_logits.tensors
    # sorted_indices = torch.argsort(logits, descending=True, dim=1)
    # gt_locs_kstar = torch.zeros(logits.shape[0], dtype=torch.int64)
    # for i, label in enumerate(labels):
    #     gt_locs_kstar[i] = (sorted_indices[i] == label).nonzero()[0]
    # kstar = torch.quantile(gt_locs_kstar.float(), 1 - alpha, interpolation='higher') + 1
    # return kstar.item()


def pick_lamda_size(model, paramtune_logits, alpha, kreg, randomized, allow_zero_sets):
    """
    Pick lambda based on prediction regions size. We test with grid of lambda values to obtain the 
    smallest prediction region.


    Parameters
    ----------
    model : ?
        Trained model
    paramtune_loader : float (0,1)

    alpha : list of str, optional
        List of categories to filter registers (default is [])

    kreg: 

    randomized: bool
        Boolean 

    allow_zero_sets: bool
        Boolean 

    Returns
    -------
    lambda_star : float
        lambda value where we obtain the smallest size.
    """
    # Calculate lamda_star
    logging.info('Pick lambda size')
    paramtune_loader = paramtune_logits.dataset
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    # predefined grid, change if more precision desired.
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]:
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg,
                                               lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        
        top1_avg, top5_avg, coverage_avg, size_avg = validate(
            paramtune_loader, conformal_model, print_bool=False)
        if size_avg < best_size:
            best_size = size_avg
            lamda_star = temp_lam
    return lamda_star








def pick_lamda_adaptiveness(model, paramtune_logits, alpha, kreg, randomized, allow_zero_sets, strata):
    """
    ?

    Parameters
    ----------
    model : 

    paramtune_loader : float

    alpha : list of str, optional
        List of categories to filter registers (default is [])
    kreg :
        Extra named arguments passed to build_json
    randomized: bool

    allow_zero_sets:

    strata: list
        List of trata (e.g. for 10 classes [[1,2],[3,6],[7,10]])

    Returns
    -------
    lamda_star: float
        Instance of the created Dataset
    """
    logging.info('Pick lambda adaptiveness')
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    # predefined grid, change if more precision desired.
    paramtune_loader = paramtune_logits.dataset 
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]:
        
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg,
                                               lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        curr_violation = get_wc_violation(
            conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star


def pick_parameters(model, calib_logits_loader:torch.utils.data.DataLoader, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion, strata):
    """
    Generalized conditional quantile function.

    Parameters
    ----------
    model : 
    calib_logits_loader : torch.utils.data.DataLoader

    alpha : list of str, optional
        List of categories to filter registers (default is [])
    kreg :
        Extra named arguments passed to build_json
    lamda:

    randomized: bool

    allow_zero_sets:
    pct_paramtune:

    batch_size

    lamda_criterion:

    strata: list

    Returns
    -------
    Dataset
        Instance of the created Dataset
    """
    if lamda_criterion not in ['adaptiveness', 'size']:
        raise ValueError("lamda_criterion debe ser 'adaptiveness' o 'size'")

    logging.info('Hyperparameter tunning')
    
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits_loader)))
    paramtune_logits, calib_logits = tdata.random_split(
        calib_logits_loader, [num_paramtune, len(calib_logits_loader)-num_paramtune])

    #paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(
                model, paramtune_logits, alpha, kreg, randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(
                model, paramtune_logits, alpha, kreg, randomized, allow_zero_sets, strata)
    return kreg, lamda, calib_logits
