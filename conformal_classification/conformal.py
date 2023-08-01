import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
import time
from tqdm import tqdm
from conformal_classification.utils import sort_sum
from conformal_classification.utils_experiments import validate
from conformal_classification.utils_experiments import get_logits_targets
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Conformalize a model with a calibration set.
# Save it to a file in .cache/modelname
# The only difference is that the forward method of ConformalModel also outputs a set.


class ConformalModel(nn.Module):
    """Represents an object for conformal claibration

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

    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False, pct_paramtune=0.3, batch_size=32, lamda_criterion='size', num_classes=20,strata = [[0,1],[2,3],[4,6],[7,10],[11,20]]):
        """
        Parameters
        ----------
        data : pandas.DataFrame
          DataFrame object that make up the dataset.
        info : dict
          Information of the dataset.
        """
        super(ConformalModel, self).__init__()
        self.model = model
        self.alpha = alpha
        # initialize (1.3 is usually a good value)
        self.T = torch.Tensor([1.3])
        self.T, calib_logits = platt(
            self, calib_loader, num_classes)  # platt algorithm (ref)
        self.randomized = randomized  # pendiente teorico
        self.allow_zero_sets = allow_zero_sets  # pendiente
        self.num_classes = len(calib_loader.dataset.dataset.classes)

        # parameter optimizing (paramtuning)
        if kreg == None or lamda == None:
            kreg, lamda, calib_logits = pick_parameters(
                model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion,strata)
        # pendiente
        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda
        # calibration dataloader (prediction,real)
        calib_loader = tdata.DataLoader(
            calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
        # referencia a mi paper
        self.Qhat = conformal_calibration_logits(self, calib_loader)

    def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        logits = self.model(*args, **kwargs)

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            # Temperature scaling (referencia)
            scores = softmax(logits_numpy/self.T.item(), axis=1)
            # step of the algorithm (referencia)
            I, ordered, cumsum = sort_sum(scores)
            # step of the algorithm (referencia), tiene nombre en mi tesis
            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum,
                    penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)

        return logits, S

# Computes the conformal calibration


def conformal_calibration(cmodel, calib_loader):
    print("Conformal calibration")
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
def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01, num_classes=20):
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
    print("Begin Platt scaling.")
    # Save logits so don't need to double compute them
    logits_dataset = get_logits_targets(
        cmodel.model, calib_loader, num_classes)
    logits_loader = torch.utils.data.DataLoader(
        logits_dataset, batch_size=calib_loader.batch_size, shuffle=False, pin_memory=True)

    T = platt_logits(cmodel, logits_loader,
                     max_iters=max_iters, lr=lr, epsilon=epsilon)

    print(f"Optimal T={T.item()}")
    return T, logits_dataset


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

    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False, naive=False, LAC=False, pct_paramtune=0.3, batch_size=32, lamda_criterion='size', strata = [[0,1],[2,3],[4,6],[7,10],[11,20]]):
        super(ConformalModelLogits, self).__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets
        self.T = platt_logits(self, calib_loader)
        self.strata = strata

        if (kreg == None or lamda == None) and not naive and not LAC:
            kreg, lamda, calib_logits = pick_parameters(
                model, calib_loader.dataset, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion, strata)
            calib_loader = tdata.DataLoader(
                calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
        # penalties
        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        if not (kreg == None) and not naive and not LAC:
            self.penalties[:, kreg:] += lamda
        # Qhat
        self.Qhat = 1-alpha
        # aun no queda claro que es LAC
        if not naive and not LAC:
            self.Qhat = conformal_calibration_logits(self, calib_loader)
        elif not naive and LAC:
            gt_locs_cal = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[
                                   0][0] for x in calib_loader.dataset])
            scores_cal = 1-np.array([np.sort(torch.softmax(calib_loader.dataset[i][0]/self.T.item(), dim=0))[
                                    ::-1][gt_locs_cal[i]] for i in range(len(calib_loader.dataset))])
            self.Qhat = np.quantile(scores_cal, np.ceil(
                (scores_cal.shape[0]+1) * (1-alpha)) / scores_cal.shape[0])

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        pdb.set_trace()
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            if not self.LAC:
                I, ordered, cumsum = sort_sum(scores)

                S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum,
                        penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)
            else:
                S = [np.where((1-scores[i, :]) < self.Qhat)[0]
                     for i in range(scores.shape[0])]

        return logits, S

# def conformal_calibration(cmodel, calib_loader):
def conformal_calibration_logits(cmodel, calib_loader):
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
    pdb.set_trace()
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()

            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum,
                               penalties=cmodel.penalties, randomized=True, allow_zero_sets=True)))

        Qhat = np.quantile(E, 1-cmodel.alpha, interpolation='higher')

        return Qhat


def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
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
    # NLL: negative log likelihood
    nll_criterion = nn.CrossEntropyLoss().to(device)
    # weights
    weights = torch.Tensor([1.3]).to(device)
    # make weights torch parameters
    T = nn.Parameter(weights)

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.to(device)
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().to(device))
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T

# CORE CONFORMAL INFERENCE FUNCTIONS

# Generalized conditional quantile function.


def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
    Creates a Dataset from a json file.

    Parameters
    ----------
    scores : str
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
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <=
                  tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1])  # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i, sizes_base[i]-1] * \
                (tau-(cumsum[i, sizes_base[i]-1]-ordered[i, sizes_base[i]-1]) -
                 penalties_cumsum[0, sizes_base[i]-1])  # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        # always predict max size if alpha==0. (Avoids numerical error.)
        sizes[:] = cumsum.shape[1]

    if not allow_zero_sets:
        # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy
        sizes[sizes == 0] = 1

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i, 0:sizes[i]], ]

    return S

# Get the 'p-value'


def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets):  # For one example
    idx = np.where(I == target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]

    U = np.random.random() #coin

    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return U * ordered[idx] + cumsum[(idx[0], idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()

# Gets the histogram of Taus.


def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i+1, :], targets[i].item(), I[i:i+1, :], ordered[i:i+1, :],
                       cumsum[i:i+1, :], penalties[0, :], randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E

# AUTOMATIC PARAMETER TUNING FUNCTIONS


def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[
                             0][0] for x in paramtune_logits])
    kstar = np.quantile(gt_locs_kstar, 1-alpha, interpolation='higher') + 1
    return kstar


def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[
        0][1].shape[0]  # number of classes
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    # predefined grid, change if more precision desired.
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]:
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg,
                                               lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(
            paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star


def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets, strata):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    # predefined grid, change if more precision desired.
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]:
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg,
                                               lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        curr_violation = get_violation(
            conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star


def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion, strata):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(
        calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
    
    calib_loader = tdata.DataLoader(
        calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(
        paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets, strata)
    return kreg, lamda, calib_logits


def get_violation(cmodel, val_loader, strata, alpha):
    #df = pd.DataFrame(columns=['size', 'correct'])
    dfs = []
    for logit, target in val_loader:
        # compute output
        # This is a 'dummy model' which takes logits, for efficiency.
        output, S = cmodel(logit)
        # measure accuracy and record loss
        size = np.array([x.size for x in S])  # vector of Prediction set sizes
        I, _, _ = sort_sum(logit.numpy())
        correct = np.zeros_like(size)  # the same size as the vector of sizes
        for j in range(correct.shape[0]):
            # for each set calculate the correctness
            correct[j] = int(target[j] in list(S[j])) #1
        batch_df = pd.DataFrame.from_dict({'size': size, 'correct': correct})
        dfs.append(batch_df)
    df = pd.concat(dfs)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[(df['size'] >= stratum[0]) & (df['size'] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean()-(1-alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation,df  # the violation
