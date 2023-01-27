import torch
import numpy as np
from . import committee
from ..datasets import dataset
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from ..utils import utils
from ..loss import loss
from . import architectures

grid = np.array([1e-10,1e-8,1e-6,1e-4,1e-2,1,100])
log_grid = np.log(grid)
maxiter = 2

gpr_loss = utils.get_rmse
ridge_loss = utils.get_rmse
torch.set_default_dtype(torch.float64) 
#ridge_loss = loss.t_get_rmse

def train_analytical_model_GPR(Feature: torch.tensor, feature_name: str, target: torch.tensor, target_name: str, x_dos: torch.tensor, kMM: np.array, train_index: np.array, train_ratio:int, cv: int = 2):
    """
    Finds optimal regularization value and returns optimal regression weights
    
    Args:
        Feature (torch.tensor): Kernel Tensor
        feature_name (str): feature_name for output printing
        target (torch.tensor): Target Tensor
        target_name (str): target name for output printing
        x_dos (torch.tensor): x_dos tensor
        kMM (np.array): kMM array
        train_index (np.array): train indices
        train_ratio (int): train_ratio for splicing        
        cv (int, optional): cross validation paramters
    
    Returns:
        weights: np.array containing weights solved analytically using the optimal regularization value
    """
    n_train = int(train_ratio * len(train_index))
    train_index = train_index[:n_train]
    errors = np.zeros(len(log_grid))
    for i, value in enumerate(log_grid):
        errors[i] = GPR_k_fold_error(value, target, Feature.detach().numpy(), kMM, cv, train_index, xdos = x_dos.detach().numpy())
    reg_init = np.exp(log_grid[errors.argmin()])
    print ("The log_grid error is:")
    print (errors)
    print ("Init value is: {}".format(reg_init))
    

    rmin = minimize(GPR_k_fold_error, [np.log(reg_init)], args=(target, Feature.detach().numpy(), kMM, cv, train_index, x_dos), method="Nelder-Mead", options={"maxiter":maxiter})
    opt_reg = np.exp(rmin["x"])[0]
    print ("Optimal Regularization value for GPR, Feature:{}, Target: {}, train_ratio: {} is: {}".format(feature_name, target_name, train_ratio, opt_reg))

    weights = utils.get_regression_weights(target[train_index].numpy(), 
                                 kMM=kMM, 
                                 regularization=opt_reg,
                                 kNM=Feature[train_index].detach().numpy())

    return weights


def GPR_k_fold_error(i_regularization: np.array, target: torch.tensor, kNM: torch.tensor, kMM: np.array, cv: int, train_idx: np.array, xdos: np.array):
    """helper function for the train_analytical_model_GPR function
    
    Args:
        i_regularization (np.array): log of the regularization value
        target (torch.tensor): target tensor
        kNM (torch.tensor)
        kMM (np.array): Similarity matrix between the sparse kernel points, for regularization
        cv (int): cross validation parameter
        train_idx (np.array): train set indexes
        xdos (np.array): xdos array
    
    Returns:
        float: k_fold MSE value
    """
    kfold = KFold(n_splits=cv, shuffle=False)
    regularization = np.exp(i_regularization)
    temp_err = 0.
    for train, test in kfold.split(train_idx):
        w = utils.get_regression_weights(target[train_idx[train]].numpy(), 
                                   kMM=kMM, 
                                   regularization=regularization, 
                                   kNM=kNM[train_idx[train]])
        target_pred = kNM @ w
        temp_err += gpr_loss(target_pred[train_idx[test]], target[train_idx[test]].numpy())#, xdos, perc=True)
    print ("The performance at reg value {} is {}".format(regularization, temp_err/cv))
    return temp_err/cv

def train_analytical_model_ridge(Feature: torch.tensor, feature_name: str, target: torch.tensor, target_name: str, x_dos: torch.tensor, train_index: np.array, train_ratio:int, cv: int = 2):
    """
    Finds optimal regularization value and returns optimal regression weights
    
    Args:
        Feature (torch.tensor): Kernel Tensor
        feature_name (str): feature_name for output printing
        target (torch.tensor): Target Tensor
        target_name (str): target name for output printing
        x_dos (torch.tensor): x_dos tensor
        train_index (np.array): train indices
        train_ratio (int): train_ratio for output printing        
        cv (int, optional): cross validation paramters
    
    Returns:
        model: sklearn model using the optimal regularization value
    
    
    """
    n_train = int(train_ratio * len(train_index))
    train_index = train_index[:n_train]
    errors = np.zeros(len(log_grid))
    for i, value in enumerate(log_grid):
        errors[i] = k_fold_Ridgeregression(value, target, Feature, cv, x_dos, train_index)
    reg_init = np.exp(log_grid[errors.argmin()])
    print ("The log_grid error is:")
    print (errors)
    print ("Init value is: {}".format(reg_init))
    rmin = minimize(k_fold_Ridgeregression, [np.log(reg_init)], args = (target, Feature, cv, x_dos, train_index), method = "Nelder-Mead", options = {"maxiter" : maxiter})
    opt_reg = np.exp(rmin["x"])[0]
    print ("Optimal regularization value for Ridge, Feature:{}, target: {}, train_ratio: {} is {}".format(feature_name, target_name, train_ratio, opt_reg))
    model = Ridge(alpha = opt_reg, fit_intercept = False, solver = 'svd')
    model.fit(Feature[train_index], target[train_index])
    
    return model        


def k_fold_Ridgeregression(reg: np.array, target: torch.tensor, Features: torch.tensor, cv: int, xdos: torch.tensor, train_index: np.array):
    """Summary
    
    Args:
        reg (np.array): log of the regularization value
        target (torch.tensor): Target
        Features (torch.tensor): Features
        cv (int): cross-validation parameters
        xdos (torch.tensor): xdos tensor
        train_index (np.array): train set indices
    
    Returns:
        TYPE: Description
    """
    reg = np.exp(reg).item()
    model = Ridge(alpha = reg, fit_intercept = False, solver = 'svd')
    kfold = KFold(n_splits = cv, shuffle=False)
    err = 0
    for train, test in kfold.split(train_index):
        model.fit(Features[train_index[train]], target[train_index[train]])
        pred = model.predict(Features[train_index[test]])
        err += ridge_loss(pred, target[train_index[test]].detach().numpy())#, xdos, perc = True)
    print ("The performance at reg value {} is {}".format(reg, err/cv))
    return err / cv

def train_torch_linear_model(feature, feature_name, target, target_name, datatype, loss, opt, lr, n_epochs, cv, xdos, train_index, device, reg, val = False):
    if datatype == "pc":
        target = target[:,None]
    model = architectures.LinearModel(feature.shape[1], target.shape[1], xdos, reg, opt, device)
    kwargs = {"pin_memory":True} if device == "cuda:0" else {}
    if val:
        n_train = int(0.2 * len(train_index)) #20% val
        val_index = train_index[:n_train]
        train_index = train_index[n_train:]
        val_data = TensorDataset(feature[val_index], target[val_index])
        valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
    else:
        valdata_loader = None
    train_data = TensorDataset(feature[train_index], target[train_index])
    traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)

    print ("Currently training torch linear model with features :{}, target: {}, opt: {}, lr:{}".format(feature_name, target_name, opt, lr))
    loss_history = model.fit(traindata_loader, valdata_loader, loss, lr, n_epochs)


    return model

def torch_linear_optimize_hypers(feature, feature_name, target, target_name, datatype, opt, lr, n_epochs, device, cv, xdos, train_index, loss, val):
    if datatype == "pc":
        target = target[:,None]
    errors = np.zeros(len(log_grid))
    for i, value in enumerate(log_grid):
        errors[i] = get_regularization_loss(value, target, feature, loss, cv, opt, lr, n_epochs, device, xdos, train_index, val)
    reg_init = np.exp(log_grid[errors.argmin()])
    print ("The log_grid error is:")
    print (errors)
    print ("Init value is: {}".format(reg_init))
    rmin = minimize(get_regularization_loss, [np.log(reg_init)], args = (target, feature, loss, cv, opt, lr, n_epochs, device, xdos, train_index, val), method = "Nelder-Mead", options = {"maxiter" : maxiter})
    opt_reg = np.exp(rmin["x"])[0]
    print ("Optimal regularization value for Ridge, Feature:{}, target: {} is {}".format(feature_name, target_name, opt_reg))

    return opt_reg

def get_regularization_loss(reg, target, feature, loss, cv, opt, lr, n_epochs, device, xdos, train_index, val = False):
    kwargs = {"pin_memory":True} if device == "cuda:0" else {}
    reg = np.exp(reg)
    kfold = KFold(n_splits = cv, shuffle = False)
    print ("Currently trying regularization :{}".format(reg))
    total_loss = 0
    for i_train, i_test in kfold.split(train_index):
        if val:
            n_val = int(0.2 * len(i_train)) #20% val
            val_index = train_index[:n_val]
            train_index = train_index[n_val:]
            val_data = TensorDataset(feature[train_index[val_index]], target[train_index[val_index]])
            valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
        else:
            valdata_loader = None
        train_data = TensorDataset(feature[train_index], target[train_index])
        traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)
        trial_model = architectures.LinearModel(feature.shape[1], target.shape[1], xdos, reg, opt, device)
        loss_history = trial_model.fit(traindata_loader, valdata_loader, loss, lr, n_epochs)

        total_loss += loss(trial_model(feature[train_index[i_test]]).to(device), target[train_index[i_test]].to(device)).item()/cv
    print ("The performance at reg value:{} is {}".format(reg, total_loss))
    return total_loss

def torch_estimate_hypers(feature, feature_name, target, target_name, datatype, opt, lr, n_epochs, device, cv, xdos, train_index, loss):
    if datatype == "pc":
        target = target[:,None]
    kwargs = {"pin_memory":True} if device == "cuda:0" else {}
    shuffler = np.random.default_rng()
    np.random.seed(0)
    for n in range(5):
        shuffler.shuffle(train_index)
        n_train = int(0.2 * len(train_index)) #20% val
        val_index = train_index[:n_train]
        train_index = train_index[n_train:]
        val_data = TensorDataset(feature[train_index[val_index]], target[train_index[val_index]])
        valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
        train_data = TensorDataset(feature[train_index], target[train_index])
        traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)
        trial_model = architectures.LinearModel(feature.shape[1], target.shape[1], xdos, 0, "Rprop", device) #not implemented yet, will finish when NN goes into production
    pass
