import numpy as np
import torch
import scipy 
from scipy.signal import convolve, correlate, correlation_lags
from sklearn.model_selection import KFold
import time
torch.set_default_dtype(torch.float64) 

def generate_train_test_split(n_samples):
    n_structures = n_samples
    np.random.seed(0)
    n_train = int(0.8 * n_structures)
    train_index = np.arange(n_structures)
    np.random.shuffle(train_index)
    test_index = train_index[n_train:]
    train_index = train_index[:n_train]
    
    return train_index, test_index

def generate_biased_train_test_split(n_samples):
    #Assumes 100 amorphous structures at the end
    n_structures = n_samples
    amorph_train = np.arange(n_samples-100, n_samples,1)
    np.random.seed(0)
    np.random.shuffle(amorph_train)
    
    amorph_test = amorph_train[:80]
    amorph_train = amorph_train[80:]

    n_structures = n_samples - 100
    np.random.seed(0)
    n_train = int(0.8 * n_samples)-20
    remaining_train_index = np.arange(n_structures)
    np.random.shuffle(remaining_train_index)

    remaining_test_index = remaining_train_index[n_train:]
    remaining_train_index = remaining_train_index[:n_train]

    biased_train_index = np.concatenate([remaining_train_index, amorph_train])
    biased_test_index = np.concatenate([remaining_test_index, amorph_test])
    
    return biased_train_index, biased_test_index

def generate_surface_holdout_split(n_samples):
    #Assumes that we are using the 110 surfaces for test which are located at 673 + 31st-57th index
    #26 structures
    
    n_test = int(0.2 * n_samples) - 26
    n_train = n_samples - n_test
    
    remaining_indexes = np.concatenate([np.arange(673+31), np.arange(673+57,n_samples,1)])
    indexes_110 = np.arange(673+31, 673+57,1)
    np.random.seed(0)
    
    np.random.shuffle(remaining_indexes)
    
    remaining_test_index = remaining_indexes[n_train:]
    remaining_train_index = remaining_indexes[:n_train]
    
    total_train_index = remaining_train_index
    total_test_index = np.concatenate([remaining_test_index, indexes_110])
    
    return total_train_index, total_test_index
    
def surface_holdout(n_samples):
    test_index = np.arange(31,57,1)
    train_index = np.concatenate([np.arange(31), np.arange(57, n_samples)])
    
    return train_index, test_index

def reverse_index(train_align, test_align, train_index, test_index):
    total_number = len(train_align) + len(test_align)
    original_index = torch.zeros(total_number).float()
    
    original_index.index_add_(0, train_index, train_align)
    original_index.index_add_(0, test_index, test_align)
    
    return original_index

        
def t_get_mse(a, b, xdos = None, perc = False):
    if xdos is not None:
        if len(a.size()) > 1:
            mse = (torch.trapezoid((a - b)**2, xdos, axis=1)).mean()
        else:
            mse = (torch.trapezoid((a - b)**2, xdos, axis=0)).mean()
        if not perc:
            return mse
        else:
            mean = b.mean(axis = 0)
            std = torch.trapezoid((b - mean)**2, xdos, axis=1).mean()
            return (100 * mse / std)
    else:
        if len(a.size()) > 2:
            mse = ((a - b)**2).mean(dim = (1,2))

        else:
            mse = ((a - b)**2).mean()
        if len(mse.shape) > 1:
            raise ValueError('Loss became 2D')
        if not perc:
            return mse
        else:
            return 100 * mse / b.std(dim=0, unbiased = True)
        
def t_get_rmse(a, b, xdos=None, perc=False): #account for the fact that DOS is continuous but we are training them pointwise
    """ computes  Root Mean Squared Error (RMSE) of array properties (DOS/aofd).
         a=pred, b=target, xdos, perc: if False return RMSE else return %RMSE"""
    #MIGHT NOT WORK FOR PC
    if xdos is not None:
        if len(a.size()) > 1:
            rmse = torch.sqrt((torch.trapezoid((a - b)**2, xdos, axis=1)).mean())
        else:
            rmse = torch.sqrt((torch.trapezoid((a - b)**2, xdos, axis=0)).mean())
        if not perc:
            return rmse
        else:
            mean = b.mean(axis = 0)
            std = torch.sqrt((torch.trapezoid((b - mean)**2, xdos, axis=1)).mean())
            return (100 * rmse / std)
    else:
        if len(a.size()) > 1:
            rmse = torch.sqrt(((a - b)**2).mean(dim =0))
        else:
            rmse = torch.sqrt(((a - b)**2).mean())
        if not perc:
            return torch.mean(rmse, 0)
        else:
            return torch.mean(100 * (rmse / b.std(dim = 0,unbiased=True)), 0)
        
#Generate shifted data
def shifted_ldos_discrete(ldos, xdos, shift): 
    shifted_ldos = torch.zeros_like(ldos)
    if len(ldos.shape) > 1:
        xdos_shift = torch.round(shift).int()
        for i in range(len(ldos)):
            if xdos_shift[i] > 0:
                shifted_ldos[i] = torch.nn.functional.pad(ldos[i,:-1*xdos_shift[i]], (xdos_shift[i],0))
            elif xdos_shift[i] < 0:
                shifted_ldos[i] = torch.nn.functional.pad(ldos[i,(-1*xdos_shift[i]):], (0,(-1*xdos_shift[i])))
            else:
                shifted_ldos[i] = ldos[i]
    else:        
        xdos_shift = int(torch.round(shift))
        if xdos_shift > 0:
            shifted_ldos = torch.nn.functional.pad(ldos[:-1*xdos_shift], (xdos_shift,0))
        elif xdos_shift < 0:
            shifted_ldos = torch.nn.functional.pad(ldos[(-1*xdos_shift):], (0,(-1*xdos_shift)))
        else:
            shifted_ldos = ldos
    return shifted_ldos


def find_optimal_discrete_shift(prediction, true):
    if true.shape == prediction.shape and len(prediction.shape) == 2:
        shift = []
        for i in range(true.shape[0]):
            corr = correlate(true[i], prediction[i], mode='full')
            shift_i = np.argmax(corr) - len(true[i]) + 1   
            shift.append(shift_i)
        
        
    elif true.shape == prediction.shape and len(prediction.shape) == 1:
        corr = correlate(true, prediction, mode='full')
        shift = np.argmax(corr) - len(true) + 1   
    else:
        print ("input shapes are not the same")
        raise Exception
    return shift

def manual_call2(spline_coefs, spline_positions, x):
    """
    spline_coefs: shape of (n x 4 x spline_positions)
    
    return value: shape of (n x x)
    
    x : shape of (n x n_points)
    """
    interval = torch.round(spline_positions[1] - spline_positions[0], decimals = 4)
    x = torch.clamp(x, min = spline_positions[0], max = spline_positions[-1]- 0.0005)
    indexes = torch.floor((x - spline_positions[0])/interval).long()
    expanded_index = indexes.unsqueeze(dim=1).expand(-1,4,-1)
    x_1 = x - spline_positions[indexes]
    x_2 = x_1 * x_1
    x_3 = x_2 * x_1
    x_0 = torch.ones_like(x_1)
    x_powers = torch.stack([x_3, x_2, x_1, x_0]).permute(1,0,2)
    value = torch.sum(torch.mul(x_powers, torch.gather(spline_coefs, 2, expanded_index)), axis = 1) 
    
    
    return value

def MSE_shift_spline3(y_pred, critical_xdos, target_splines, spline_positions, n_epochs):
    all_shifts = []
    all_mse = []
    optim_search_mse = []
    offsets = torch.arange(-28,29,2)
    with torch.no_grad():
        for offset in offsets:
            shifts = torch.zeros(y_pred.shape[0]) + offset
            shifted_target = manual_call2(target_splines, spline_positions, critical_xdos + shifts.view(-1,1))
            loss_i = ((y_pred - shifted_target)**2).mean(dim = 1)
            optim_search_mse.append(loss_i)
        optim_search_mse = torch.vstack(optim_search_mse)
        min_index = torch.argmin(optim_search_mse, dim = 0)
        optimal_offset = offsets[min_index]
    
    offsets = [optimal_offset -2, optimal_offset, optimal_offset+2]
    
    for offset in (offsets):
        shifts = torch.nn.parameter.Parameter(offset.float())
        opt_adam = torch.optim.Adam([shifts], lr = 1e-1, weight_decay = 0)
        best_error = torch.tensor(100)
        best_shifts = shifts.clone()
        for i in (range(n_epochs)):
            def closure():
                opt_adam.zero_grad()
                shifted_target = manual_call2(target_splines, spline_positions, critical_xdos + shifts.view(-1,1))
                loss_i = ((y_pred - shifted_target)**2).mean()
                loss_i.backward(gradient = torch.tensor(1), inputs = shifts)
                return (loss_i)

            mse = opt_adam.step(closure)
            if mse < best_error:
                best_shifts = shifts.clone()
                best_error = mse.clone()
        #Evaluate
        all_shifts.append(best_shifts)
        shifted_target = manual_call2(target_splines, spline_positions, critical_xdos + best_shifts.view(-1,1))
        mse = ((y_pred - shifted_target)**2).mean(dim = 1)
        all_mse.append(mse)
    all_shifts = torch.vstack(all_shifts)
    all_mse = torch.vstack(all_mse)
    min_index = torch.argmin(all_mse, dim = 0)
    optimal_shift = []
    for i in range(len(min_index)):
        optimal_shift.append(all_shifts.T[i, min_index[i]])
    optimal_shift = torch.tensor(optimal_shift)
    shifted_target = manual_call2(target_splines, spline_positions, critical_xdos + optimal_shift.view(-1,1))
    rmse = t_get_rmse(y_pred, shifted_target, critical_xdos, perc = True)
    return rmse, optimal_shift


def find_kernelreg_CV(features, target, target_splines, spline_positions, xdos, kMM, regularization, train_index, test_index, cv = 2):
    kf = KFold(n_splits = cv)
    features = torch.hstack([features, torch.ones(features.shape[0]).view(-1,1)])
    n_col = features.shape[1]
    rtkMM = torch.tensor(np.real(scipy.linalg.sqrtm(kMM)))
    reg = torch.hstack([(torch.tensor(regularization * rtkMM)), torch.zeros(kMM.shape[0]).view(-1,1)])
    reg = torch.vstack([reg, torch.zeros(n_col)])
    target = target
    m_train_loss = 0
    m_test_loss = 0
    
    
    
    for (cv_train_index, cv_test_index) in kf.split(features[train_index]):
        A = torch.vstack([features[train_index[cv_train_index]], reg])
        b = torch.vstack([target[train_index[cv_train_index]], torch.zeros(n_col,target.shape[1])])

        weights = torch.linalg.lstsq(A, b, driver = "gelsd", rcond = 1e-10).solution
        
        train_pred = features[train_index[cv_train_index]] @ weights
        test_pred = features[train_index[cv_test_index]] @ weights
        
        train_loss, _ = MSE_shift_spline3(train_pred, xdos, target_splines[train_index[cv_train_index]], spline_positions, 200)
        test_loss, _ = MSE_shift_spline3(test_pred, xdos, target_splines[train_index[cv_test_index]], spline_positions, 200)
    
        
        m_train_loss += train_loss/cv
        m_test_loss += test_loss/cv
        
    
    A = torch.vstack([features[train_index], reg])
    b = torch.vstack([target[train_index], torch.zeros(n_col,target.shape[1])])
    
    weights = torch.linalg.lstsq(A, b, driver = "gelsd", rcond = 1e-10).solution
    
    
    
    train_pred = features[train_index] @ weights
    
    test_pred = features[test_index] @ weights

    train_loss, final_train_shifts = MSE_shift_spline3(train_pred, xdos, target_splines[train_index], spline_positions, 200)
    test_loss, final_test_shifts = MSE_shift_spline3(test_pred, xdos, target_splines[test_index], spline_positions, 200)
    
    return weights, m_train_loss, m_test_loss, train_loss, test_loss, final_train_shifts, final_test_shifts

def find_reg_CV(features, target, target_splines, spline_positions, xdos, regularization, train_index, test_index, cv = 2):
    kf = KFold(n_splits = cv)
    features = torch.hstack([features, torch.ones(features.shape[0]).view(-1,1)])
    m = features.shape[1]
    reg = torch.tensor(regularization * torch.eye(m))

    reg[-1, -1] = 0
    m_train_loss = 0
    m_test_loss = 0
    
    
    
    for (cv_train_index, cv_test_index) in kf.split(features[train_index]):
        A = torch.vstack([features[train_index[cv_train_index]], reg])
        b = torch.vstack([target[train_index[cv_train_index]], torch.zeros(m,target.shape[1])])

        weights = torch.linalg.lstsq(A, b, rcond = 1e-10).solution
        
        train_pred = features[train_index[cv_train_index]] @ weights
        test_pred = features[train_index[cv_test_index]] @ weights
        
        train_loss, _ = MSE_shift_spline3(train_pred, xdos, target_splines[train_index[cv_train_index]], spline_positions, 200)
        test_loss, _ = MSE_shift_spline3(test_pred, xdos, target_splines[train_index[cv_test_index]], spline_positions, 200)
    
        
        m_train_loss += train_loss/cv
        m_test_loss += test_loss/cv
        
    
    A = torch.vstack([features[train_index], reg])
    b = torch.vstack([target[train_index], torch.zeros(m,target.shape[1])])
    
    
    weights = torch.linalg.lstsq(A, b, rcond = 1e-10).solution
    
    
    
    train_pred = features[train_index] @ weights
    
    test_pred = features[test_index] @ weights

    train_loss, final_train_shifts = MSE_shift_spline3(train_pred, xdos, target_splines[train_index], spline_positions, 200)
    test_loss, final_test_shifts = MSE_shift_spline3(test_pred, xdos, target_splines[test_index], spline_positions, 200)
    
    return weights, m_train_loss, m_test_loss, train_loss, test_loss, final_train_shifts, final_test_shifts
        
def adam_find_kernelreg_CV(features, target, target_splines, spline_positions, xdos, kMM, regularization, train_index, test_index, batch_size, n_epochs, lr, cv = 2):
    kf = KFold(n_splits = cv)
    Features = torch.hstack([features, torch.ones(features.shape[0]).view(-1,1)])
    n_col = Features.shape[1]
    rtkMM = torch.tensor(np.real(scipy.linalg.sqrtm(kMM)))
    reg = torch.hstack([(torch.tensor(regularization * rtkMM)), torch.zeros(kMM.shape[0]).view(-1,1)])
    reg = torch.vstack([reg, torch.zeros(n_col)])
    Target = target
    m_train_loss = 0
    m_test_loss = 0
    
    
    
    for (cv_train_index, cv_test_index) in kf.split(features[train_index]):
        Sampler = torch.utils.data.RandomSampler(cv_train_index, replacement = False)
        Batcher = torch.utils.data.BatchSampler(Sampler, batch_size, False)
        weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
        opt = torch.optim.Adam([weights], lr = 1e-3, weight_decay = 0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience = 200, threshold = 1e-5, min_lr = 1e-8)
        best_state = weights.clone()
        best_mse = torch.tensor(100)
        for i in range(n_epochs):
            for i_batch in Batcher:
                def closure():
                    opt.zero_grad()
                    reg_features_i = torch.vstack([Features[train_index[i_batch]], reg])
                    target_i = torch.vstack([Target[train_index[i_batch]], torch.zeros(n_col, Target.shape[1])])
                    pred_i = reg_features_i @ weights
                    loss_i = t_get_mse(pred_i, target_i)
                    loss_i.backward()
                    return loss_i
                mse = opt.step(closure)
            with torch.no_grad():
                pred = Features @ weights
                mse = t_get_mse(pred, Target, xdos)
                if mse < best_mse:
                    best_mse = mse.clone()
                    best_state = weights.clone()
                scheduler.step(mse)
                if Batcher.batch_size/2 > len(cv_train_index):
                    break
                if opt.param_groups[0]['lr'] < 1e-4:
                    Batcher.batch_size *= 2
                    opt.param_groups[0]['lr'] = lr

        
        train_pred = Features[train_index[cv_train_index]] @ best_state
        test_pred = Features[train_index[cv_test_index]] @ best_state
        
        train_loss, _ = MSE_shift_spline3(train_pred, xdos, target_splines[train_index[cv_train_index]], spline_positions, 200)
        test_loss, _ = MSE_shift_spline3(test_pred, xdos, target_splines[train_index[cv_test_index]], spline_positions, 200)
    
        
        m_train_loss += train_loss/cv
        m_test_loss += test_loss/cv
        
    Sampler = torch.utils.data.RandomSampler(train_index, replacement = False)
    Batcher = torch.utils.data.BatchSampler(Sampler, batch_size, False)
    weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
    opt = torch.optim.Adam([weights], lr = 1e-3, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience = 200, threshold = 1e-5, min_lr = 1e-8)
    best_state = weights.clone()
    best_mse = torch.tensor(100)
    for i in range(n_epochs):
        for i_batch in Batcher:
            def closure():
                opt.zero_grad()
                reg_features_i = torch.vstack([Features[i_batch], reg])
                target_i = torch.vstack([Target[i_batch], torch.zeros(n_col, Target.shape[1])])
                pred_i = reg_features_i @ weights
                loss_i = t_get_mse(pred_i, target_i)
                loss_i.backward()
                return loss_i
            mse = opt.step(closure)
        with torch.no_grad():
            pred = Features @ weights
            mse = t_get_mse(pred, Target, xdos)
            if mse < best_mse:
                best_mse = mse.clone()
                best_state = weights.clone()
            scheduler.step(mse)
            if Batcher.batch_size/2 > len(cv_train_index):
                break
            if opt.param_groups[0]['lr'] < 1e-4:
                Batcher.batch_size *= 2
                opt.param_groups[0]['lr'] = lr
                print ("The batch_size is now: ", Batcher.batch_size)


    train_pred = Features[train_index] @ best_state
    test_pred = Features[test_index] @ best_state

    train_loss, final_train_shifts = MSE_shift_spline3(train_pred, xdos, target_splines[train_index], spline_positions, 200)
    test_loss, final_test_shifts = MSE_shift_spline3(test_pred, xdos, target_splines[test_index], spline_positions, 200)
    
    return best_state, m_train_loss, m_test_loss, train_loss, test_loss, final_train_shifts, final_test_shifts

def adam_find_normalreg_CV(features, target, target_splines, spline_positions, xdos, regularization, train_index, test_index, batch_size, n_epochs, lr, cv = 2):
    kf = KFold(n_splits = cv)
    Features = torch.hstack([features, torch.ones(features.shape[0]).view(-1,1)])
    n_col = Features.shape[1]
    reg = torch.tensor(regularization * torch.eye(n_col))
    Target = target
    reg[-1, -1] = 0
    m_train_loss = 0
    m_test_loss = 0
    
    
    
    for (cv_train_index, cv_test_index) in kf.split(features[train_index]):
        Sampler = torch.utils.data.RandomSampler(cv_train_index, replacement = False)
        Batcher = torch.utils.data.BatchSampler(Sampler, batch_size, False)
        weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
        opt = torch.optim.Adam([weights], lr = 1e-3, weight_decay = 0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience = 200, threshold = 1e-5, min_lr = 1e-8)
        best_state = weights.clone()
        best_mse = torch.tensor(100)
        for i in range(n_epochs):
            for i_batch in Batcher:
                def closure():
                    opt.zero_grad()
                    reg_features_i = torch.vstack([Features[train_index[i_batch]], reg])
                    target_i = torch.vstack([Target[train_index[i_batch]], torch.zeros(n_col, Target.shape[1])])
                    pred_i = reg_features_i @ weights
                    loss_i = t_get_mse(pred_i, target_i)
                    loss_i.backward()
                    return loss_i
                mse = opt.step(closure)
            with torch.no_grad():
                pred = Features @ weights
                mse = t_get_mse(pred, Target, xdos)
                if mse < best_mse:
                    best_mse = mse.clone()
                    best_state = weights.clone()
                scheduler.step(mse)
                if Batcher.batch_size/2 > len(cv_train_index):
                    break
                if opt.param_groups[0]['lr'] < 1e-4:
                    Batcher.batch_size *= 2
                    opt.param_groups[0]['lr'] = lr

        
        train_pred = Features[train_index[cv_train_index]] @ best_state
        test_pred = Features[train_index[cv_test_index]] @ best_state
        
        train_loss, _ = MSE_shift_spline3(train_pred, xdos, target_splines[train_index[cv_train_index]], spline_positions, 200)
        test_loss, _ = MSE_shift_spline3(test_pred, xdos, target_splines[train_index[cv_test_index]], spline_positions, 200)
        
        m_train_loss += train_loss/cv
        m_test_loss += test_loss/cv
        
    
    Sampler = torch.utils.data.RandomSampler(train_index, replacement = False)
    Batcher = torch.utils.data.BatchSampler(Sampler, batch_size, False)
    weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
    opt = torch.optim.Adam([weights], lr = 1e-3, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience = 200, threshold = 1e-5, min_lr = 1e-8)
    best_state = weights.clone()
    best_mse = torch.tensor(100)
    for i in range(n_epochs):
        for i_batch in Batcher:
            def closure():
                opt.zero_grad()
                reg_features_i = torch.vstack([Features[i_batch], reg])
                target_i = torch.vstack([Target[i_batch], torch.zeros(n_col, Target.shape[1])])
                pred_i = reg_features_i @ weights
                loss_i = t_get_mse(pred_i, target_i)
                loss_i.backward()
                return loss_i
            mse = opt.step(closure)
        with torch.no_grad():
            pred = Features @ weights
            mse = t_get_mse(pred, Target, xdos)
            if mse < best_mse:
                best_mse = mse.clone()
                best_state = weights.clone()
            scheduler.step(mse)
            if Batcher.batch_size/2 > len(cv_train_index):
                break
            if opt.param_groups[0]['lr'] < 1e-4:
                Batcher.batch_size *= 2
                opt.param_groups[0]['lr'] = lr
                print ("The batch_size is now: ", Batcher.batch_size)


    train_pred = Features[train_index] @ best_state
    test_pred = Features[test_index] @ best_state

    train_loss, final_train_shifts = MSE_shift_spline3(train_pred, xdos, target_splines[train_index], spline_positions, 200)
    test_loss, final_test_shifts = MSE_shift_spline3(test_pred, xdos, target_splines[test_index], spline_positions, 200)
    
    
    return weights, m_train_loss, m_test_loss, train_loss, test_loss, final_train_shifts, final_test_shifts

def L_find_kernelreg_CV(features, target, target_splines, spline_positions, xdos, kMM, regularization, train_index, test_index, n_epochs, lr, cv = 2):
    kf = KFold(n_splits = cv)
    Features = torch.hstack([features, torch.ones(features.shape[0]).view(-1,1)])
    n_col = Features.shape[1]
    rtkMM = torch.tensor(np.real(scipy.linalg.sqrtm(kMM)))
    reg = torch.hstack([(torch.tensor(regularization * rtkMM)), torch.zeros(kMM.shape[0]).view(-1,1)])
    reg = torch.vstack([reg, torch.zeros(n_col)])
    Target = target
    m_train_loss = 0
    m_test_loss = 0
    
    
    
    for (cv_train_index, cv_test_index) in kf.split(features[train_index]):
        weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
        opt = torch.optim.LBFGS([weights], lr = lr, line_search_fn = "strong_wolfe", tolerance_grad = 1e-20, tolerance_change = 1-20, history_size = 200)
        best_state = weights.clone()
        best_mse = torch.tensor(100)
        for i in range(n_epochs):
            def closure():
                opt.zero_grad()
                reg_features_i = torch.vstack([Features[train_index[cv_train_index]], reg])
                target_i = torch.vstack([Target[train_index[cv_train_index]], torch.zeros(n_col, Target.shape[1])])
                pred_i = reg_features_i @ weights
                loss_i = t_get_mse(pred_i, target_i)
                loss_i.backward()
                return loss_i
            mse = opt.step(closure)
            with torch.no_grad():
                pred = Features @ weights
                mse = t_get_mse(pred, Target, xdos)
                if mse < best_mse:
                    best_mse = mse.clone()
                    best_state = weights.clone()

        
        train_pred = Features[train_index[cv_train_index]] @ best_state
        test_pred = Features[train_index[cv_test_index]] @ best_state
        
        train_loss, _ = MSE_shift_spline3(train_pred, xdos, target_splines[train_index[cv_train_index]], spline_positions, 200)
        test_loss, _ = MSE_shift_spline3(test_pred, xdos, target_splines[train_index[cv_test_index]], spline_positions, 200)
    
        
        m_train_loss += train_loss/cv
        m_test_loss += test_loss/cv
        
    weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
    opt = torch.optim.LBFGS([weights], lr = lr, line_search_fn = "strong_wolfe", tolerance_grad = 1e-20, tolerance_change = 1-20, history_size = 200)
    best_state = weights.clone()
    best_mse = torch.tensor(100)
    for i in range(n_epochs):
        def closure():
            opt.zero_grad()
            reg_features_i = torch.vstack([Features[train_index[cv_train_index]], reg])
            target_i = torch.vstack([Target[train_index[cv_train_index]], torch.zeros(n_col, Target.shape[1])])
            pred_i = reg_features_i @ weights
            loss_i = t_get_mse(pred_i, target_i)
            loss_i.backward()
            return loss_i
        mse = opt.step(closure)
        with torch.no_grad():
            pred = Features @ weights
            mse = t_get_mse(pred, Target, xdos)
            if mse < best_mse:
                best_mse = mse.clone()
                best_state = weights.clone()


    train_pred = Features[train_index] @ best_state
    test_pred = Features[test_index] @ best_state

    train_loss, final_train_shifts = MSE_shift_spline3(train_pred, xdos, target_splines[train_index], spline_positions, 200)
    test_loss, final_test_shifts = MSE_shift_spline3(test_pred, xdos, target_splines[test_index], spline_positions, 200)
    
    return best_state, m_train_loss, m_test_loss, train_loss, test_loss, final_train_shifts, final_test_shifts

def L_find_normalreg_CV(features, target, target_splines, spline_positions, xdos, regularization, train_index, test_index, n_epochs, lr, cv = 2):
    kf = KFold(n_splits = cv)
    Features = torch.hstack([features, torch.ones(features.shape[0]).view(-1,1)])
    n_col = Features.shape[1]
    reg = torch.tensor(regularization * torch.eye(n_col))
    Target = target
    reg[-1, -1] = 0
    m_train_loss = 0
    m_test_loss = 0
    
    
    
    for (cv_train_index, cv_test_index) in kf.split(features[train_index]):
        weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
        opt = torch.optim.LBFGS([weights], lr = lr, line_search_fn = "strong_wolfe", tolerance_grad = 1e-20, tolerance_change = 1-20, history_size = 200)
        best_state = weights.clone()
        best_mse = torch.tensor(100)
        for i in range(n_epochs):
            def closure():
                opt.zero_grad()
                reg_features_i = torch.vstack([Features[train_index[cv_train_index]], reg])
                target_i = torch.vstack([Target[train_index[cv_train_index]], torch.zeros(n_col, Target.shape[1])])
                pred_i = reg_features_i @ weights
                loss_i = t_get_mse(pred_i, target_i)
                loss_i.backward()
                return loss_i
            mse = opt.step(closure)
            with torch.no_grad():
                pred = Features @ weights
                mse = t_get_mse(pred, Target, xdos)
                if mse < best_mse:
                    best_mse = mse.clone()
                    best_state = weights.clone()

        
        train_pred = Features[train_index[cv_train_index]] @ best_state
        test_pred = Features[train_index[cv_test_index]] @ best_state
        
        train_loss, _ = MSE_shift_spline3(train_pred, xdos, target_splines[train_index[cv_train_index]], spline_positions, 200)
        test_loss, _ = MSE_shift_spline3(test_pred, xdos, target_splines[train_index[cv_test_index]], spline_positions, 200)
    
        
        m_train_loss += train_loss/cv
        m_test_loss += test_loss/cv
        
    weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)    
    opt = torch.optim.LBFGS([weights], lr = lr, line_search_fn = "strong_wolfe", tolerance_grad = 1e-20, tolerance_change = 1-20, history_size = 200)
    best_state = weights.clone()
    best_mse = torch.tensor(100)
    for i in range(n_epochs):
        def closure():
            opt.zero_grad()
            reg_features_i = torch.vstack([Features[train_index[cv_train_index]], reg])
            target_i = torch.vstack([Target[train_index[cv_train_index]], torch.zeros(n_col, Target.shape[1])])
            pred_i = reg_features_i @ weights
            loss_i = t_get_mse(pred_i, target_i)
            loss_i.backward()
            return loss_i
        mse = opt.step(closure)
        with torch.no_grad():
            pred = Features @ weights
            mse = t_get_mse(pred, Target, xdos)
            if mse < best_mse:
                best_mse = mse.clone()
                best_state = weights.clone()


    train_pred = Features[train_index] @ best_state
    test_pred = Features[test_index] @ best_state

    train_loss, final_train_shifts = MSE_shift_spline3(train_pred, xdos, target_splines[train_index], spline_positions, 200)
    test_loss, final_test_shifts = MSE_shift_spline3(test_pred, xdos, target_splines[test_index], spline_positions, 200)
    
    return best_state, m_train_loss, m_test_loss, train_loss, test_loss, final_train_shifts, final_test_shifts



n_surfaces = 154
n_bulkstructures = 773
n_total_structures = 773 + 154

bulk_index = torch.concat([torch.arange(673), torch.arange(827,927)])
surface_index = torch.arange(673, 827)


surface_train_index, surface_test_index = generate_train_test_split(n_surfaces)
bulk_train_index, bulk_test_index = generate_train_test_split(n_bulkstructures)
total_train_index, total_test_index = generate_train_test_split(n_total_structures)
surface_holdout_train_index, surface_holdout_test_index = surface_holdout(n_surfaces)
bulk_biased_train_index, bulk_biased_test_index = generate_biased_train_test_split(n_bulkstructures)
total_biased_train_index, total_biased_test_index = generate_biased_train_test_split(n_total_structures)
holdout_train_index, holdout_test_index = generate_surface_holdout_split(n_total_structures)

with torch.no_grad():
    xdos = torch.load("../../data/xdos.pt")
    y = torch.load("..")
    feats = torch.load("..")
#    kMM = torch.load("../")
    index_train, index_test = surface_train_index, surface_test_index
    regularization_value = 0
    splines = torch.load("../../data/")
    splines_positions = torch.arange(-24.5537 - 2,11.3464 + 2,0.001)
    

time_now = time.time()
# weights, m_train_loss, m_test_loss, train_loss, test_loss, opt_shift_train, opt_shift_test = find_reg_CV(feats, y, splines, splines_positions, xdos, regularization_value, index_train, index_test, cv = 2)
# weights, m_train_loss, m_test_loss, train_loss, test_loss, opt_shift_train, opt_shift_test = find_kernelreg_CV(feats, y, splines, splines_positions, xdos, kMM, regularization_value, index_train, index_test, cv = 2)
# weights, m_train_loss, m_test_loss, train_loss, test_loss, opt_shift_train, opt_shift_test =  adam_find_kernelreg_CV(feats, y, splines, splines_positions, xdos, kMM, regularization_value, surface_train_index, surface_test_index, 16, 100, 1e-3)
# weights, m_train_loss, m_test_loss, train_loss, test_loss, opt_shift_train, opt_shift_test =  adam_find_normalreg_CV(feats, y, splines, splines_positions, xdos, regularization_value, surface_train_index, surface_test_index, 16, 100, 1e-3)
# weights, m_train_loss, m_test_loss, train_loss, test_loss, opt_shift_train, opt_shift_test =  L_find_kernelreg_CV(feats, y, splines, splines_positions, xdos, kMM, regularization_value, surface_train_index, surface_test_index, 60, 1)
# weights, m_train_loss, m_test_loss, train_loss, test_loss, opt_shift_train, opt_shift_test = L_find_normalreg_CV(feats, y, splines, splines_positions, xdos, regularization_value, surface_train_index, surface_test_index, 60, 1)
time_end = time.time()


opt_shift = reverse_index(opt_shift_train, opt_shift_test, index_train, index_test)
errors = [m_train_loss, m_test_loss, train_loss, test_loss]

torch.save(errors, "./errors.pt")
torch.save(weights, "./optimal_weights.pt")
torch.save(opt_shift, "./opt_shift.pt")

results = ["The train error is {} \n".format(train_loss) , "The test error is {} \n".format(test_loss), "Time taken: {}".format(round(time_end - time_now),2)]

w = open("./results.txt", "w")
w.writelines(results)
w.close()