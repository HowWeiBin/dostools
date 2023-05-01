import numpy as np
import torch
import copy
import scipy 
import copy
import ase
import ase.io
from scipy.signal import convolve, correlate, correlation_lags
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
        if len(a.size()) > 1:
            mse = ((a - b)**2).mean(dim = 1)
        else:
            mse = ((a - b)**2).mean()
        if len(mse.shape) > 1:
            raise ValueError('Loss became 2D')
        if not perc:
            return torch.mean(mse, 0)
        else:
            return torch.mean(100 * (mse / b.std(dim=0, unbiased = True)),0)
        
        
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


def normal_reg_train_Ad(feat, target, train_index, test_index, regularization, n_epochs, batch_size, lr):
    patience = 20
    index = train_index
    t_index = test_index

    features = torch.hstack([feat, torch.ones(feat.shape[0]).view(-1,1)])

    Sampler = torch.utils.data.RandomSampler(index, replacement = False)
    Batcher = torch.utils.data.BatchSampler(Sampler, batch_size, False)

    Features = features[index]
    t_Features = features[t_index]
    n_col = Features.shape[1]


    Target = target[index]
    t_Target = target[t_index]


    # reg_features = torch.vstack([Features, reg])
    # reg_target = torch.vstack([Target, torch.zeros(n_col,Target.shape[1])])


    reg = regularization * torch.eye(n_col)
    reg[-1, -1] = 0


    weights = torch.nn.Parameter((torch.rand(Features.shape[1], Target.shape[1])- 0.5))
    opt = torch.optim.Adam([weights], lr = lr, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience = 500, threshold = 1e-7, min_lr = 1e-8)
    best_mse = torch.tensor(100)
    
    for epoch in range(n_epochs):
        for i_batch in Batcher:
            def closure():
                opt.zero_grad()
                reg_features_i = torch.vstack([Features[i_batch], reg])
                target_i = torch.vstack([Target[i_batch], torch.zeros(n_col, Target.shape[1])])
                pred_i = reg_features_i @ weights
                opt_shift = find_optimal_discrete_shift(np.array(pred_i[:len(i_batch)].detach()),np.array(target_i[:len(i_batch)].detach()))
                pred_i[:len(i_batch)] = shifted_ldos_discrete(pred_i[:len(i_batch)], xdos, torch.tensor(opt_shift))
                loss_i = t_get_mse(pred_i, target_i)
                loss_i.backward()
                return loss_i
            opt.step(closure)

        with torch.no_grad():
            preds = Features @ weights
            opt_shift = find_optimal_discrete_shift(np.array(preds),np.array(Target))
            preds = shifted_ldos_discrete(preds, xdos, torch.tensor(opt_shift))
            epoch_mse = t_get_mse(preds, Target, xdos)

            if epoch_mse < best_mse:
                best_mse = epoch_mse
                best_state = weights.clone()

            scheduler.step(epoch_mse)

            if Batcher.batch_size > 1024:
                break

            if opt.param_groups[0]['lr'] < 1e-4:
                Batcher.batch_size *= 2
                opt.param_groups[0]['lr'] = lr
                print ("The batch_size is now: ", Batcher.batch_size)

    
    with torch.no_grad():
        final_preds = Features @ best_state 
        final_t_preds = t_Features @ best_state

        opt_shift_train = find_optimal_discrete_shift(np.array(final_preds),np.array(Target))
        final_preds2 = shifted_ldos_discrete(final_preds, xdos, torch.tensor(opt_shift_train))
        opt_shift_test = find_optimal_discrete_shift(np.array(final_t_preds),np.array(t_Target))
        final_t_preds2 = shifted_ldos_discrete(final_t_preds, xdos, torch.tensor(opt_shift_test))

        loss_dos = t_get_rmse(final_preds2, Target, xdos, perc = True)
        test_loss_dos = t_get_rmse(final_t_preds2, t_Target, xdos, perc = True)
        return best_state, loss_dos, test_loss_dos, opt_shift_train, opt_shift_test

def normal_reg_train_L(feat, target, train_index, test_index, regularization, n_epochs, lr):    
    patience = 20
    index = train_index
    t_index = test_index
    features = torch.hstack([feat, torch.ones(feat.shape[0]).view(-1,1)])
    Features = features[index]
    t_Features = features[t_index]
    n_col = Features.shape[1]
    Target = target[index]
    t_Target = target[t_index]
    reg = regularization * torch.eye(n_col)
    reg[-1, -1] = 0
    reg_features = torch.vstack([Features, reg])
    reg_target = torch.vstack([Target, torch.zeros(n_col,Target.shape[1])])
    
    weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5)
    opt = torch.optim.LBFGS([weights], lr = lr, line_search_fn = "strong_wolfe", tolerance_grad = 1e-20, tolerance_change = 1-20, history_size = 200)
    lowest_loss = torch.tensor(100)
    best_state = weights.clone()
    for epoch in range(n_epochs):
        def closure():
            opt.zero_grad()
            pred_i = reg_features @ weights
            opt_shift = find_optimal_discrete_shift(np.array(pred_i[:len(index)].detach()),np.array(reg_target[:len(index)].detach()))
            pred_i[:len(index)] = shifted_ldos_discrete(pred_i[:len(index)], xdos, torch.tensor(opt_shift))
            loss_i = t_get_mse(pred_i, reg_target)
            loss_i.backward()
            return loss_i
        mse_loss = opt.step(closure)
        
        if mse_loss < lowest_loss:
            best_state = weights.clone()


    
    with torch.no_grad():
        final_preds = Features @ best_state 
        final_t_preds = t_Features @ best_state

        opt_shift_train = find_optimal_discrete_shift(np.array(final_preds),np.array(Target))
        final_preds2 = shifted_ldos_discrete(final_preds, xdos, torch.tensor(opt_shift_train))
        opt_shift_test = find_optimal_discrete_shift(np.array(final_t_preds),np.array(t_Target))
        final_t_preds2 = shifted_ldos_discrete(final_t_preds, xdos, torch.tensor(opt_shift_test))

        loss_dos = t_get_rmse(final_preds2, Target, xdos, perc = True)
        test_loss_dos = t_get_rmse(final_t_preds2, t_Target, xdos, perc = True)
        return best_state, loss_dos, test_loss_dos, opt_shift_train, opt_shift_test
        
def kernel_reg_train_Ad(feat, target, train_index, test_index, kMM, regularization, n_epochs, batch_size, lr):
    index = train_index
    t_index = test_index
    features = torch.hstack([feat, torch.ones(feat.shape[0]).view(-1,1)])
    Features = features[index]
    t_Features = features[t_index]
    n_col = Features.shape[1]
    Target = target[index]
    t_Target = target[t_index]
    Sampler = torch.utils.data.RandomSampler(index, replacement = False)
    Batcher = torch.utils.data.BatchSampler(Sampler, batch_size, False)
    rtkMM = scipy.linalg.sqrtm(kMM)
    reg = torch.hstack([(torch.tensor(regularization * rtkMM)), torch.zeros(kMM.shape[0]).view(-1,1)])
    reg = torch.vstack([reg, torch.zeros(n_col)])

    weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5) 

    opt = torch.optim.Adam([weights], lr = 1e-3, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience = 200, threshold = 1e-5, min_lr = 1e-8)
    
    best_state = weights.clone()
    best_mse = torch.tensor(100)
    for epoch in range(n_epochs):
        for i_batch in Batcher:
            def closure():
                opt.zero_grad()
                reg_features_i = torch.vstack([Features[i_batch], reg])
                target_i = torch.vstack([Target[i_batch], torch.zeros(n_col, Target.shape[1])])
                pred_i = reg_features_i @ weights
                opt_shift = find_optimal_discrete_shift(np.array(pred_i[:len(i_batch)].detach()),np.array(target_i[:len(i_batch)].detach()))
                pred_i[:len(i_batch)] = shifted_ldos_discrete(pred_i[:len(i_batch)], xdos, torch.tensor(opt_shift))
                loss_i = t_get_mse(pred_i, target_i)
                loss_i.backward()
                return loss_i
            opt.step(closure)

        with torch.no_grad():
            preds = Features @ weights
            opt_shift = find_optimal_discrete_shift(np.array(preds),np.array(Target))
            preds = shifted_ldos_discrete(preds, xdos, torch.tensor(opt_shift))            
            epoch_mse = t_get_mse(preds, Target, xdos)


            if epoch_mse < best_mse:
                best_mse = epoch_mse
                best_state = weights.clone()

            scheduler.step(epoch_mse)

            if Batcher.batch_size > 1024:
                break

            if opt.param_groups[0]['lr'] < 1e-6:
                Batcher.batch_size *= 2
                opt.param_groups[0]['lr'] = lr
                print ("The batch_size is now: ", Batcher.batch_size)


    

    with torch.no_grad():
        final_preds = Features @ best_state 
        final_t_preds = t_Features @ best_state

        opt_shift_train = find_optimal_discrete_shift(np.array(final_preds),np.array(Target))
        final_preds2 = shifted_ldos_discrete(final_preds, xdos, torch.tensor(opt_shift_train))
        opt_shift_test = find_optimal_discrete_shift(np.array(final_t_preds),np.array(t_Target))
        final_t_preds2 = shifted_ldos_discrete(final_t_preds, xdos, torch.tensor(opt_shift_test))

        loss_dos = t_get_rmse(final_preds2, Target, xdos, perc = True)
        test_loss_dos = t_get_rmse(final_t_preds2, t_Target, xdos, perc = True)
        return best_state, loss_dos, test_loss_dos, opt_shift_train, opt_shift_test
    


def kernel_reg_train_L(feat, target, train_index, test_index, kMM, regularization, n_epochs, lr):
    index = train_index
    t_index = test_index
    features = torch.hstack([feat, torch.ones(feat.shape[0]).view(-1,1)])
    Features = features[index]
    t_Features = features[t_index]
    n_col = Features.shape[1]
    Target = target[index]
    t_Target = target[t_index]
    rtkMM = scipy.linalg.sqrtm(kMM)
    reg = torch.hstack([(torch.tensor(regularization * rtkMM)), torch.zeros(kMM.shape[0]).view(-1,1)])
    reg = torch.vstack([reg, torch.zeros(n_col)])

    reg_features = torch.vstack([Features, reg])
    reg_target = torch.vstack([Target, torch.zeros(n_col,Target.shape[1])])
    weights = torch.nn.Parameter(torch.rand(Features.shape[1], Target.shape[1])- 0.5) 
    opt = torch.optim.LBFGS([weights], lr = lr, line_search_fn = "strong_wolfe", tolerance_grad = 1e-20, tolerance_change = 1-20, history_size = 200)

    best_state = weights.clone()
    best_mse = torch.tensor(100)
    for epoch in range(n_epochs):
        def closure():
            opt.zero_grad()
            pred_i = reg_features @ weights
            opt_shift = find_optimal_discrete_shift(np.array(pred_i[:len(index)].detach()),np.array(reg_target[:len(index)].detach()))
            pred_i[:len(index)] = shifted_ldos_discrete(pred_i[:len(index)], xdos, torch.tensor(opt_shift))
            loss_i = t_get_mse(pred_i, reg_target)
            loss_i.backward()
            return loss_i
        mse_loss = opt.step(closure)

        if mse_loss < best_mse:
            best_state = weights.clone()

    with torch.no_grad():
        final_preds = Features @ best_state 
        final_t_preds = t_Features @ best_state

        opt_shift_train = find_optimal_discrete_shift(np.array(final_preds),np.array(Target))
        final_preds2 = shifted_ldos_discrete(final_preds, xdos, torch.tensor(opt_shift_train))
        opt_shift_test = find_optimal_discrete_shift(np.array(final_t_preds),np.array(t_Target))
        final_t_preds2 = shifted_ldos_discrete(final_t_preds, xdos, torch.tensor(opt_shift_test))

        loss_dos = t_get_rmse(final_preds2, Target, xdos, perc = True)
        test_loss_dos = t_get_rmse(final_t_preds2, t_Target, xdos, perc = True)
        return best_state, loss_dos, test_loss_dos, opt_shift_train, opt_shift_test
        


n_surfaces = 154
n_bulkstructures = 773
n_total_structures = 773 + 154


surface_train_index, surface_test_index = generate_train_test_split(n_surfaces)
bulk_train_index, bulk_test_index = generate_train_test_split(n_bulkstructures)
total_train_index, total_test_index = generate_train_test_split(n_total_structures)
surface_holdout_train_index, surface_holdout_test_index = surface_holdout(n_surfaces)
bulk_biased_train_index, bulk_biased_test_index = generate_biased_train_test_split(n_bulkstructures)
total_biased_train_index, total_biased_test_index = generate_biased_train_test_split(n_total_structures)
holdout_train_index, holdout_test_index = generate_surface_holdout_split(n_total_structures)

with torch.no_grad():
    xdos = torch.load("../../data/xdos.pt")
    y = torch.load("../../data/total_aligned_dos3.pt")
    feats = torch.load("../../data/total_soap.pt")
    index_train, index_test = total_train_index, total_test_index
    regularization_value = 0.01



weights , loss_dos, test_loss_dos, opt_shift_train, opt_shift_test = normal_reg_train_L(feats, y, index_train, index_test,regularization_value, 60, 1) 


torch.save(weights, "./optimal_weights.pt")
torch.save(opt_shift_train, "./opt_shift_train.pt")
torch.save(opt_shift_test, "./opt_shift_test.pt")

results = ["The train error is {}".format(loss_dos), "The test error is {}".format(test_loss_dos)]

w = open("./results.txt", "w")
w.writelines(results)
w.close()