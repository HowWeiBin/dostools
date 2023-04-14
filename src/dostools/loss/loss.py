import torch
import numpy as np
import random
from ..consistency import consistency
import tqdm.notebook as tq

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
        
def t_get_shifting_rmse(prediction, true, shift, xdos = None, perc = False):
    if xdos is not None:
        shifted_predictions = consistency.shifted_ldos(prediction, xdos, shift)
        loss = t_get_rmse(shifted_predictions, true, xdos = xdos, perc = perc)
    else:
        loss = t_get_rmse(prediction, true, xdos, perc = perc)
    
    return loss

def t_get_BF_shift_index_mse(prediction, true, shift_range, xdos = None, perc = False):
    
    if xdos is not None:
        if len(prediction.shape) > 1:
            mse = torch.zeros(true.shape[0])
            index = torch.zeros(true.shape[0])
            for i, pred in enumerate((prediction)):
                shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[0],1), xdos, shift_range)
                mse[i], index[i] = torch.min(t_get_each_mse(shifted_preds, true[i].repeat(shift_range.shape[0],1)),0)
            mse = torch.mean(mse, 0)

                   
        else: 
            mse = torch.zeros(1)
            index = torch.zeros(1)
            shifted_preds = consistency.shifted_ldos(prediction.repeat(shift_range.shape[0],1), xdos, shift_range)
            mse, index = torch.min(t_get_each_mse(shifted_preds, true.repeat(shift_range.shape[0],1)),0)
            
        return mse,index 
    
    else:
        raise ValueError("xdos not defined")



def t_get_BF_shift_rmse(prediction, true, shift_range, xdos = None, perc = False):
    if xdos is not None:
        loss = torch.zeros(true.shape[0])
        mean = true.mean(axis = 0)
        std = torch.sqrt(torch.trapezoid((true - mean)**2, xdos, axis=1).mean())
        for i, pred in enumerate(prediction):
            shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[0],1), xdos, shift_range) 
            loss[i] = torch.min(t_get_each_mse(shifted_preds, true[i].repeat(shift_range.shape[0],1), xdos = xdos, perc = False))
        loss = torch.sqrt(torch.mean(loss, 0))
        if perc:
            loss = 100 * loss / std
    else:
        loss = t_get_rmse(prediction, true, xdos, perc = perc)
    
    return loss

def t_get_BF_shift_index_error(prediction, true, shift_range, xdos = None, perc = False):
    if xdos is not None:
        loss = torch.zeros(true.shape[0])
        index = torch.zeros(true.shape[0])
        mean = true.mean(axis = 0)
        std = torch.sqrt(torch.trapezoid((true - mean)**2, xdos, axis=1).mean())
        for i, pred in enumerate((prediction)):
            shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[0],1), xdos, shift_range) 
            loss[i], index[i] = torch.min(t_get_each_mse(shifted_preds, true[i].repeat(shift_range.shape[0],1), xdos = xdos, perc = False),0)
        loss = torch.sqrt(torch.mean(loss, 0))
        if perc:
            loss = 100 * loss / std
    else:
        loss = t_get_rmse(prediction, true, xdos, perc = perc)
    
    return loss, index

def t_get_tail_shift_rmse(prediction, true, xdos = None, perc = False):
    if xdos is not None:
        threshold = 1e-3
        _, pred_align_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_align_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        right_shift = (pred_align_indexes - true_align_indexes)
        xdos_step = xdos[1] - xdos[0]
        shifted_predictions = consistency.shifted_ldos(prediction, xdos, right_shift * xdos_step)
        loss = t_get_rmse(shifted_predictions, true, xdos, perc)
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss

def t_get_jitter_tail_shift_rmse(prediction, true, xdos = None, perc = False):
    if xdos is not None:
        threshold = 1e-3
        _, pred_align_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_align_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        right_shift = (pred_align_indexes - true_align_indexes)
        jitter = torch.randint(0, 50, (right_shift.shape[0],))
        right_shift += jitter
        xdos_step = xdos[1] - xdos[0]
        shifted_predictions = consistency.shifted_ldos(prediction, xdos, right_shift * xdos_step)
        loss = t_get_rmse(shifted_predictions, true, xdos, perc)
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss

def t_get_peak_shift_rmse(prediction, true, xdos = None, perc = False):
    if xdos is not None:
        threshold = 1e-3
        _, pred_align_indexes = torch.max(abs(prediction),1) 
        _, true_align_indexes = torch.max(abs(true),1)
        right_shift = (true_align_indexes - pred_align_indexes)
        xdos_step = xdos[1] - xdos[0]
        shifted_predictions = consistency.shifted_ldos(prediction, xdos, right_shift * xdos_step)
        loss = t_get_rmse(shifted_predictions, true, xdos, perc)
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss

def t_min_fourshifts_shift_rmse(prediction, true, xdos = None, perc = False):
    if xdos is not None:
        loss = torch.zeros(true.shape[0])
        shift = torch.zeros(true.shape[0])
        threshold = 1e-3
        _, pred_maxalign_indexes = torch.max(abs(prediction),1) 
        _, true_maxalign_indexes = torch.max(abs(true),1)
        _, pred_tailalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_tailalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        _, pred_headalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_headalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        xdos_step = xdos[1] - xdos[0]
        max_right_shift = (true_maxalign_indexes - pred_maxalign_indexes) * xdos_step
        tail_right_shift = (pred_tailalign_indexes - true_tailalign_indexes) * xdos_step
        head_right_shift = (true_headalign_indexes - pred_headalign_indexes) * xdos_step
        no_shift = torch.zeros_like(max_right_shift)
        shift_range = torch.vstack([max_right_shift, tail_right_shift, head_right_shift, no_shift]).T
        mean = true.mean(axis = 0)
        std = torch.sqrt(torch.trapezoid((true - mean)**2, xdos, axis=1))
        for i, pred in enumerate((prediction)):
            shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[1],1), xdos, shift_range[i]) 
            loss[i], index = torch.min(t_get_each_rmse(shifted_preds, true[i].repeat(shift_range.shape[1],1), xdos = xdos, perc = perc, std_dev = std[i]),0)
            shift[i] = shift_range[i][index]
        loss = torch.mean(loss, 0)  
           
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss, shift
    
def t_min_four_shift_rmse(prediction, true, xdos = None, perc = False):
    if xdos is not None:
        loss = torch.zeros(true.shape[0])
        threshold = 1e-3
        _, pred_maxalign_indexes = torch.max(abs(prediction),1) 
        _, true_maxalign_indexes = torch.max(abs(true),1)
        _, pred_tailalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_tailalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        _, pred_headalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_headalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        xdos_step = xdos[1] - xdos[0]
        max_right_shift = (true_maxalign_indexes - pred_maxalign_indexes) * xdos_step
        tail_right_shift = (pred_tailalign_indexes - true_tailalign_indexes) * xdos_step
        head_right_shift = (true_headalign_indexes - pred_headalign_indexes) * xdos_step
        no_shift = torch.zeros_like(max_right_shift)
        shift_range = torch.vstack([max_right_shift, tail_right_shift, head_right_shift, no_shift]).T
        mean = true.mean(axis = 0)
        std = torch.sqrt(torch.trapezoid((true - mean)**2, xdos, axis=1))
        for i, pred in enumerate((prediction)):
            shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[1],1), xdos, shift_range[i]) 
            loss[i] = torch.min(t_get_each_rmse(shifted_preds, true[i].repeat(shift_range.shape[1],1), xdos = xdos, perc = perc, std_dev = std[i]))
        loss = torch.mean(loss, 0)  
           
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss    

def t_min_fiveshifts_shift_rmse(prediction, true, xdos = None, perc = False):
    if xdos is not None:
        loss = torch.zeros(true.shape[0])
        shift = torch.zeros(true.shape[0])
        threshold = 1e-3
        _, pred_maxalign_indexes = torch.max(abs(prediction),1) 
        _, true_maxalign_indexes = torch.max(abs(true),1)
        _, pred_tailalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_tailalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        _, pred_headalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_headalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        pred_bodyalign_position = torch.trapezoid(xdos * abs(prediction), xdos, dim = 1)/torch.trapezoid(abs(prediction), xdos, dim = 1)
        true_bodyalign_position = torch.trapezoid(xdos * abs(true), xdos, dim = 1)/torch.trapezoid(abs(true), xdos, dim = 1)    
        xdos_step = xdos[1] - xdos[0]
        max_right_shift = (true_maxalign_indexes - pred_maxalign_indexes) * xdos_step
        tail_right_shift = (pred_tailalign_indexes - true_tailalign_indexes) * xdos_step
        head_right_shift = (true_headalign_indexes - pred_headalign_indexes) * xdos_step
        no_shift = torch.zeros_like(max_right_shift)
        body_right_shift = true_bodyalign_position - pred_bodyalign_position
        shift_range = torch.vstack([max_right_shift, tail_right_shift, head_right_shift, body_right_shift, no_shift]).T
        mean = true.mean(axis = 0)
        std = torch.sqrt(torch.trapezoid((true - mean)**2, xdos, axis=1))
        for i, pred in enumerate((prediction)):
            shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[1],1), xdos, shift_range[i]) 
            loss[i], index = torch.min(t_get_each_rmse(shifted_preds, true[i].repeat(shift_range.shape[1],1), xdos = xdos, perc = perc, std_dev = std[i]),0)
            shift[i] = shift_range[i][index]
        loss = torch.mean(loss, 0)  
           
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss, shift

def t_min_fiveshifts_shift_rmse_sq(prediction, true, xdos = None, perc = False):
    if xdos is not None:
        loss = torch.zeros(true.shape[0])
        shift = torch.zeros(true.shape[0])
        threshold = 1e-3
        _, pred_maxalign_indexes = torch.max(abs(prediction),1) 
        _, true_maxalign_indexes = torch.max(abs(true),1)
        _, pred_tailalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_tailalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        _, pred_headalign_indexes = torch.max(torch.flip(abs(prediction) > threshold, [1]),1) 
        _, true_headalign_indexes = torch.max(torch.flip(abs(true) > threshold, [1]),1)
        pred_bodyalign_position = (torch.trapezoid(xdos * torch.pow(prediction,2), xdos, dim = 1))/(torch.trapezoid(torch.pow(prediction,2), xdos, dim = 1))
        true_bodyalign_position = (torch.trapezoid(xdos * torch.pow(true,2), xdos, dim = 1))/(torch.trapezoid(torch.pow(true,2), xdos, dim = 1))     
        xdos_step = xdos[1] - xdos[0]
        max_right_shift = (true_maxalign_indexes - pred_maxalign_indexes) * xdos_step
        tail_right_shift = (pred_tailalign_indexes - true_tailalign_indexes) * xdos_step
        head_right_shift = (true_headalign_indexes - pred_headalign_indexes) * xdos_step
        no_shift = torch.zeros_like(max_right_shift)
        body_right_shift = true_bodyalign_position - pred_bodyalign_position
        shift_range = torch.vstack([max_right_shift, tail_right_shift, head_right_shift, body_right_shift, no_shift]).T
        mean = true.mean(axis = 0)
        std = torch.sqrt(torch.trapezoid((true - mean)**2, xdos, axis=1))
        for i, pred in enumerate((prediction)):
            shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[1],1), xdos, shift_range[i]) 
            loss[i], index = torch.min(t_get_each_rmse(shifted_preds, true[i].repeat(shift_range.shape[1],1), xdos = xdos, perc = perc, std_dev = std[i]),0)
            shift[i] = shift_range[i][index]
        loss = torch.mean(loss, 0)  
           
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss, shift

def t_get_jitter_loss(prediction, true, shifts, xdos = None, perc = False):
    if xdos is not None:
        loss = torch.zeros(true.shape[0])
        new_shifts = torch.zeros_like(shifts)
        xdos_step = xdos[1] - xdos[0]
        shift_range = torch.vstack([shifts-1, shifts, shifts+1]).T * xdos_step
        if perc:
            mean = true.mean(axis = 0)
            std = torch.sqrt(torch.trapezoid((true - mean)**2, xdos, axis=1)).mean()
        else:
            std = None
        for i, pred in enumerate(prediction):
            shifted_preds = consistency.shifted_ldos(pred.repeat(shift_range.shape[1],1), xdos, shift_range[i])
            loss[i], index = torch.min(t_get_each_rmse(shifted_preds, true[i].repeat(shift_range.shape[1],1), xdos = xdos, perc = perc, std_dev = std),0)
            new_shifts[i] = shift_range[i][index]
        
        loss = torch.mean(loss, 0)  
    else:
        loss = t_get_rmse(prediction, true, xdos, perc)
    
    return loss, new_shifts/xdos_step

def t_get_each_rmse(prediction_array, true, xdos = None, perc = False, std_dev = None):
    if xdos is not None:
        rmse = torch.sqrt(torch.trapezoid((prediction_array - true)**2, xdos, axis=1))
        if not perc:
            return rmse
        else:
            return 100 * rmse / std_dev
            
    else:
        rmse = torch.sqrt((prediction_array - true)**2)
        if not perc: 
            return rmse
        else:
            return (100 * (rmse / b.std(dim = 0,unbiased=True)))

def t_get_each_mse(prediction_array, true, xdos = None, perc = False, std_dev = None):
    if xdos is not None:
        mse = (torch.trapezoid((prediction_array - true)**2, xdos, axis=1))
        if not perc:
            return mse
        else:
            return 100 * mse / std_dev
            
    else:
        mse = ((prediction_array - true)**2).mean(dim = 1)
        if not perc: 
            return mse
        else:
            return (100 * (mse / true.std(dim = 0,unbiased=True)))
