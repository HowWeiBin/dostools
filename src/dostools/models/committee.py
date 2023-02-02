import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from scipy.optimize import minimize

from tqdm import tqdm

from sklearn.model_selection import KFold

from ..loss.loss import t_get_rmse

from . import architectures

import copy

import matplotlib.pyplot as plt

import os
import pickle

class Committee:
    def __init__(self, x, y, train_ratio, xdos, loss, opt, model = "Linear", data_type = "pw", n_com = 8, cv = 2, val = False, reg = None, alpha = None, device = None, val_ratio = 0.1): 
        """
        Committee class is in charge of:
            -Holding the training data
            -Training individual committee models
                *Hyperparameter Tuning
                *Model Fitting
            -Holding every member in the commmittee and the alpha value
            -Committee Prediction
            -Saving & Loading
            -Committee Evaluation
        
        Args:
            x (tensor): Features tensor
            y (tensor): Target tensor
            train_ratio (float): Ratio for train-test split
            xdos (tensor): Xdos tensor, used for loss function
            loss (function): loss function used for training and evaluation
            opt (string): "Adam" or "LBFGS" optimizer
            model (str, optional): Type of model used for committee members
            data_type (str, optional): "pw" or "pc", pointwise or principal components
            n_com (int, optional): Number of committee members
            cv (int, optional): k term for k-fold cross validation to determine optimal hyperparams
            val (bool, optional): Using a validation set during training
            reg (None, optional): Regularization term, float for "pw" data_type and tensor for "pc" data_type
            alpha (None, optional): Alpha value for the committee
            device (None, optional): Device to run computation on
        """
        #Parameters initialization
        if device == None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.x = x
        self.y = y
        self.targetdims = self.y.shape[1] if len(self.y.shape) > 1 else 1
        self.train_ratio = train_ratio
        #Generates training index
        self.test_index, self.train_index = self.generate_train_test_index()
        if data_type == "pw":
            if xdos is not None:
                self.xdos = xdos.to(self.device)
            else:
                self.xdos = None
        else:
            self.xdos = None
        self.loss = loss
        self.opt = opt
        self.model = model
        self.data_type = data_type
        self.n_com = n_com
        self.cv = cv
        if val:
            self.val = val
            self.val_ratio = val_ratio
        else:
            self.val = val
        self.reg = reg
        self.committee = {}
        self.alpha = alpha
        self.committee_loss_history = {}

    def generate_train_test_index(self):
        """
        Generates train_index and test_index, test will always be 20% of the dataset but train can be variable, based on train_ratio
        
        Returns:
            test_index (np.array): test_index
            train_index (np.array): train_index
        """
        
        np.random.seed(0)
        test_split = int(0.8 * self.y.shape[0])
        train_split = int(self.train_ratio * test_split)        
        train_index = np.arange(self.y.shape[0])
        np.random.shuffle(train_index)
        test_index = train_index[test_split:]
        train_index = train_index[:train_split]

        return (test_index, train_index)

    def optimize_hypers(self, max_iter, lr, dir_name, file_name):
        """
        Uses gridsearch to find optimal starting point for scipy optimize minimize
        Then uses scipy optimize minimize to find hypers with least loss
        Plots reg curves and saves output
        Args:
            max_iter (int): Maximum number of iterations for scipy optimize minimize
            lr (float): Learning rate 
            dir_name(string): Path to directory
        """
        #self.reg_grid = torch.tensor([1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 10,100], requires_grad = False)
        self.reg_grid = torch.tensor([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1,100], requires_grad = False)
        
        logreg_values = torch.log(self.reg_grid)

        if self.data_type == "pw":
            min_value, min_index = self.reg_gridsearch(logreg_values, lr)
            print ("Finding optimal list of regularization term using {}-fold CV with init = {}, loss = {}".format(self.cv, logreg_values[min_index.item()], min_value.item()))
            rmin = minimize(self.find_regularization_loss, [logreg_values[min_index.item()]], args = (lr), method = "Nelder-Mead", options = {"maxiter":max_iter})
            self.reg = torch.tensor(np.exp(rmin["x"])[0], requires_grad = False)
            print("Optimization procedure, Success:{} after iterations: {}".format(rmin["success"], rmin["nit"]))
            self.plot_reg_curves(dir_name, file_name)
            print("The optimal regularization term is {}".format(self.reg))

        elif self.data_type == "pc":
            self.reg = torch.zeros(self.targetdims)
            for i in range(self.targetdims):
                self.i = i
                min_value, min_index = self.reg_gridsearch(logreg_values, lr)
                print ("Finding optimal regularization term for PC{} using {}-fold CV with init = {}, loss = {}".format(self.i, self.cv, logreg_values[min_index.item()], min_value.item()))
                rmin = minimize(self.find_regularization_loss, [logreg_values[min_index.item()]], args = (lr), method="Nelder-Mead", options={"maxiter":max_iter})
                self.reg[i] = np.exp(rmin["x"])[0].item()
                print ("Optimization procedure for PC{}, Success:{} after iterations: {}".format(self.i,rmin["success"], rmin["nit"]))
                self.plot_reg_curves(dir_name, file_name+"_pc{}".format(i))
            print("The optimal list of regularization terms is {}".format(self.reg))

    def reg_gridsearch(self, logreg_values, lr):
        """
        Finds optimal regularization value using gridsearch
        
        Args:
            logreg_values (tensor): log-grid values
            lr (float): learning rate
        
        Returns:
            min_value (float): Loss value at optimal regularization value
            min_index (int): Index of optimal regularization value
        """
        self.reg_gridloss = torch.zeros(logreg_values.shape[0], requires_grad = False)
        print ("Performing reg_gridsearch using {}-fold CV".format(self.cv))
        for i, value in enumerate(logreg_values):
            self.reg_gridloss[i] = self.find_regularization_loss(value.item(), lr=lr).item()
        self.reg_gridloss = torch.nan_to_num(self.reg_gridloss, nan=1000, posinf = 1000, neginf = 1000)
        min_value, min_index = torch.min(self.reg_gridloss, dim=0)
        return (min_value, min_index)

    def find_regularization_loss(self, i_regularization, lr):
        """
        Finds the regularization loss for a given regularization term using k-fold CV
        
        Args:
            i_regularization (float): log(regularization) value, to allow for scipy optimize to move in log-space
            lr (float): Learning rate
        
        Returns:
            np.array: regularization loss
        """
        kfold = KFold(n_splits = self.cv, shuffle = False)
        regularization = np.exp(i_regularization)
        print ("Currently trying regularization: {}".format(regularization))
        total_loss = 0
        if self.data_type == "pw":
            if self.targetdims == 1:
                y = self.y[self.train_index,][:,None]
            else:
                y = self.y[self.train_index,:]
        elif self.data_type =="pc":
            y = self.y[self.train_index,self.i][:,None]

        for i_train, i_test in kfold.split(self.x[self.train_index]):
            if self.val: 
                n_train = int(0.9 * len(i_train)) #10% val
                i_val= i_train[n_train:]
                i_train = i_train[:n_train] 
            
            if self.model == "Linear":
                kwargs = {"pin_memory":True} if self.device == "cuda:0" else {}
                train_data = TensorDataset(self.x[self.train_index[i_train],:], y[i_train,])
                traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)
                if self.val:
                    val_data = TensorDataset(self.x[self.train_index[i_val],:], y[i_val,])
                    valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
                else:
                    valdata_loader = None
                if self.data_type == "pc":
                    trialmodel = architectures.LinearModel(self.x.shape[1], 1, self.xdos, reg = regularization, opt = self.opt, device = self.device)
                elif self.data_type == "pw":
                    trialmodel = architectures.LinearModel(self.x.shape[1], y.shape[1], self.xdos, reg= regularization, opt=self.opt, device = self.device)
                if self.opt == "Adam":
                    n_epochs = 1000000
                elif self.opt =="LBFGS":
                    n_epochs = 5000
                _ = trialmodel.fit(traindata_loader, valdata_loader, self.loss, lr = lr, n_epochs = n_epochs)
                total_loss += self.loss(trialmodel(self.x[self.train_index[i_test]].to(self.device)), y[i_test,].to(self.device), self.xdos, perc = True)/self.cv

        print ("The loss at reg {} is {}".format(regularization, total_loss))
        return (total_loss.cpu().detach().numpy())

    def plot_reg_curves(self, dir_name, filename):
        """
        Generates plot based on regularization values obtained via reg_gridsearch and saves it
        
        Args:
            dir_name (string): Name of directory to save to
            filename (string): Name of file to save to
        """
        plt.clf()
        plt.plot(torch.log10(self.reg_grid), self.reg_gridloss)
        plt.xlabel("Regularization value")
        plt.ylabel("Loss")
        plt.title("Regularization curve for {}".format(filename))
        CHECK_FOLDER = os.path.isdir(dir_name)
        if not CHECK_FOLDER:
            os.makedirs(dir_name) 
        plt.savefig(dir_name + '/' + filename + ".png")

        regvalues ={
            "reg_grid": self.reg_grid,
            "reg_gridloss": self.reg_gridloss
        }
        
        f = open(dir_name + "/" + filename + "_regvalues.pkl", "wb")
        pickle.dump(regvalues, f)
        f.close()
        
    def train_evaluate_single_model(self, lr, n_epochs, model_dir_name, model_name, loss_dir_name, loss_name):
        """
        Trains and evaluates a single model to compare with analytical methods
        
        Args:
            lr (float): Learning rate
            n_epochs (int): Max epochs
            model_dir_name (string): Directory to save the model parameters
            model_name (string): Filename to save model
            loss_dir_name (string): Directory to save loss
            loss_name (string): Filename to save loss
        
        Returns:
            TYPE: Description
        """
        total_pred = torch.zeros_like(self.y)
        self.loss_history = {}
        kwargs = {"pin_memory":True} if self.device == "cuda:0" else {}
        if self.model == "Linear":
            if self.val:
                num_train = int(0.9 * len(self.train_index))
                i_val= self.train_index[num_train:]
                i_train = self.train_index[:num_train]
            else:
                i_train = self.train_index
                valdata_loader = None

            if self.data_type == "pw":
                model = architectures.LinearModel(self.x.shape[1], self.targetdims, self.xdos, self.reg, self.opt, self.device)
                if self.targetdims == 1:
                    train_data = TensorDataset(self.x[i_train,:],self.y[i_train,][:,None])  
                else:
                    train_data = TensorDataset(self.x[i_train,:],self.y[i_train,])                                       
                traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)
                if self.val:
                    if self.targetdims == 1:
                        val_data = TensorDataset(self.x[i_val,:],self.y[i_val,:][:,None]) 
                    else:
                        val_data = TensorDataset(self.x[i_val,:],self.y[i_val,:]) 
                    valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
                self.loss_history['model'] = model.fit(traindata_loader, valdata_loader, self.loss, lr = lr, n_epochs = n_epochs)
                total_pred = model(self.x.to(self.device))
                CHECK_FOLDER = os.path.isdir(model_dir_name)
                if not CHECK_FOLDER:
                    os.makedirs(model_dir_name) 
                torch.save(model.state_dict(), (model_dir_name + '/' + model_name + ".pt"))

            elif self.data_type == "pc":
                for pc in range(self.targetdims):
                    train_data = TensorDataset(self.x[i_train,:],self.y[i_train,pc][:,None])                                       
                    traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)
                    if self.val:
                        val_data = TensorDataset(self.x[i_val,:],self.y[i_val,pc][:,None])
                        valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
                    print ('Now training model' + "_pc_" + str(pc))
                    model = architectures.LinearModel(self.x.shape[1], 1, self.xdos, self.reg[pc].item(), self.opt, self.device)
                    self.loss_history['model_pc_{}'.format(pc)] = model.fit(traindata_loader, valdata_loader, self.loss, lr = lr, n_epochs = n_epochs)
                    total_pred[:,pc] = model(self.x.to(self.device))[:,0]
                    CHECK_FOLDER = os.path.isdir(model_dir_name)
                    if not CHECK_FOLDER:
                        os.makedirs(model_dir_name) 
                    torch.save(model.state_dict(), (model_dir_name + '/' + model_name + "_{}".format(pc)+".pt"))
                total_pred = total_pred
            #total pred is done
            CHECK_FOLDER = os.path.isdir(loss_dir_name)
            if not CHECK_FOLDER:
                os.makedirs(loss_dir_name) 
            train_error_p = self.loss(total_pred[self.train_index,:], self.y[self.train_index,:], xdos = self.xdos, perc = True).item()
            test_error_p = self.loss(total_pred[self.test_index,:], self.y[self.test_index,:], xdos = self.xdos, perc = True).item()
            total_error_p = self.loss(total_pred, self.y, xdos = self.xdos, perc = True).item()
            train_error = self.loss(total_pred[self.train_index,:], self.y[self.train_index,:], xdos = self.xdos, perc = False).item()
            test_error = self.loss(total_pred[self.test_index,:], self.y[self.test_index,:], xdos = self.xdos, perc = False).item()
            total_error = self.loss(total_pred, self.y, xdos = self.xdos, perc = False).item()

            RMSES = np.array([train_error_p, test_error_p, total_error_p, train_error, test_error, total_error])
            np.save(loss_dir_name + "/" + loss_name, RMSES)

            return total_pred.cpu()



        

    def train(self, lr, n_epochs):
        """
        Trains each committee member and calculate the alpha value of the committee using the subsampling method
        
        Args:
            lr (float): Learning rate
            n_epochs (int): Max number of epochs
        """
        n_train = self.train_index.shape[0]
        committee_sumresults = torch.zeros(n_train, self.targetdims, requires_grad = False).to(self.device)
        committee_sumsquaredresults = torch.zeros(n_train, self.targetdims, requires_grad = False).to(self.device)
        absent_index = torch.zeros(n_train, requires_grad = False).to(self.device)
        total_index = set(list(range(n_train)))
        random.seed(0)
        kwargs = {"pin_memory":True} if self.device == "cuda:0" else {}
        for i in range(self.n_com):
            subsampled_indices = random.sample(range(n_train), int(n_train/2)) #50% subsampling
            unsampled_indices = list(total_index - set(subsampled_indices))
            absent_index[unsampled_indices] += 1

            if self.model == "Linear":
                if self.val:
                    num_train = int(0.9 * len(subsampled_indices)) #10% val
                    i_val= subsampled_indices[num_train:]
                    i_train = subsampled_indices[:num_train]
                else:
                    i_train = subsampled_indices
                    valdata_loader = None                                  
                if self.data_type == "pw":
                    self.committee['model_' + str(i)] = architectures.LinearModel(self.x.shape[1], self.targetdims, self.xdos, self.reg, self.opt, self.device)
                    if self.targetdims == 1:
                        train_data = TensorDataset(self.x[self.train_index[i_train],:],self.y[self.train_index[i_train],][:,None])  
                    else:
                        train_data = TensorDataset(self.x[self.train_index[i_train],:],self.y[self.train_index[i_train],])                                       
                    traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)
                    if self.val:
                        if self.targetdims == 1:
                            val_data = TensorDataset(self.x[self.train_index[i_val],:],self.y[self.train_index[i_val],:][:,None]) 
                        else:
                            val_data = TensorDataset(self.x[self.train_index[i_val],:],self.y[self.train_index[i_val],:]) 
                        valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
                    self.committee_loss_history['model_' + str(i)] = self.committee['model_' + str(i)].fit(traindata_loader, valdata_loader, self.loss, lr = lr, n_epochs = n_epochs)
                    total_pred = self.committee['model_' + str(i)](self.x[self.train_index,].to(self.device))
                    committee_sumresults[unsampled_indices,:] += total_pred[unsampled_indices,]
                    committee_sumsquaredresults[unsampled_indices,:] += torch.pow(total_pred[unsampled_indices,],2)

                elif self.data_type == "pc":
                    for pc in range(self.targetdims):
                        train_data = TensorDataset(self.x[self.train_index[i_train],:],self.y[self.train_index[i_train],pc][:,None])                                       
                        traindata_loader = DataLoader(train_data, batch_size = len(train_data), shuffle = False, **kwargs)
                        if self.val:
                            val_data = TensorDataset(self.x[self.train_index[i_val],:],self.y[self.train_index[i_val],pc][:,None])
                            valdata_loader = DataLoader(val_data, batch_size = len(val_data), shuffle = False, **kwargs)
                        print ("Training " + 'model_' + str(i) + "_pc_" + str(pc))
                        self.committee['model_' + str(i) + "_pc_" + str(pc)] = architectures.LinearModel(self.x.shape[1], 1, self.xdos, self.reg[pc].item(), self.opt, self.device)
                        self.committee_loss_history['model_' + str(i) + "_pc_" + str(pc)] = self.committee['model_' + str(i) + "_pc_" + str(pc)].fit(traindata_loader, valdata_loader, self.loss, lr = lr, n_epochs = n_epochs)
                        total_pc_pred = self.committee['model_' + str(i) + "_pc_" + str(pc)](self.x[self.train_index,:].to(self.device))
                        committee_sumresults[unsampled_indices,pc] += total_pc_pred[unsampled_indices,0]
                        committee_sumsquaredresults[unsampled_indices,pc] += torch.pow(total_pc_pred[unsampled_indices,0],2)
            elif self.model == "DNN":
                pass
        selected_index = torch.where(absent_index > 4)[0].cpu()
        print ("The number of selected_index is {}".format(selected_index.shape[0]))
        y_best = committee_sumresults[selected_index,:] / absent_index[selected_index].unsqueeze(1)
        y_err = torch.sqrt(committee_sumsquaredresults[selected_index,:]/absent_index[selected_index].unsqueeze(1) - torch.pow(y_best,2))
        self.alpha = torch.sqrt(((self.n_com-3)/(self.n_com-1)) * torch.mean(torch.pow((y_best - self.y[selected_index,].to(self.device)),2)/torch.pow(y_err,2),0) - (1/self.n_com))
        self.alpha = torch.nan_to_num(self.alpha, nan=1, posinf = 1, neginf = 1)

        return self.committee_loss_history

    def predict(self, test_x):
        """
        Predicts the target based on the features of test_x and applies the alpha correction scaling
        
        Args:
            test_x (tensor): Features of test_x
        
        Returns:
            tensor: predictions scaled by alpha
        """
        with torch.no_grad():
            committee_predictions = torch.zeros(self.n_com, test_x.shape[0], self.targetdims, requires_grad = False).to(self.device)

            if self.data_type == "pw":
                if self.model == "Linear":
                    for i in range(self.n_com):
                        committee_predictions[i,:] = self.committee['model_' + str(i)](test_x)
            elif self.data_type == "pc":
                if self.model == "Linear":
                    for i in range(self.n_com):
                        for pc in range(self.y.shape[1]):
                            committee_predictions[i, :, pc] = self.committee['model_' + str(i) + "_pc_" + str(pc)](test_x)[:,0]

            adjusted_predictions = torch.mean(committee_predictions, 0) + self.alpha * (committee_predictions - torch.mean(committee_predictions, 0))
            return adjusted_predictions

    def save(self, dir_name):
        """
        Saves the committee models and parameters in folder, dir_name. Creates the folder if it doesn't already exist
        
        Args:
            dir_name (String): path to directory
        """
        CHECK_FOLDER = os.path.isdir(dir_name)
        if not CHECK_FOLDER:
            os.makedirs(dir_name) 
        for key, value in self.committee.items():
            torch.save(value.state_dict(), (dir_name + '/' + key + ".pt"))
        params = {
            "x" : self.x[:5],
            "y" : self.y[:5],
            "train_ratio" : self.train_ratio,
            "train_index" : self.train_index,
            "test_index" : self.test_index,
            "xdos" : self.xdos,
            "model": self.model,
            "data_type": self.data_type,
            "n_com": self.n_com,
            "cv": self.cv,
            "reg": self.reg,
            "alpha": self.alpha            
        }
        f = open(dir_name + "/params.pkl", "wb")
        pickle.dump(params, f)
        f.close()

    def load(self, dir_name):
        """
        Loads the committee models from directory
        
        Args:
            dir_name (String): path to directory
        """
        if self.model == "Linear":
            for i in range(self.n_com):
                if self.data_type == "pw":
                    self.committee['model_' + str(i)] = architectures.LinearModel(self.x.shape[1],self.y.shape[1], self.xdos, self.reg)
                    self.committee['model_' + str(i)].load_state_dict(torch.load((dir_name +'model_' + str(i) + ".pt")))
                if self.data_type == "pc":
                    for pc in range(self.targetdims):
                        self.committee['model_' + str(i) + "_pc_" + str(pc)] = architectures.LinearModel(self.x.shape[1],1, self.xdos)
                        self.committee['model_' + str(i) + "_pc_" + str(pc)].load_state_dict(torch.load((dir_name + 'model_' + str(i) + "_pc_" + str(pc)+".pt")))
    
    def evaluate(self, dir_name):
        """
        Evaluates committee performance based on loss function and saves performance to disk
        
        Args:
            dir_name (String): path to directory
        """
        CHECK_FOLDER = os.path.isdir(dir_name)
        if not CHECK_FOLDER:
            os.makedirs(dir_name) 
        with torch.no_grad():
            train_predictions = torch.mean(self.predict(self.x[self.train_index,:].to(self.device)),0).cpu()
            test_predictions = torch.mean(self.predict(self.x[self.test_index,:].to(self.device)),0).cpu()
            total_predictions = torch.mean(self.predict(self.x.to(self.device)),0).cpu()

            train_rmse_p =  self.loss(train_predictions, self.y[self.train_index,:].to(self.device), self.xdos, perc = True).item()
            test_rmse_p = self.loss(test_predictions, self.y[self.test_index,:].to(self.device), self.xdos, perc = True).item()
            total_rmse_p = self.loss(total_predictions, self.y.to(self.device), self.xdos, perc = True).item()

            train_rmse =  self.loss(train_predictions, self.y[self.train_index,:].to(self.device), self.xdos, perc = False).item()
            test_rmse = self.loss(test_predictions, self.y[self.test_index,:].to(self.device), self.xdos, perc = False).item()
            total_rmse = self.loss(total_predictions, self.y.to(self.device), self.xdos, perc = False).item()

            RMSES = np.array([train_error_p, test_error_p, total_error_p, train_error, test_error, total_error])
            np.save(loss_dir_name + "/" + loss_name, RMSES)

            return RMSES


'''
class LinearModel(nn.Module):
    def __init__(self, inputSize, outputSize, xdos, reg, opt, device):
        """
        LinearModel implements a neural network with 0 hidden layers, no activation functions, with linear transformation of the data
        i.e. a linear model 
        
        Args:
            inputSize (int): Dimensions of input features
            outputSize (int): Dimensions of output
            xdos (tensor): xdos for loss 
            reg (float): Regularization value
            opt (string): Type of optimizer to use
            device (string): "cuda:0" or "cpu" for the device
        """
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize, bias = False)
        self.xdos = xdos
        self.opt = opt
        self.device = device
        self.reg = torch.tensor(reg, requires_grad = False).to(self.device)       
        self.to(self.device)

    def forward(self, x):
        """
        Performs the transformations to the features based on the model
        
        Args:
            x (tensor): input features
        
        Returns:
            tensor: output
        """
        out = self.linear(x)
        return out

    def fit(self, traindata_loader, valdata_loader, loss, lr ,n_epochs):
        """
        Fits the model based on the training data, early stopping is based on performance on training data (or validation data)
        Returns the loss history 
        
        Args:
            traindata_loader (DataLoader): Train dataloader
            valdata_loader (DataLoader): Validation dataloader
            loss (function): Loss function
            lr (float): Learning rate
            n_epochs (int): Max number of epochs
        
        Returns:
            list: Loss history of the training process
        """
        if self.opt == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = self.reg.item())
            if valdata_loader is not None:
                threshold = 20000
                scheduler_threshold = 20000
            else:
                threshold = 2000
                scheduler_threshold = 2000
            tol = 1e-2
        if self.opt == "LBFGS":
            opt = torch.optim.LBFGS(self.parameters(), lr = lr)
            if valdata_loader is not None:
                threshold = 2000
                scheduler_threshold = 2000
            else:
                threshold = 30
                scheduler_threshold = 30
            tol = 1e-2
        scheduler = torch.optim.lr_scheduler.StepLR(opt, scheduler_threshold, gamma = 0.5)
        best_state = copy.deepcopy(self.state_dict())
        lowest_loss = torch.tensor(9999)
        pred_loss = torch.tensor(0)
        trigger = 0
        loss_history =[]
        pbar = tqdm(range(n_epochs))
        
        for epoch in pbar:
            pbar.set_description(f"Epoch: {epoch}")
            if valdata_loader is not None:
                pbar.set_postfix(val_loss = lowest_loss.item(), trigger = trigger, train_loss = pred_loss.item())
            else:
                pbar.set_postfix(pred_loss = pred_loss.item(), lowest_loss = lowest_loss.item(), trigger = trigger)

            for x_data, y_data in traindata_loader:
                opt.zero_grad()
                x_data, y_data = x_data.to(self.device), y_data.to(self.device)
                if self.opt == "LBFGS":
                    def closure(predictions = False):
                        """
                        Function is necessary for LBFGS, returns the total loss of the model
                        
                        Args:
                            predictions (bool, optional): Returns prediction loss if true, returns total loss if False
                        
                        Returns:
                            tensor: Loss
                        """
                        opt.zero_grad()
                        _pred = self.forward(x_data)
                        _pred_loss = loss(_pred, y_data, self.xdos, perc = True)       
                        _pred_loss = torch.nan_to_num(_pred_loss, nan=lowest_loss.item(), posinf = lowest_loss.item(), neginf = lowest_loss.item())                 
                        _reg_loss = torch.sum(torch.pow(self.linear.weight,2))
                        _reg_loss *= self.reg.item()
                        _new_loss = _pred_loss + _reg_loss
                        _new_loss.backward()
                        if predictions:
                            return _pred_loss
                        return _new_loss
                    opt.step(closure)
                    with torch.no_grad():
                        pred = self.forward(x_data)
                        pred_loss = loss(pred, y_data, self.xdos, perc = True)
                        reg_loss = torch.sum(torch.pow(self.linear.weight,2))
                        reg_loss *= self.reg.item()
                        new_loss = pred_loss + reg_loss
                    if pred_loss >100000 or (pred_loss.isnan().any()) :
                        print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                        self.load_state_dict(best_state)
                        opt = torch.optim.LBFGS(self.parameters(), lr = lr)
                    if epoch %10 == 1:
                        loss_history.append(lowest_loss.item())
                elif self.opt == "Adam":
                    pred = self.forward(x_data)
                    pred_loss = loss(pred, y_data, self.xdos, perc = True)
                    new_loss = pred_loss
                    pred_loss.backward()
                    opt.step()
                    if pred_loss >100000 or (pred_loss.isnan().any()) :
                        print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                        self.load_state_dict(best_state)
                        opt = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = self.reg.item())
                    if epoch %1000 == 1:
                        loss_history.append(lowest_loss.item())

            with torch.no_grad():
                if valdata_loader is not None:
                    new_loss = torch.zeros(1, requires_grad = False).to(self.device)
                    for x_val, y_val in valdata_loader:
                        x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                        val_pred = self.forward(x_val)
                        new_loss += loss(val_pred, y_val, self.xdos, perc = False)

                if lowest_loss - new_loss > tol: #threshold to stop training
                    best_state = copy.deepcopy(self.state_dict())
                    lowest_loss = new_loss
                    trigger = 0

                else:
                    trigger +=1
                    scheduler.step()
                    if trigger > threshold:
                        self.load_state_dict(best_state)
                        print ("Implemented early stopping with lowest_loss: {}".format(lowest_loss))
                        return loss_history
        return loss_history



class DNN(torch.nn.Module):
    
    def __init__(self, input_dims, L1, L2, target_dims, xdos, dropout, reg, opt, device):
        super(DNN, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.fc1 = torch.nn.Linear(input_dims, L1)
        self.fc2 = torch.nn.Linear(L1, L2)
        self.fc3 = torch.nn.Linear(L2,target_dims)
        self.gelu = torch.nn.GELU()
        self.xdos = xdos
        self.dropout = torch.nn.Dropout(p = dropout)
        self.device = device
        self.reg = torch.tensor(reg).to(self.device)
        self.opt = opt
        self.to(self.device)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gelu(x)        
        x = self.fc3(x)
        x = self.gelu(x)
        
        return x
    
    def fit(self, traindata_loader, valdata_loader, loss, lr ,n_epochs):
        """
        Fits the model based on the training data, early stopping is based on performance on training data (or validation data)
        Returns the loss history 
        
        Args:
            traindata_loader (DataLoader): Train dataloader
            valdata_loader (DataLoader): Validation dataloader
            loss (function): Loss function
            lr (float): Learning rate
            n_epochs (int): Max number of epochs
        
        Returns:
            list: Loss history of the training process
        """
        if self.opt == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = self.reg.item())
            if valdata_loader is not None:
                threshold = 20000
                scheduler_threshold = 20000
            else:
                threshold = 2000
                scheduler_threshold = 2000
            tol = 1e-2
        if self.opt == "LBFGS":
            opt = torch.optim.LBFGS(self.parameters(), lr = lr)
            if valdata_loader is not None:
                threshold = 2000
                scheduler_threshold = 2000
            else:
                threshold = 30
                scheduler_threshold = 30
            tol = 1e-2
        scheduler = torch.optim.lr_scheduler.StepLR(opt, scheduler_threshold, gamma = 0.5)
        best_state = copy.deepcopy(self.state_dict())
        lowest_loss = torch.tensor(9999)
        pred_loss = torch.tensor(0)
        trigger = 0
        loss_history =[]
        pbar = tqdm(range(n_epochs))
        
        for epoch in pbar:
            self.train()
            pbar.set_description(f"Epoch: {epoch}")
            if valdata_loader is not None:
                pbar.set_postfix(val_loss = lowest_loss.item(), trigger = trigger, train_loss = pred_loss.item())
            else:
                pbar.set_postfix(pred_loss = pred_loss.item(), lowest_loss = lowest_loss.item(), trigger = trigger)
            
            if self.opt == "LBFGS":
                def closure():
                    opt.zero_grad()
                    _atomic_results = []
                    for x_data in traindata_loader:
                        x_data = x_data.to(self.device)
                        _pred = self.forward(x_data)
                        _atomic_results.append(_pred)
                    _atomic_results = torch.vstack(_atomic_results)
                    _structure_results = torch.zeros_like(traindata_loader.dataset.y)
                    _structure_results = structure_results.index_add_(0, traindata_loader.dataset.index, _atomic_results)
                    _pred_loss = loss(structure_results, traindata_loader.dataset.y.to(self.device), self.xdos, perc = True)
                    _reg_loss = torch.sum(torch.pow(self.fc1.weight,2)) + torch.sum(torch.pow(self.fc2.weight,2)) + torch.sum(torch.pow(self.fc3.weight,2))
                    _new_loss = _pred_loss + _reg_loss
                    _new_loss.backward()
                    return _new_loss
                opt.step(closure)
                with torch.no_grad():
                    self.eval()
                    atomic_results = []
                    for x in traindata_loader:                        
                        pred = self.forward(x)
                        atomic_results.append(pred)
                    atomic_results = torch.vstack(atomic_results)
                    structure_results = torch.zeros_like(traindata_loader.dataset.y)
                    structure_results = structure_results.index_add_(0, traindata_loader.dataset.index, atomic_results)
                    pred_loss = loss(structure_results, traindata_loader.dataset.y.to(self.device), self.xdos, perc = True)
                    reg_loss = torch.sum(torch.pow(self.fc1.weight,2)) + torch.sum(torch.pow(self.fc2.weight,2)) + torch.sum(torch.pow(self.fc3.weight,2))
                    new_loss = pred_loss + reg_loss
                    if pred_loss >100000 or (pred_loss.isnan().any()) :
                        print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                        self.load_state_dict(best_state)
                        opt = torch.optim.LBFGS(self.parameters(), lr = lr)
                    if epoch %10 == 1:
                        loss_history.append(lowest_loss.item())
            
            elif self.opt == "Adam":
                opt.zero_grad()
                atomic_results = []
                for x_data in traindata_loader:
                    x_data = x_data.to(self.device)
                    pred = self.forward(x_data)
                    atomic_results.append(pred)
                atomic_results = torch.vstack(atomic_results)
                structure_results = torch.zeros_like(traindata_loader.dataset.y)
                structure_results = structure_results.index_add_(0, traindata_loader.dataset.index, atomic_results)
                pred_loss = loss(structure_results, traindata_loader.dataset.y.to(self.device), self.xdos, perc = True)
                new_loss = pred_loss
                pred_loss.backward()
                opt.step()
                if pred_loss >100000 or (pred_loss.isnan().any()) :
                    print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                    self.load_state_dict(best_state)
                    opt = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = self.reg.item())
                if epoch %1000 == 1:
                    loss_history.append(lowest_loss.item())

            with torch.no_grad():
                if valdata_loader is not None:
                    self.eval()
                    with torch.no_grad():
                        new_loss = torch.zeros(1, requires_grad = False).to(self.device)
                        for x_val, y_val in valdata_loader:
                            x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                            val_pred = self.forward(x_val)
                            new_loss += loss(val_pred, y_val, self.xdos, perc = False)

                if lowest_loss - new_loss > tol: #threshold to stop training
                    best_state = copy.deepcopy(self.state_dict())
                    lowest_loss = new_loss
                    trigger = 0

                else:
                    trigger +=1
                    scheduler.step()
                    lr = scheduler.get_last_lr()[0]
                    if trigger > threshold:
                        self.load_state_dict(best_state)
                        print ("Implemented early stopping with lowest_loss: {}".format(lowest_loss))
                        return loss_history
        return loss_history    


'''





    
