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
        #x = self.gelu(x)
        
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
                threshold = 2000
                scheduler_threshold = 2000
            else:
                threshold = 200
                scheduler_threshold = 20
            tol = 1e-2
        if self.opt == "LBFGS":
            opt = torch.optim.LBFGS(self.parameters(), lr = lr)
            if valdata_loader is not None:
                threshold = 2000
                scheduler_threshold = 2000
            else:
                threshold = 30
                scheduler_threshold = 25
            tol = 1e-2
        scheduler = torch.optim.lr_scheduler.StepLR(opt, scheduler_threshold, gamma = 0.5)
        best_state = copy.deepcopy(self.state_dict())
        lowest_loss = torch.tensor(9999)
        pred_loss = torch.tensor(0)
        trigger = 0
        loss_history =[]
        pbar = range(n_epochs)
        fig, ax_list = plt.subplots(4,1)
        lines = []
        for i in range(4):
            line, = ax_list[i].plot(self.xdos.cpu(), traindata_loader.dataset.y[i], label = "Prediction")
            lines.append(line)
            ax_list[i].plot(self.xdos.cpu(), traindata_loader.dataset.y[i], label = "True")
        ax_list[3].legend()
        for epoch in tqdm(pbar):
            self.train()
          
            if self.opt == "LBFGS":
                def closure():
                    opt.zero_grad()
                    _atomic_results = []
                    for x_data in traindata_loader:
                        x_data = x_data.to(self.device).double()
                        _pred = self.forward(x_data)
                        _atomic_results.append(_pred)
                    _atomic_results = torch.vstack(_atomic_results).to(self.device)
                    _structure_results = torch.zeros_like(traindata_loader.dataset.y).to(self.device)
                    _structure_results = _structure_results.index_add_(0, traindata_loader.dataset.index.to(self.device), _atomic_results)
                    _pred_loss = loss(_structure_results, traindata_loader.dataset.y.to(self.device), self.xdos, perc = True)
                    #_reg_loss = torch.sum(torch.pow(self.fc1.weight,2)) + torch.sum(torch.pow(self.fc2.weight,2)) + torch.sum(torch.pow(self.fc3.weight,2))
                    _new_loss = _pred_loss #+ _reg_loss
                    _new_loss.backward()
                    return _new_loss
                opt.step(closure)
                with torch.no_grad():
                    self.eval()
                    atomic_results = []
                    for x in traindata_loader:
                        x = x.to(self.device).double()
                        pred = self.forward(x)
                        atomic_results.append(pred)
                    atomic_results = torch.vstack(atomic_results).to(self.device)
                    structure_results = torch.zeros_like(traindata_loader.dataset.y).to(self.device)
                    structure_results = structure_results.index_add_(0, traindata_loader.dataset.index.to(self.device), atomic_results)
                    pred_loss = loss(structure_results, traindata_loader.dataset.y.to(self.device), self.xdos, perc = True)
                    #reg_loss = torch.sum(torch.pow(self.fc1.weight,2)) + torch.sum(torch.pow(self.fc2.weight,2)) + torch.sum(torch.pow(self.fc3.weight,2))
                    new_loss = pred_loss# + reg_loss
                    if pred_loss >100000 or (pred_loss.isnan().any()) :
                        print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                        self.load_state_dict(best_state)
                        opt = torch.optim.LBFGS(self.parameters(), lr = lr)
                    if epoch %10 == 1:
                        loss_history.append(lowest_loss.item())
                    #Updating Figures here
                    for i in range(4):
                        lines[i].set_ydata(structure_results[i].cpu())
                    
                    ax_list[0].set_title("Epochs: {}, Pred loss: {}, Lowest loss:{}, Trigger: {}".format(epoch, pred_loss.item(), lowest_loss.item(), trigger))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.1)
                    print ("Done one epoch")
            elif self.opt == "Adam":
                opt.zero_grad()
                atomic_results = []
                for x_data in traindata_loader:
                    x_data = x_data.double()
                    pred = self.forward(x_data)
                    atomic_results.append(pred)
                atomic_results = torch.vstack(atomic_results).to(self.device)
                structure_results = torch.zeros_like(traindata_loader.dataset.y).to(self.device)
                structure_results = structure_results.index_add_(0, traindata_loader.dataset.index.to(self.device), atomic_results)
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
                plt.clf()
                plt.plot(self.xdos.cpu(), structure_results[0].detach().cpu(), label = "Prediction")
                plt.plot(self.xdos.cpu(), traindata_loader.dataset.y[0], label = "True")
                plt.legend()
                plt.draw()
                plt.pause(0.01)
                
                
#                 for i in range(4):
#                     ax_list[i].clear()
#                     ax_list[i].plot(self.xdos.cpu(), structure_results[i].detach().cpu(), label = "Prediction")
#                     ax_list[i].plot(self.xdos.cpu(), traindata_loader.dataset.y[i], label = "True")
#                 ax_list[3].legend()
#                 ax_list[0].set_title("Epochs: {}, Pred loss: {}, Lowest loss:{}, Trigger: {}".format(epoch, pred_loss.item(), lowest_loss.item(), trigger))
#                 fig.show()
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


importlib.reload(loss)
importlib.reload(consistency)
importlib.reload(postprocessing)

class Alignment_LinearModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize, train_size, xdos, reg, opt, device):
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
        super(Alignment_LinearModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias = False)
        self.xdos = xdos
        self.opt = opt
        self.device = device
        self.reg = torch.tensor(reg, requires_grad = False).to(self.device)       
        #self.alignment = [torch.ones(train_size, device = self.device, requires_grad = True)]
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
            #opt2 = torch.optim.Adam(self.alignment, lr = 10, weight_decay = 0)
            threshold = 2000
            scheduler_threshold = 2000
            tol = 1e-2
        if self.opt == "LBFGS":
            opt = torch.optim.LBFGS(self.parameters(), lr = lr)
            #opt = torch.optim.LBFGS(list(self.parameters()) + self.alignment, lr = lr)
            threshold = 50
            scheduler_threshold = 30
            tol = 1e-2
        scheduler = torch.optim.lr_scheduler.StepLR(opt, scheduler_threshold, gamma = 0.5)
        best_state = copy.deepcopy(self.state_dict())
        lowest_loss = torch.tensor(9999)
        pred_loss = torch.tensor(0)
        trigger = 0
        loss_history =[]
        pbar = range(n_epochs)
        #indexes = np.array([369, 341, 745, 521, 278, 5, 193, 37])
        indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        fig, ax_list = plt.subplots(4,2, sharex = True)
        ax_list = ax_list.flatten()
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace = 0)
        lines = []
        for i in range(4 * 2):
            target_index = indexes[i]
            line, = ax_list[i].plot(self.xdos.cpu(), traindata_loader.dataset.y[target_index], label = "Prediction", alpha = 0.8)
            lines.append(line)
            line, = ax_list[i].plot(self.xdos.cpu(), traindata_loader.dataset.y[target_index], label = "Shifted Prediction", alpha = 0.8)
            lines.append(line)
            ax_list[i].plot(self.xdos.cpu(), traindata_loader.dataset.y[target_index], label = "True")
        fig.legend(loc = "lower center", labels = ["Prediction", "Shifted Prediction", "True"], bbox_to_anchor = (0.5, 0), ncol = 3, fancybox = True)
        for epoch in pbar:
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
                        _pred_loss, shift_values = loss(_pred, y_data, self.xdos, perc = True)       
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
                        pred_loss, shift_values = loss(pred, y_data, self.xdos, perc = True)
                        shifted_preds = consistency.shifted_ldos(pred, self.xdos, shift_values)
                        reg_loss = torch.sum(torch.pow(self.linear.weight,2))
                        reg_loss *= self.reg.item()
                        new_loss = pred_loss + reg_loss
                    if pred_loss >100000 or (pred_loss.isnan().any()) :
                        print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                        self.load_state_dict(best_state)
                        opt = torch.optim.LBFGS(self.parameters(), lr = lr)
                    if epoch %10 == 1:
                        loss_history.append(lowest_loss.item())
                    #Updating Figures here
                    for i in range(4 * 2):
                        index = 2*i
                        target_index = indexes[i]
                        lines[index].set_ydata(pred[target_index].cpu())
                        lines[index+1].set_ydata(shifted_preds[target_index].cpu())
                        
                    fig.suptitle("Epochs: {}, Pred loss: {}, Lowest loss:{}, Trigger: {}".format(epoch, pred_loss.item(), lowest_loss.item(), trigger))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.1)
                elif self.opt == "Adam":
                    pred = self.forward(x_data)
                    pred_loss, shift_values = loss(pred, y_data, self.xdos, perc = True)
                    shifted_preds = consistency.shifted_ldos(pred, self.xdos, shift_values)
                    new_loss = pred_loss
                    pred_loss.backward()
                    opt.step()
                    #opt2.step()
                    if epoch %1000 == 1:
                        loss_history.append(lowest_loss.item())
                    for i in range(4):
                        index = i * 2
                        lines[index].set_ydata(pred[i].detach().cpu())
                        lines[index+1].set_ydata(shifted_preds[i].detach().cpu())
                    ax_list[0].set_title("Epochs: {}, Pred loss: {}, Lowest loss:{}, Trigger: {}".format(epoch, pred_loss.item(), lowest_loss.item(), trigger), loc = 'right', )
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.1)

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