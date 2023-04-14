import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq, minimize
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

def fd_distribution(x, mu, beta):
    """Fermi-Dirac distribution
        INPUTS:
        =======
        x: array energy axis (eV)
        mu: Fermi energy (eV)
        beta: inverse temperature (eV)
        """
    y = (x-mu)*beta
    ey = np.exp(-np.abs(y)) #np.exp(y) can lead to overflow
    if hasattr(x,"__iter__"):
        negs = (y<0)
        pos = (y>=0)
        try:
            y[negs] = 1 / (1+ey[negs])        
            y[pos] = ey[pos] / (1+ey[pos])
        except:
            print (x, negs, pos)
            raise
        return y
    else:
        if y<0: return 1/(1+ey)
        else: return ey/(1+ey)
    
        
def nelec(dos, mu, beta, xdos):
    """ computes the number of electrons from the DOS 
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        beta: inverse temperature
        xdos: array energy axis
        """
    #dos * fd_distribution gives occupation at a particular energy level
    #xdos is the axis
    #trapezoid does riemann sums along axis
    return trapezoid(dos * fd_distribution(xdos, mu, beta), xdos)


def getmu(dos, beta, xdos, n=2.):
    """ computes the Fermi energy of structures based on the DOS 
        INPUTS:
        =======
        dos: array of the DOS
        beta: inverse temperature
        xdos: array energy axis
        n: number of electrons
        """
    #brentq finds the root of a function
    #returns solution when nelec = n (number of electrons in the system)
    #min and max denotes range for the algorithmn to search within
    return brentq(lambda x: nelec(dos ,x ,beta, xdos)-n, xdos.min(), xdos.max())


def get_dos_fermi(dos, mu, xdos):
    """retrun the DOS value at the Fermi energy for one structure
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        """
    idos = interp1d(xdos, dos) #gives function that uses interpolation to get new points, since kind is not specified, its a linear interpolation
    dos_fermi = idos(mu) #finds dos value at fermi level
    return dos_fermi


def get_band_energy(dos, mu, xdos, beta):
    """compute the band energy of one srtucture
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        beta: inverse temperature
        """
    #basically same as nelec but we multiply number of electrons at an energy level with the energy level
    #gets total energy of all electrons
    return trapezoid(dos * xdos * fd_distribution(xdos, mu, beta), xdos)


def get_aofd(ldos, mu, xdos, beta):
    """compute the excitaion spectrum of one structure"""
    #takes in one set of ldos
    #assumes ~0K behaviour
    dx = xdos[1] - xdos[0] #Separation between each point, represents the energy of absorption
    xxc = np.asarray(range(len(xdos)), float)*dx #x axis for absorption, depicts each absorption energy
    lxc = np.zeros(len(xxc)) #Container for absorption intensity
    for i in range(len(xdos)): #for each point in the dos
        lxc[i] = np.sum(ldos[:len(xdos)-i] * fd_distribution(xdos[:len(xdos)-i], mu, beta) * #Slowly shifts occupied to the right
                              ldos[i:] * (1 - fd_distribution(xdos[i:], mu, beta)))          #Slowly shifts unoccupied to the left 
    lxc *= dx #lxc is the product but integral denotes the area 
    return xxc, lxc 


def get_charge(local_dos, mu, xdos, beta, nel):
    """compute the local charges of one srtucture
        INPUTS:
        =======
        local_dos: array of the LDOS
        mu: Fermi energy
        xdos: array energy axis
        beta: inverse temperature
        nel: number of valence electrons
        """
    #works the same as nelec except local dos instead of dos
    return nel - trapezoid(local_dos * fd_distribution(xdos, mu, beta), xdos, axis=1)


def gauss(x):
    #just a gauss function
    return np.exp(-0.5*x**2)


def build_dos(sigma, eeigv, dx, emin, emax, natoms=None, weights=None):
    """build the DOS (per state) knowing the energy resolution required in eV
        works with FHI-aims, needs to be modified for QuantumEspresso
        INPUTS:
        =======
        sigma: Gaussian broadening
        eeigv: list of eigenergies of all the structures
        dx: energy grid spacing
        emin: minimum energy value on the grid
        emax: maximum energy value on the grid
        natoms: array of the number of atoms per structure
        weights: if you are using FHI-aims, keep value equal to None. If you are using QuantumEspresso, provide the the k-point weights. 
        
        OUTPUTS:
        xdos: energy grid
        ldos: array containing the DOS"""
    
    if natoms is None:
        raise Exception("please provide 'natoms' array containing the number of atoms per structure")
        
    beta = 1. / sigma #does nothing

    ndos = int((emax-emin+3) / dx) #number of points
    xdos = np.linspace(emin-1.5, emax+1.5, ndos) # extend the energy grid by 3eV 
    ldos = np.zeros((len(eeigv), ndos))
    
    if weights == None:
        for i in range(len(eeigv)):#for every structure    
            for ei in eeigv[i].flatten():#for every energy level
                iei = int((ei-(emin-1.5))*2/sigma) #does nothing
                ldos[i] += np.exp(-0.5*((xdos[:]-ei)/sigma)**2) #puts a gaussian centered on the energy level
            ldos[i] *= 1/np.sqrt(2*np.pi*sigma**2)/natoms[i]/len(eeigv[i]) #normalize
            
    else:
        for i in range(len(eeigv)):
            for j in range(len(eeigv[i])):
                for ei in eeigv[i][j].flatten():
                    ldos[i,: ] += weights[i][j]*gauss((xdos[:]-ei)/sigma)
            ldos[i] *= 1/np.sqrt(2*np.pi*sigma**2)
    return xdos, ldos


def get_regression_weights(train_target, regularization=1e-3, kMM=[], kNM=[], jitter=1e-9): 
    """get the regression weights.. can be used without the train_model function.. follows the same logic in librascal train_gap_model"""
    KNM = kNM.copy()
    Y = train_target.copy()
    
    KMM = kMM.copy()
    KMM[np.diag_indices_from(KMM)] += jitter
    
    
    nref = len(kMM)
    delta = np.var(train_target) / kMM.trace() / nref
    
    KNM /= regularization / delta
    Y /= regularization / delta
    
    KNM = np.hstack([ KNM, np.ones(len(KNM)).reshape(-1,1)])
    z = np.empty((nref + 1, nref + 1))
    z[:nref,:nref] = KMM
    
    
    K = z + KNM.T @ KNM
    Y = KNM.T @ Y
    
    weights = np.linalg.lstsq(K, Y, rcond=1e-10)[0]#None)[0]
    return weights


def get_rmse(a, b, xdos=None, perc=False): #account for the fact that DOS is continuous but we are training them pointwise
    """ computes  Root Mean Squared Error (RMSE) of array properties (DOS/aofd).
         a=pred, b=target, xdos, perc: if False return RMSE else return %RMSE"""
    
    if xdos is not None:
        rmse = np.sqrt(trapezoid((a - b)**2, xdos, axis=1).mean())
        if not perc:
            return rmse
        else:
            mean = b.mean(axis=0)
            std = np.sqrt(trapezoid((b - mean)**2, xdos, axis=1).mean())
            return 100 * rmse / std
    else:
        rmse = np.sqrt(((a - b)**2).mean())
        if not perc:
            return rmse
        else:
            return 100 * rmse / b.std(ddof=1)
        

def pred_error(i_regularization, train_target, kNM, kMM, cv, train_idx, xdos):
    """helper function for the train_model function"""
    kfold = KFold(n_splits=cv, shuffle=False) #gives indices for train-test split to do Kfold
    regularization = np.exp(i_regularization[0]) 
    
    temp_err = 0.
    for train, test in kfold.split(train_idx):
        w = get_regression_weights(train_target[train], 
                                   kMM=kMM, 
                                   regularization=regularization, 
                                   kNM=kNM[train])
        target_pred = kNM @ w
        temp_err += get_rmse(target_pred[test], train_target[test], xdos, perc=True)
    print ("Loss for regularization term: {} is {}".format(regularization, temp_err/cv))
    return temp_err/cv


def train_model(train_target, kNM=[], kMM=[], cv=2, i_regularization=1e-6, maxiter=8, xdos=None):
    """returns the weights of the trained model
        INPUTS:
        =======
        train_target: DOS of the training set (with or without their mean
        kNM: KNM matrix of the training set
        kMM: kernle matrix of the sparse points
        cv: number of the folds for the cross-validation
        i_regularization: initial guess for the regularizer
        maxiter: number of max iterations for the optimizer
        xdos: energy grid of the DOS"""
    
    train_idx = np.arange(len(train_target))
    rmin = minimize(pred_error, [np.log(i_regularization)], args=(train_target, kNM, kMM, cv, train_idx, xdos), method="Nelder-Mead", options={"maxiter":maxiter})
    print(rmin)
    regularization = np.exp(rmin["x"])[0]
    print(regularization)

    # weights of the model
    print(train_target.shape)
    weights = get_regression_weights(train_target, 
                                 kMM=kMM, 
                                 regularization=regularization,
                                 kNM=kNM)
    return weights


def build_truncated_dos(basis, coeffs, mean, n_pc=10):
    """ builds an approximate DOS providing the basis elements and coeffs""" 
    return coeffs @ basis[:, :n_pc].T + mean


def build_pc(dos, dosmean, n_pc=10):
    """
    n_pc: the number of prinicpal components to keep
    """
   
    #dosmean = dos.mean(axis=0)
    cdos = dos - dosmean
    doscov = (cdos.T @ cdos) / len(dos)
    doseva, doseve = np.linalg.eigh(doscov)
    doseva = np.flip(doseva, axis = 0)
    doseve = np.flip(doseve, axis = 1)     
    print('Variance covered with {} PCs is = {}'.format(n_pc, doseva[:n_pc].sum()/doseva.sum()))
    return doseva, doseve[:, :n_pc]
        
    
def build_coeffs(dos, doseve):
    """ finds basis elements and projection coefs of the DOS 
        INPUTS:
        =======
        dos: DOS of the strcutures, should be centered wrt to training set
        doseve: the principal components
        OUPUTS:
        dosproj: projection coefficients on the retained """
    
    dosproj = dos @ doseve #dot product
    return dosproj


def domse(lsigma, delta, KKa, KKb, kMM, sya, syb, ktrain, ytrain, sparse_jitter=1e-12): 
    """auxiliary function that computes the errors for a certain regularization
    parameter in fun do_krr """
    sigma = np.exp(lsigma[0])
    nref = len(kMM)
    ska = kMM * delta * sigma**2 + KKa
    skb = kMM * delta * sigma**2 + KKb
    wa = np.linalg.solve(ska + np.eye(nref)*sparse_jitter, sya)
    wb = np.linalg.solve(skb + np.eye(nref)*sparse_jitter, syb)
    pab = np.dot(ktrain[ntrain//2:], wa)
    pba = np.dot(ktrain[:ntrain//2], wb)
    mse = np.sum((pab-ytrain[ntrain//2:])**2 + (pba-ytrain[:ntrain//2])**2)/ntrain
    return np.log(mse) #log?


def do_krr(kNM, kMM, itrain, itest, target, s=None, sparse_jitter=1e-12):
    """ performs KRR, if the regularization parameter is provided.
    Otherwise, it finds the optimal parameter using 2-fold cross validation.
    Returns sigma, weights, pred
    sigma is the regularization parameter (or a list of them)"""
    y = target
    ktrain = kNM[itrain]
    if (len(y.shape) > 1):
        ytrain = y[itrain] - y[itrain].mean(axis=0) #for 2d and above
    else:
        ytrain = y[itrain] - y[itrain].mean() #for 1d
        
    if (ytrain.std() != 0): #it ifs 0 it means its all the same?? aka all zeros since the mean is 0.
        delta = np.var(ytrain)/(np.trace(kMM)/nref)
        sya = delta*np.dot(delta*ktrain[:ntrain//2].T,ytrain[:ntrain//2])
        syb = delta*np.dot(delta*ktrain[ntrain//2:].T,ytrain[ntrain//2:])    
        KKa = np.dot(ktrain[:ntrain//2].T,ktrain[:ntrain//2])*delta**2
        KKb = np.dot(ktrain[ntrain//2:].T,ktrain[ntrain//2:])*delta**2

        if (s==None):
            rmin = minimize(domse, [np.log(0.01)], (delta, KKa, KKb, kMM, sya, syb, ktrain, ytrain), #domse is here 
                            method = "Nelder-Mead", options={"maxiter":8})
            sigma = np.exp(rmin["x"])[0]
        else:
            sigma = s

        sparseK = kMM * delta * sigma**2 +  np.dot(kNM[itrain].T,kNM[itrain])*delta**2
        sparseY = delta*np.dot(delta*kNM[itrain].T,ytrain)
        w = np.linalg.solve(sparseK + np.eye(nref)*sparse_jitter*delta, sparseY)
    
    else:
        w = 1.0 * np.zeros((nref))
        sigma = 1e3
     
    if (len(y.shape) > 1):
        ypred = np.dot(kNM,w) + y[itrain].mean(axis=0)
    else:
        ypred = np.dot(kNM, w) + y[itrain].mean()
    return sigma, w, ypred


def do_resampling_pw(kNM, kMM, itrain, itest, target, sigma, sparse_jitter=1e-12): #??
    """ performs an error estimation using resampling/bootstrapping.
    VERY SPECIFIC TO THE PW REPRESENATION OF DOS/DOS AS A SINGLE PROPERTY.
    weights is an array of zeros!!! (TODO)
    returns: yrs, weights, alpha"""
    y = target - target[itrain].mean(axis=0) # zero-center
    nref = len(kMM) #number of reference points
    delta = np.var(y[itrain])/(np.trace(kMM)/nref) #?
    print (delta)
    ypred = np.zeros((nrs, target.shape[0], target.shape[1])) #container for predicted y
    irs = np.zeros((nrs, ntrs), int) #? defined as nrs = 8 and ntrs - int(ntrain*.5) ? nrs = number of committee models? ntrs = ntrain/2 because of 2-fold cross validation
    weights = np.zeros((nrs, nref)) #container for weights
    for j in range(nrs): #for all the models, do training and prediction
        irs[j] = np.random.choice(itrain, ntrs, replace=False)
        sparseK = kMM * delta * sigma**2 +  np.dot(kNM[irs[j]].T,kNM[irs[j]])*delta**2
        sparseY = delta*np.dot(delta*kNM[irs[j]].T,y[irs[j]])
        w = np.linalg.solve(sparseK + np.eye(nref)*sparse_jitter*delta, sparseY) 
        ypred[j] = np.dot(kNM,w) #get predictions per CV
    
    alpha = np.zeros(target.shape[1]) #shape of the target (DOS points)
    print (alpha.shape)
    yrs = np.zeros(ypred.shape) #same thing?
    
    for i in range(target.shape[1]):#for every point on xdos
        rsy = np.zeros(ntot); rsy2 = np.zeros(ntot); rsny = np.zeros(ntot, int); #ntot = total number of samples (1039)
        for j in range(nrs):#for each committee model
            rstest = np.setdiff1d(itrain, irs[j]) #elements in itrain that is not in irs[j], basically the other half of the cross validation
            rsy[rstest]  += ypred[j,rstest, i]
            rsy2[rstest] += ypred[j,rstest, i]**2
            rsny[rstest] += 1
        selstat = np.where(rsny>4)[0] #?
        ybest = rsy[selstat]/rsny[selstat] #shape of n,1
        yerr = np.sqrt(rsy2[selstat]/rsny[selstat] - (rsy[selstat]/rsny[selstat])**2)
        try:
            alpha[i] = np.sqrt(np.mean((ybest - y[selstat, i])**2/yerr**2))
        except:
            print ('there is a problem with index {}'.format(i))
        yrs[:, :, i] = np.mean(ypred[:, :, i],axis=0) + alpha[i]* (ypred[:, :, i]- np.mean(ypred[:, :, i],axis=0))
    yrs +=  target[itrain].mean(axis=0)
    print ('yrs shape is = ', yrs.shape)
    return yrs, weights, alpha

'''
def build_coeffs(dos, itrain, nmax=10): # has both build pc + build coeffs (above)
    """ finds basis elements and projection coefs of the DOS """
    dosmean = dos[itrain].mean(axis=0)
    cdos = dos[itrain] - dosmean
    
    doscov = np.dot(cdos.T, cdos)/ntot
    doseva, doseve = np.linalg.eigh(doscov)
    doseva = np.flip(doseva, axis = 0)
    doseve = np.flip(doseve, axis = 1)
    
    dosproj = np.dot(dos-dosmean,doseve[:,:nmax])
    
    print('Variance covered with {} PCs is = {}'.format(nmax, doseva[:nmax].sum()/doseva.sum()))
    return dosproj, doseva, doseve
'''

def do_krr_all_coeffs(kNM, kMM, itrain, itest, targets, nmax_columns=10):
    """ This is very specific to the coefficients of the basis set of the DOS.
    It performs a KRR action on all of the projection coefficients.
    Returns sigma_array, weights and pred. """
    sigmas_arr = np.zeros(nmax_columns)
    weights = np.zeros((nmax_columns, nref)) #nref = reference PP-GPR points
    targets_pred = np.zeros((nmax_columns, ntot)) #ntot = total number of structures 
    for q in tnrange(nmax_columns):
        sigmas_arr[q], weights[q], targets_pred[q] = do_krr(kNM, kMM, itrain, itest, targets[: ,q])#do krr finds the best regularization constant
        inter_rmse = np.sqrt(((targets_pred.T[itest, q] - targets[itest, q])**2).mean())
        inter_std = targets[itest, q].std()
        print("component {} / {}: sigma = {:{prec}}, RMSE = {:{prec}} /"\
            "std = {:{prec}} = {:{prec}}%".format(q+1, nmax_columns, sigmas_arr[q], inter_rmse,
                                                 inter_std, 100.*inter_rmse/inter_std,
                                                 prec='0.4'))
    return sigmas_arr, weights, targets_pred #gives optimal sigma, weights and targets


def do_resampling_all_coeffs(kNM, kMM, itrain, itest, targets, sigmas_arr, basis, mean, nmax_columns=10, sparse_jitter=1e-12): #need to look at the differences
    y = targets #already zero centered (?)
    ktrain = kNM[itrain]
    ytrain = y[itrain]
    nref = len(kMM) #number of reference points
    irs = np.zeros((nrs, ntrs), int) #? defined as nrs = 8 and ntrs - int(ntrain*.5) ? nrs = number of committee models? ntrs = ntrain/2 [its the subsampling ratio]
    ypred = np.zeros((nrs, nmax_columns, ntot))
    weights = np.zeros((nrs, nmax_columns, nref)) #committee model #, PCA #, nref denotes the shape of the weights 
    for j in range(nrs): #for each committee model 
        irs[j] = np.random.choice(itrain, ntrs, replace=False)
        
        for q in range(nmax_columns): #for each PCA component
            delta = np.var(ytrain[:, q])/(np.trace(kMM)/nref)
            sparseK = kMM * delta * sigmas_arr[q]**2 +  np.dot(kNM[irs[j]].T,kNM[irs[j]])*delta**2
            sparseY = delta*np.dot(delta*kNM[irs[j]].T,y[irs[j], q])
            w = np.linalg.solve(sparseK + np.eye(nref)*sparse_jitter*delta, sparseY)
            ypred[j, q] = np.dot(kNM,w) 
            weights[j, q] = w 
    
    yrs = np.zeros((nrs, nmax_columns, ntot)) #container for committee results 
    alphas = np.zeros(nmax_columns)
    for q in range(nmax_columns):
        rsy = np.zeros(ntot); rsy2 = np.zeros(ntot); rsny = np.zeros(ntot, int);
        for j in range(nrs):
            rstest = np.setdiff1d(itrain, irs[j])
            rsy[rstest] += ypred[j, q, rstest]
            rsy2[rstest] += ypred[j, q, rstest]**2
            rsny[rstest] += 1
        selstat = np.where(rsny>4)[0]
        ybest = rsy[selstat]/rsny[selstat]
        yerr = np.sqrt(rsy2[selstat]/rsny[selstat] - (rsy[selstat]/rsny[selstat])**2)
        alphas[q] = np.sqrt(np.mean((ybest - y[selstat, q])**2/yerr**2))
        yrs[:, q] = np.mean(ypred[:, q],axis=0) + alphas[q]* (ypred[:, q]- np.mean(ypred[:, q],axis=0))
    return np.einsum('jki, xk -> ijx', yrs, basis[:, :nmax_columns]) + mean, yrs, weights, alphas #first term regenerates the dos

'''   
def build_truncated_dos(basis, coeffs, mean, nmax_columns=10):
    """ builds an approximate DOS providing the basis elements and coeffs""" 
    return np.dot(coeffs, basis[:, :nmax_columns].T) + mean
'''