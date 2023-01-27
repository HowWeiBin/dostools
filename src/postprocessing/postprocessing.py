import torch
from scipy.optimize import brentq, minimize
from scipy.interpolate import interp1d

def t_get_charge(local_dos, mu, xdos, beta, nel):
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
    return nel - torch.trapezoid(local_dos * t_fd_distribution(xdos, mu, beta), xdos, axis=0)

def t_get_aofd(ldos, mu, xdos, beta):
    """compute the excitaion spectrum of one structure"""
    #takes in one set of ldos
    #assumes ~0K behaviour
    dx = xdos[1] - xdos[0] #Separation between each point, represents the energy of absorption
    xxc = torch.as_tensor(range(len(xdos)))*dx #x axis for absorption, depicts each absorption energy
    lxc = torch.zeros(len(xxc)) #Container for absorption intensity
    for i in range(len(xdos)): #for each point in the dos
        lxc[i] = torch.sum(ldos[:len(xdos)-i] * t_fd_distribution(xdos[:len(xdos)-i], mu, beta) * #Slowly shifts occupied to the right
                              ldos[i:] * (1 - t_fd_distribution(xdos[i:], mu, beta)))          #Slowly shifts unoccupied to the left 
    lxc *= dx #?Why the need to scale? is this "dE"? then shouldnt it be dE * dE'?
    return xxc, lxc 

def t_get_band_energy(dos, mu, xdos, beta):
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
    return torch.trapezoid(dos * xdos * t_fd_distribution(xdos, mu, beta), xdos)

def t_get_dos_fermi(dos, mu, xdos):
    """retrun the DOS value at the Fermi energy for one structure
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        """
    idos = interp1d(xdos.detach().numpy(), dos.detach().numpy()) #gives function that uses interpolation to get new points, since kind is not specified, its a linear interpolation
    dos_fermi = idos(mu) #finds dos value at fermi level
    return torch.tensor(dos_fermi)


def t_getmu(dos, beta, xdos, n=2.):
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
    return brentq(lambda x: t_nelec(dos ,x ,beta, xdos)-n, xdos.min(), xdos.max())

def t_fd_distribution(x, mu, beta):
    """Fermi-Dirac distribution
        INPUTS:
        =======
        x: array energy axis (eV)
        mu: Fermi energy (eV)
        beta: inverse temperature (eV)
        """
    y = (x-mu)*beta
    ey = torch.exp(-torch.abs(y)) #torch.exp(y) can lead to overflow
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
    
        
def t_nelec(dos, mu, beta, xdos):
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
    return torch.trapezoid(dos * t_fd_distribution(xdos, mu, beta), xdos)

def t_build_truncated_dos(basis, coeffs, mean, n_pc=10):
    """ builds an approximate DOS providing the basis elements and coeffs""" 
    return coeffs @ basis[:, :n_pc].T + mean

def t_build_dos_from_CDF(cdf):
    if len(cdf.shape) == 3:
        DOSes = (cdf[:,:,2:] - cdf[:,:,:-2])/2
        DOSes = torch.nn.functional.pad(DOSes, pad = (1,1))
        return DOSes
    elif len(cdf.shape) == 2:
        DOS = (cdf[:,2:] - cdf[:,:-2])/2
        DOS = torch.nn.functional.pad(DOS, pad = (1,1))
        return DOS 