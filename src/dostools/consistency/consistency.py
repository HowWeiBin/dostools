import torch 
import ase
import ase.io as ase_io
import numpy as np

from ..postprocessing import postprocessing 
#For def t_get_charge(local_dos, mu, xdos, beta, nel) and t_getmu(dos, beta, xdos, n=2.)

def PotentialConsistency(ldos_tensor, xdos, structure, max_runs, threshold, step = 1000, QMPotential = None):
    #ldos_tensor is y_best from committee
    #xdos is a torch tensor
    #structure is an ase object
    #max_runs is the maximum number of runs
    #threshold is the convergence threshold
    #QMPotential is the potential grid obtained from QM, if None it builds the potential grid on the fly
    
    #should we have a step_size to minimize ldos shifts?
    
    
    n_atoms = len(structure)
    n_ve = 4 * n_atoms
    T_0 = 200
    beta_0 = 1 / (ase.units.kB * T_0) # inverse temperature
    if QMPotential is None:
        inv_distance_matrix = torch.tensor(structure.get_all_distances(mic = True)) * 10e-10
        inv_distance_matrix = torch.pow(inv_distance_matrix, -1)
        inv_distance_matrix.fill_diagonal_(0)
    q = torch.zeros(n_atoms)
    v = torch.zeros(n_atoms)
    shift = torch.zeros(n_atoms)
    ldos = ldos_tensor.clone()
    check = False
    for run in range(max_runs):
        print ("Now on run {}".format(run))
        #if check:
        #    print ("last_q is {}".format(last_q))
        tdos = torch.sum(ldos, dim = 0)
        mu = postprocessing.t_getmu(tdos, beta_0, xdos, n = n_ve)
        #print ("mu is {}".format(mu))
        for i in range(n_atoms):
            ldos_i = ldos[i]
            q[i] = postprocessing.t_get_charge(ldos_i, mu, xdos, beta_0, 4)
            if QMPotential is not None:
                shift[i] += q[i] * get_potential(i, None, structure, None, QMPotential) 
        if check == False:
            print (q)
            q = q+50000
            print (q)
        v = get_potential(None, q, None, inv_distance_matrix, QMPotential)
        shift = q * v
        print (shift)
        shift *= step
        for i in range(n_atoms):
            ldos[i] = shifted_ldos(ldos_tensor[i], xdos, shift[i]) 
        
        if check:
            difference = torch.sum(torch.abs(q - last_q))/n_atoms
            #print ("the difference is {}".format(difference))
            #print ("q is :{}".format(q))
            #print ("last_q is: {}".format(last_q))
            if torch.equal(q,last_q):
                #print ("q and last_q is the same")
            #if difference == 0:
                #print ("Charges have converged after {} runs".format(run))
                return ldos, q, v, shift
                break
            else:
                #print ("q and last_q are different")
                last_q = q.clone()
                
        else:
            #print ("q is: {}".format(q))
            last_q = q.clone()
            
            check = True
            
    print ("Charges did not converge")
    return ldos, q, v, shift

def get_potential(atom_index, charges, structure, inv_distance_matrix, QMPotential = None):
    #get pairwise distance
    #calculate potential
    if QMPotential:
        print ("Does not work yet")
        position = structure[atom_index] 
        v = QMPotential[x,y,z]
    else:
        v = inv_distance_matrix.float() @ charges.float()* 1.6e-19
        return v

def shifted_ldos(ldos, xdos, shift): 
    xdos_step = xdos[1] - xdos[0]
    shifted_ldos = torch.zeros_like(ldos)
    if len(ldos.shape) > 1:
        xdos_shift = torch.round(shift/xdos_step).int()
        for i in range(len(ldos)):
            if xdos_shift[i] > 0:
                shifted_ldos[i] = torch.nn.functional.pad(ldos[i,:-1*xdos_shift[i]], (xdos_shift[i],0))
            elif xdos_shift[i] < 0:
                shifted_ldos[i] = torch.nn.functional.pad(ldos[i,(-1*xdos_shift[i]):], (0,(-1*xdos_shift[i])))
            else:
                shifted_ldos[i] = ldos[i]
    else:        
        xdos_shift = int(torch.round(shift/xdos_step))
        if xdos_shift > 0:
            shifted_ldos = torch.nn.functional.pad(ldos[:-1*xdos_shift], (xdos_shift,0))
        if xdos_shift < 0:
            shifted_ldos = torch.nn.functional.pad(ldos[(-1*xdos_shift):], (0,(-1*xdos_shift)))
        else:
            shifted_ldos[i] = ldos[i]
    return shifted_ldos
    
    
            
        