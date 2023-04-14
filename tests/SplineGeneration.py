#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
from scipy.interpolate import CubicHermiteSpline
import dostools.datasets.data as data
import numpy as np
from tqdm import tqdm

with torch.no_grad():
    sigma = 0.3
    structures = data.load_structures(":")
    n_structures = len(structures) #total number of structures
    for structure in structures:#implement periodicity
        structure.wrap(eps = 1e-12) 
    n_atoms = np.zeros(n_structures, dtype = int) #stores number of atoms in each structures
    for i in range(n_structures):
        n_atoms[i] = len(structures[i])


    eigen_energies, emin, emax = data.load_eigenenergies(unpack = True, n_structures = n_structures)
    full_eigen_energies = [torch.tensor(i.flatten()) for i in eigen_energies]
    eigenenergy_length = [len(i) for i in full_eigen_energies]
    eigenenergy_length_t = torch.tensor(eigenenergy_length)
    normalization_eiglength = [len(i) for i in eigen_energies]
    normalization_eiglength_t = torch.tensor(normalization_eiglength)
    normalization = 1/torch.sqrt(2*torch.tensor(np.pi)*sigma**2)/n_atoms/normalization_eiglength_t
    normalization_quartic = 1/n_atoms/normalization_eiglength_t


lower_bound = -24.553719539983-4.5
#upper_bound = 11.346414696331+3
upper_bound = -4.3758 + 5.5
xaxis = torch.arange(lower_bound,upper_bound,0.001)
total_coefs = []
for i in tqdm(range(1039)):
    def value_fn(x):
        l_dos_E = torch.sum(torch.exp(-0.5*((x - full_eigen_energies[i].view(-1,1))/sigma)**2), dim = 0) * 2 * normalization[i]
        return l_dos_E
    def derivative_fn(x):
        dfn_E = torch.sum(torch.exp(-0.5*((x - full_eigen_energies[i].view(-1,1))/sigma)**2) *
                          (-1 * ((x - full_eigen_energies[i].view(-1,1))/sigma)**2), dim =0) * 2 * normalization[i]
        return dfn_E

    spliner = CubicHermiteSpline(xaxis, value_fn(xaxis), derivative_fn(xaxis))
    total_coefs.append(torch.tensor(spliner.c))
    
a = torch.stack(total_coefs)
torch.save(a, "./Splines_dataset4.pt")


# In[ ]:




