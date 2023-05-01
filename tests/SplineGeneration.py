#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
from scipy.interpolate import CubicHermiteSpline
import dostools.datasets.data as data
import numpy as np
from tqdm import tqdm
<<<<<<< HEAD
import ase
import json
import ase.io
with torch.no_grad():
    sigma = 0.3
    structures = ase.io.read("./surfaces.xyz", ":")
=======

with torch.no_grad():
    sigma = 0.3
    structures = data.load_structures(":")
>>>>>>> d22a6c4f71ffcfe88e87f8f7402568f1c6e91048
    n_structures = len(structures) #total number of structures
    for structure in structures:#implement periodicity
        structure.wrap(eps = 1e-12) 
    n_atoms = np.zeros(n_structures, dtype = int) #stores number of atoms in each structures
    for i in range(n_structures):
        n_atoms[i] = len(structures[i])


<<<<<<< HEAD
    #eigen_energies, emin, emax = data.load_eigenenergies(unpack = True, n_structures = n_structures)
    #full_eigen_energies = [torch.tensor(i.flatten()) for i in eigen_energies]
    #eigenenergy_length = [len(i) for i in full_eigen_energies]
    #eigenenergy_length_t = torch.tensor(eigenenergy_length)
    #normalization_eiglength = [len(i) for i in eigen_energies]
    #normalization_eiglength_t = torch.tensor(normalization_eiglength)
    #normalization = 1/torch.sqrt(2*torch.tensor(np.pi)*sigma**2)/n_atoms/normalization_eiglength_t
    #normalization_quartic = 1/n_atoms/normalization_eiglength_t
    f = open("./eigenenergies.json")
    surfaces_eigen_energies = json.load(f)
    f = open("./vacuum_reference.json")
    vacuum_reference = json.load(f)

    t_surfaces_eigen_energies = [torch.tensor(surfaces_eigen_energies[i]) for i in surfaces_eigen_energies]
    k_normalization = torch.tensor([len(i) for i in t_surfaces_eigen_energies])
    t_surfaces_eigen_energies = [i[i>-70] for i in t_surfaces_eigen_energies]
    normalization_surface = 1/torch.sqrt(2*torch.tensor(np.pi)*sigma**2)/n_atoms/k_normalization
=======
    eigen_energies, emin, emax = data.load_eigenenergies(unpack = True, n_structures = n_structures)
    full_eigen_energies = [torch.tensor(i.flatten()) for i in eigen_energies]
    eigenenergy_length = [len(i) for i in full_eigen_energies]
    eigenenergy_length_t = torch.tensor(eigenenergy_length)
    normalization_eiglength = [len(i) for i in eigen_energies]
    normalization_eiglength_t = torch.tensor(normalization_eiglength)
    normalization = 1/torch.sqrt(2*torch.tensor(np.pi)*sigma**2)/n_atoms/normalization_eiglength_t
    normalization_quartic = 1/n_atoms/normalization_eiglength_t


>>>>>>> d22a6c4f71ffcfe88e87f8f7402568f1c6e91048
lower_bound = -24.553719539983-4.5
#upper_bound = 11.346414696331+3
upper_bound = -4.3758 + 5.5
xaxis = torch.arange(lower_bound,upper_bound,0.001)
total_coefs = []
<<<<<<< HEAD
for i in tqdm(range(n_structures)):
    def value_fn(x):
        l_dos_E = torch.sum(torch.exp(-0.5*((x - t_surfaces_eigen_energies[i].view(-1,1))/sigma)**2), dim = 0) * 2 * normalization_surface[i]
        return l_dos_E
    def derivative_fn(x):
        dfn_E = torch.sum(torch.exp(-0.5*((x - t_surfaces_eigen_energies[i].view(-1,1))/sigma)**2) *
                          (-1 * ((x - t_surfaces_eigen_energies[i].view(-1,1))/sigma)**2), dim =0) * 2 * normalization_surface[i]
=======
for i in tqdm(range(1039)):
    def value_fn(x):
        l_dos_E = torch.sum(torch.exp(-0.5*((x - full_eigen_energies[i].view(-1,1))/sigma)**2), dim = 0) * 2 * normalization[i]
        return l_dos_E
    def derivative_fn(x):
        dfn_E = torch.sum(torch.exp(-0.5*((x - full_eigen_energies[i].view(-1,1))/sigma)**2) *
                          (-1 * ((x - full_eigen_energies[i].view(-1,1))/sigma)**2), dim =0) * 2 * normalization[i]
>>>>>>> d22a6c4f71ffcfe88e87f8f7402568f1c6e91048
        return dfn_E

    spliner = CubicHermiteSpline(xaxis, value_fn(xaxis), derivative_fn(xaxis))
    total_coefs.append(torch.tensor(spliner.c))
    
a = torch.stack(total_coefs)
<<<<<<< HEAD
torch.save(a, "./Splines_Surfaces_03smear.pt")
=======
torch.save(a, "./Splines_dataset4.pt")
>>>>>>> d22a6c4f71ffcfe88e87f8f7402568f1c6e91048


# In[ ]:




