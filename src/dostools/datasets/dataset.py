import rascaline
import numpy as np
from equistore import TensorBlock, TensorMap, Labels
import torch
from torch.utils.data import Dataset
from ..utils.utils import *
from skcosmo.feature_selection import FPS

class TensorFeatures:
    def __init__(self, structures, descriptor = "soap", HYPERS = None):
        self.structures = structures
        self.descriptor = descriptor
        self.HYPERS = HYPERS
        
        if not self.HYPERS:
            print ("Please define hyperparameters")
            return None
        
        if self.descriptor == "soap":
            calculator = rascaline.SoapPowerSpectrum(**self.HYPERS)
            descriptors = calculator.compute(self.structures)   
            descriptors.keys_to_samples("species_center")
            descriptors.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])
            
        else:
            print ("Other descriptors not implemented yet")
        
        self.extract_dataset(descriptors)
    
    def extract_dataset(self, descriptors):
        n_refs = 500
        n_atoms = descriptors.block(0).values.shape[0]
        n_structures = np.unique(descriptors.block(0).samples["structure"])
        self.Features = {}
        self.Features["structure_sumdescriptors"] = torch.zeros(len(n_structures), descriptors.block(0).values.shape[1])
        self.Features["structure_avedescriptors"] = torch.zeros(len(n_structures), descriptors.block(0).values.shape[1])
        self.Features["structure_sumkerneldescriptors"] = torch.zeros(len(n_structures), n_refs)
        self.Features["structure_avekerneldescriptors"] = torch.zeros(len(n_structures), n_refs)
        self.Features["structure_descriptors"] = []
        self.Features["structure_kerneldescriptors"] = []
        self.Features["atom_descriptors"] = torch.tensor(descriptors.block(0).values)
        self.Features['atom_descriptors'] = torch.nn.functional.normalize(self.Features["atom_descriptors"], dim = 1)
        selector = FPS(n_to_select = n_refs,
               progress_bar = True,
               score_threshold = 1e-12,
               full = False,
               initialize = 0
              )
        selector.fit(self.Features['atom_descriptors'].T)
        references = selector.transform(self.Features['atom_descriptors'].T).T
        self.Features['atomkernel_descriptors'] = torch.pow(self.Features['atom_descriptors'] @ references.T, 2) #25k, 1000
        #self.atomkernel_descriptors = torch.nn.functional.normalize(self.atomkernel_descriptors, dim = 1)
        #Computing sum_structure
        for structure_i in n_structures:
            a_i = descriptors.block(0).samples["structure"] == structure_i #find atoms in i
            self.Features["structure_sumdescriptors"][structure_i, :] = torch.tensor(np.sum(descriptors.block(0).values[a_i, :], axis = 0))
            self.Features["structure_avedescriptors"][structure_i, :] = torch.tensor(np.sum(descriptors.block(0).values[a_i, :], axis = 0))/np.sum(a_i)
            self.Features["structure_descriptors"].append(torch.tensor(descriptors.block(0).values[a_i,:]).float())
            self.Features["structure_sumkerneldescriptors"][structure_i, :] = torch.sum(self.Features["atomkernel_descriptors"][a_i, :], axis = 0)
            self.Features["structure_avekerneldescriptors"][structure_i, :] = torch.sum(self.Features["atomkernel_descriptors"][a_i, :], axis = 0)/np.sum(a_i)
            self.Features["structure_kerneldescriptors"].append(self.Features["atomkernel_descriptors"][a_i,:])

class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]

class AtomicDataset(Dataset):
    def __init__(self, X, y, n_atoms_per_structure):
        self.X = X
        self.y = y
        self.index = self.generate_atomstructure_index(n_atoms_per_structure)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx])

    def generate_atomstructure_index(self, n_atoms_per_structure):
        n_structures = len(n_atoms_per_structure)
        total_index = []
        for i, atoms in enumerate(n_atoms_per_structure):
            indiv_index = torch.zeros(atoms) + i
            total_index.append(indiv_index)
        total_index = torch.hstack(total_index)
        return total_index.long()
