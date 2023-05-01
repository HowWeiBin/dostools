import numpy as np
import pickle
import json
import ase
import ase.io as ase_io
import sys
from . import dataset
import torch 



from pathlib import Path
data_path = Path(__file__).parent/"data"

eigenenergies_fileloc = data_path.joinpath("FHI-aims_data/train_energies.json")
structures_fileloc = data_path.joinpath("Structures/training_dataset.xyz")
xdos_fileloc = data_path.joinpath("xdos.npy")
ldos_fileloc = data_path.joinpath("ldos.npy")
kMM_fileloc = data_path.joinpath("kMM.npy")
features_fileloc = data_path.joinpath("Silicon_Features.pickle")
s_coeff_fileloc = data_path.joinpath("structure_coefficients.pt")
s_eigvals_fileloc = data_path.joinpath("structure_eigvals.pt")
reconstructed_ldos_fileloc = data_path.joinpath("reconstructed_ldos.pt")

def load_eigenenergies(unpack: bool, n_structures: int = None):
	"""
	Loads eigenenergies and returns a list of eigenenergies, min_energy and max_energy if unpack, else a dictionary  
	
	Args:
	    unpack (bool): Decide if data should be unpacked or not
	    n_structures (int, optional): Total number of structures
	
	Returns:
	    eigen_energies (dict): Dictionary loaded from json
	    eigenenergies (list): list of eigenergies corresponding to each structure
	    emin (np.float64): Minimum energy in the dataset
	    emax (np.float64): Maximum energy in the dataset
	"""
	with open(eigenenergies_fileloc) as f:
		eigen_energies = json.load(f)
	print (eigen_energies["info"])

	if not unpack:
		return eigen_energies

	else:
		#Unpack eigenenergies
		eigenenergies = []
		for i in range(n_structures):
		    bandenergies = []
		    kpointenergy = eigen_energies["%d"%i]["kpoints"]
		    for k in kpointenergy.keys():
		        bandenergies.append(kpointenergy[k])
		    eigenenergies.append(np.array(bandenergies))
		    
		# determine the minimum and maximum energies to establish DOS range
		emin = np.min(np.array([np.min(eigenenergies[i]) for i in range(len(eigenenergies))]))
		emax = np.max(np.array([np.max(eigenenergies[i]) for i in range(len(eigenenergies))]))

	return eigenenergies, emin, emax

def load_structures(index: str):
	"""
	Loads structures using ase.io and returns list of ASE structures
	
	Args:
	    index (str): Indexes of structures to read
	
	Returns:
	    List: List of ase structures
	"""
	structures = ase_io.read(structures_fileloc, index)

	return structures

def load_xdos():
	"""
	Loads xdos file and returns a np array
	
	Returns:
	    nparray: xdos np array
	"""
	xdos = np.load(xdos_fileloc)
	return xdos

def load_ldos():
	"""
	loads ldos file and returns a np array
	
	Returns:
	    nparray: ldos np array
	"""
	ldos = np.load(ldos_fileloc)
	return ldos

def load_kMM():
	"""
	loads kMM file and returns a np array
	
	Returns:
	    nparray: kMM np array
	"""
	kMM = np.load(kMM_fileloc)
	return kMM

def load_features():
    """
    loads features file and returns TensorFeatures
    
    Returns:
        TYPE: Description
    """
    with open(features_fileloc,'rb') as file:
        features = pickle.load(file)
    return features

def load_structure_gaussians():
    eigvals = torch.load(s_eigvals_fileloc)
    coeffs = torch.load(s_coeff_fileloc)
    reconstructed = torch.load(reconstructed_ldos_fileloc)
    
    return eigvals, coeffs, reconstructed