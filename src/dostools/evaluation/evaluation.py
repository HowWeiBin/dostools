import torch
import numpy as np
from ..loss.loss import t_get_rmse
from ..postprocessing import postprocessing

class Evaluator:
    def __init__(self, targets: dict, x_dos: torch.tensor, mean_dos_per_atom):
        self.targets = targets
        self.pc_vectors = {}
        self.x_dos = x_dos
        self.mean = mean_dos_per_atom

    def GetTargetRMSE(self, total_preds, target_name, train_index, test_index):
        with torch.no_grad():
            total_preds = torch.as_tensor(total_preds)
            if target_name == "pw":
                total_DOS_preds =  total_preds + self.mean
                Ref_DOS = self.targets[target_name]
                x_dos = self.x_dos
            if target_name == "lcdf":
                total_DOS_preds = total_preds
                Ref_DOS = self.targets[target_name]
                x_dos = self.x_dos
            if target_name == "pc":
                total_DOS_preds = total_preds
                Ref_DOS = self.targets[target_name]
                x_dos = None
            if target_name == "pc_S":
                total_DOS_preds = total_preds
                Ref_DOS = self.targets[target_name]
                x_dos = None
            if target_name == "pc_S2":
                total_DOS_preds = total_preds
                Ref_DOS = self.targets[target_name]
                x_dos = None       

            DOS_train_error_p = t_get_rmse(total_DOS_preds[train_index,:], Ref_DOS[train_index,:], xdos = x_dos, perc = True).item()
            DOS_test_error_p = t_get_rmse(total_DOS_preds[test_index,:], Ref_DOS[test_index,:], xdos = x_dos, perc = True).item()
            DOS_total_error_p = t_get_rmse(total_DOS_preds, Ref_DOS, xdos = x_dos, perc = True).item()
            DOS_train_error = t_get_rmse(total_DOS_preds[train_index,:], Ref_DOS[train_index,:], xdos = x_dos, perc = False).item()
            DOS_test_error = t_get_rmse(total_DOS_preds[test_index,:], Ref_DOS[test_index,:], xdos = x_dos, perc = False).item()
            DOS_total_error = t_get_rmse(total_DOS_preds, Ref_DOS, xdos = x_dos, perc = False).item()

            DOSRMSES = np.array([DOS_train_error_p, DOS_test_error_p, DOS_total_error_p, DOS_train_error, DOS_test_error, DOS_total_error])

            return DOSRMSES

    def GetDosRMSE(self, total_preds, target_name, train_index, test_index):
        with torch.no_grad():
            total_preds = torch.as_tensor(total_preds)
            if target_name == "pw":
                total_DOS_preds =  total_preds + self.mean
                Ref_DOS = self.targets[target_name]
                x_dos = self.x_dos
            if target_name == "lcdf":
                total_DOS_preds = postprocessing.t_build_dos_from_CDF(total_preds)+ self.mean
                Ref_DOS = self.targets["pw"]
                x_dos = self.x_dos
            if target_name == "pc":
                total_DOS_preds = postprocessing.t_build_truncated_dos(self.pc_vectors[target_name], total_preds, self.mean, n_pc = 10)
                Ref_DOS = postprocessing.t_build_truncated_dos(self.pc_vectors[target_name], self.targets[target_name], self.mean, n_pc = 10)
                x_dos = self.x_dos
            if target_name == "pc_S":
                total_DOS_preds = postprocessing.t_build_truncated_dos(self.pc_vectors[target_name], total_preds, self.mean[None,:-171], n_pc = 10)
                Ref_DOS = postprocessing.t_build_truncated_dos(self.pc_vectors[target_name], self.targets[target_name], self.mean[None,:-171], n_pc = 10)
                x_dos = self.x_dos[:-171]
            if target_name == "pc_S2":
                total_DOS_preds = postprocessing.t_build_truncated_dos(self.pc_vectors[target_name], total_preds, self.mean['pw'][None,160:-171], n_pc = 10)
                Ref_DOS = postprocessing.t_build_truncated_dos(self.pc_vectors[target_name], self.targets[target_name], self.mean['pw'][None,160:-171], n_pc = 10)
                x_dos = self.x_dos[160:-171]        

            DOS_train_error_p = t_get_rmse(total_DOS_preds[train_index,:], Ref_DOS[train_index,:], xdos = x_dos, perc = True).item()
            DOS_test_error_p = t_get_rmse(total_DOS_preds[test_index,:], Ref_DOS[test_index,:], xdos = x_dos, perc = True).item()
            DOS_total_error_p = t_get_rmse(total_DOS_preds, Ref_DOS, xdos = x_dos, perc = True).item()
            DOS_train_error = t_get_rmse(total_DOS_preds[train_index,:], Ref_DOS[train_index,:], xdos = x_dos, perc = False).item()
            DOS_test_error = t_get_rmse(total_DOS_preds[test_index,:], Ref_DOS[test_index,:], xdos = x_dos, perc = False).item()
            DOS_total_error = t_get_rmse(total_DOS_preds, Ref_DOS, xdos = x_dos, perc = False).item()

            DOSRMSES = np.array([DOS_train_error_p, DOS_test_error_p, DOS_total_error_p, DOS_train_error, DOS_test_error, DOS_total_error])

            return DOSRMSES
