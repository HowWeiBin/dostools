import torch
import numpy as np
from . import training

class GPR:
	def __init__(self, feature, feature_name, target, target_name, x_dos, kMM):
		self.weights = {}
		self.feature = feature
		self.feature_name = feature_name
		self.target = target
		self.target_name = target_name
		self.x_dos = x_dos
		self.kMM = kMM
		self.DOSrmse = {}
		self.targetrmse = {}

	def obtain_weights(self, data_intervals, train_index):
		self.data_intervals = data_intervals
		for train_ratio in data_intervals:
			self.weights[train_ratio] = training.train_analytical_model_GPR(self.feature, self.feature_name, self.target, self.target_name, self.x_dos,
				self.kMM, train_index, train_ratio, 2)

	def get_DosRMSE(self, evaluator, train_index, test_index):
		for train_ratio in self.data_intervals:
			n_train = int(train_ratio * len(train_index))
			train_index = train_index[:n_train]
			pred = np.hstack([self.feature, np.ones(len(self.feature)).reshape(-1,1)]) @ self.weights[train_ratio]
			self.DOSrmse[train_ratio] = evaluator.GetDosRMSE(pred, self.target_name, train_index, test_index)

	def get_targetRMSE(self, evaluator, train_index, test_index):
		for train_ratio in self.data_intervals:
			n_train = int(train_ratio * len(train_index))
			train_index = train_index[:n_train]
			pred = np.hstack([self.feature, np.ones(len(self.feature)).reshape(-1,1)]) @ self.weights[train_ratio]
			self.targetrmse[train_ratio] = evaluator.GetTargetRMSE(pred, self.target_name, train_index, test_index)

class RidgeRegression:
	def __init__(self, feature: torch.tensor, feature_name: str, target: torch.tensor, target_name: str, x_dos: torch.tensor):
		self.models = {}
		self.feature = feature
		self.feature_name = feature_name
		self.target = target
		self.target_name = target_name
		self.x_dos = x_dos
		self.DOSrmse = {}
		self.targetrmse = {}
	def obtain_weights(self, data_intervals, train_index):
		self.data_intervals = data_intervals
		for train_ratio in data_intervals:
			self.models[train_ratio] = training.train_analytical_model_ridge(self.feature, self.feature_name, self.target, self.target_name, self.x_dos, train_index, train_ratio, 2)

	def get_DosRMSE(self, evaluator, train_index, test_index):
		for train_ratio in self.data_intervals:
			feature = torch.hstack([self.feature, torch.ones(len(self.feature)).view(-1,1)])
			pred = feature @ self.models[train_ratio]
			self.DOSrmse[train_ratio] = evaluator.GetDosRMSE(pred, self.target_name, train_index, test_index)

	def get_targetRMSE(self, evaluator, train_index, test_index):
		for train_ratio in self.data_intervals:
			feature = torch.hstack([self.feature, torch.ones(len(self.feature)).view(-1,1)])
			pred = feature @ self.models[train_ratio]
			self.targetrmse[train_ratio] = evaluator.GetTargetRMSE(pred, self.target_name, train_index, test_index)	

class TRegression: 
	def __init__(self, feature: torch.tensor, feature_name: str, target: torch.tensor, target_name: str, datatype: str, x_dos: torch.tensor, opt:str, device: str):
		self.models = {}
		self.feature = feature
		self.feature_name = feature_name
		self.target = target
		self.target_name = target_name
		self.datatype = datatype
		self.x_dos = x_dos
		self.opt = opt
		self.device = device
		self.DOSrmse = {}
		self.targetrmse = {}
		self.val = False

	def obtain_weights(self, data_intervals, train_index, reg, lr, n_epochs, loss):
		if reg is None:
			reg = training.torch_linear_optimize_hypers(self.feature, self.feature_name, self.target, self.target_name, self.datatype, self.opt, lr, n_epochs, self.device, 2, self.x_dos, train_index, loss, self.val)
		self.data_intervals = data_intervals
		for train_ratio in data_intervals:
			self.models[train_ratio] = training.train_torch_linear_model(self.feature, self.feature_name, self.target, self.target_name, self.datatype, loss, self.opt, 
				lr, n_epochs, 2, self.x_dos, train_index, self.device, reg, self.val)

	def get_DosRMSE(self, evaluator, train_index, test_index):
		for train_ratio in self.data_intervals:
			pred = self.models[train_ratio](self.feature)
			self.DOSrmse[train_ratio] = evaluator.GetDosRMSE(pred, self.target_name, train_index, test_index)

	def get_targetRMSE(self, evaluator, train_index, test_index):
		for train_ratio in self.data_intervals:
			pred = self.models[train_ratio](self.feature)
			self.targetrmse[train_ratio] = evaluator.GetTargetRMSE(pred, self.target_name, train_index, test_index)	

