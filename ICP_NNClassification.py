import numpy as np
from numpy.random import rand
import scipy.special
import copy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from kmeans import *
import torch

class ICP_Classification():
	'''
	Inductive Conformal Prediction for a multilabel classification problem 

	Xc: input points of the calibration set
	Yc: labels corresponding to points in the calibration set
	mondrian_flag: if True computes class conditional p-values
	trained_model: function that takes x as input and returns the prob. of associating it to the positive class

	Remark: the default labels are 0 (negative class) and 1 (positive class)
	Careful: if different labels are considered, used the method set_labels
			(the non conformity scores are not well-defined otherwise)
	'''


	def __init__(self, Xc, Lc, trained_model, num_classes, alpha, plots_path, mondrian_flag=False):
		
		self.Xc = Xc
		self.Lc = Lc
		self.num_classes = num_classes
		self.mondrian_flag = mondrian_flag 
		self.trained_model = trained_model
		self.q = Xc.shape[0]
		self.alpha = alpha
		self.plots_path = plots_path

	def set_calibration_scores(self):

		cal_lkh = self.trained_model(torch.tensor(self.Xc)).detach().numpy()

		# distance of every calibr trajectory to its assigned centroid
		cal_dist = (1 - cal_lkh[:,self.Lc]).flatten() #[n_cal_points*num_cal_trajs,]
		self.calibr_scores = np.sort(cal_dist)[::-1]
		
		fig = plt.figure()
		plt.scatter(np.arange(len(self.calibr_scores)), self.calibr_scores, color='g')
		plt.title('class: calibr ncm')
		plt.tight_layout()
		plt.savefig(self.plots_path+f'/calibr_nonconf_scores_classification.png')
		plt.close()

	def get_nonconformity_scores(self, x, class_index):

		lkhs = self.trained_model(torch.tensor(x)).detach().numpy()

		dist = (1-lkhs[:,class_index]).flatten()

		return dist

	def get_pvalues(self, x):
		'''
		calibr_scores: non conformity measures computed on the calibration set and sorted in descending order
		x: new input points (shape: (n_points,x_dim)
		
		return: p-values for each class
		
		'''
		n_points = x.shape[0]
		
		p_values = np.zeros((n_points, self.num_classes))
		 
		
		A =  np.zeros((n_points, self.num_classes))			
		for k in range(self.num_classes):

			A[:,k] = self.get_nonconformity_scores(class_index=k, x=x)
			
		for i in range(n_points):
			Ca = np.zeros(self.num_classes)
			Cb = np.zeros(self.num_classes)
			for k in range(self.num_classes):
				for j in range(self.q):
					if self.calibr_scores[j] > A[i,k]:
						Ca[k] += 1
					elif self.calibr_scores[j] == A[i,k]:
						Cb[k] += 1
					else:
						break
				
				p_values[i,k] = ( Ca[k] + rand() * (Cb[k] + 1) ) / (self.q + 1)
		
		return p_values


	def compute_prediction_region(self, pvalues):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		#		epsilon = confidence_level
		# OUTPUT: one-hot encoding of the prediction region [shape: (n_points,2)]
		# 		first column: negative class
		# 		second column: positive class
		n_points = pvalues.shape[0]

		pred_region = np.zeros((n_points,self.num_classes)) 
		for i in range(n_points):
			for k in range(self.num_classes):
				if pvalues[i,k] > self.alpha:
					pred_region[i,k] = 1

		return pred_region

	def get_prediction_region(self, x):
		
		pvalues = self.get_pvalues(x)

		return self.compute_prediction_region(pvalues)



	def get_coverage(self, pred_region, labels):

		n_points = pred_region.shape[0]
		
		
		Cov = 0
		for i in range(n_points):
			if pred_region[i,int(labels[i])] == 1:
				Cov += 1

		return Cov/n_points


	def get_efficiency(self, pred_region):

		
		n_points = pred_region.shape[0]

		n_singletons = 0
		for i in range(n_points):
			if np.sum(pred_region[i]) == 1:
				n_singletons += 1
		return n_singletons/n_points


