import copy
import math
import torch
import numpy as np
import scipy.special
import scipy.spatial
from numpy.random import rand
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
from torch.autograd import Variable
from csdi_utils import *

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class ClustCQR():
	'''
	The CQR class implements Conformalized Quantile Regression, i.e. it applies CP to a QR
	Inputs: 
		- Xc, Yc: the calibration set
		- trained_qr_model: pre-trained quantile regressor
		- quantiles: the quantiles used to train the quantile regressor
		- test_hist_size, cal_hist_size: number of observations per point in the test and calibration set respectively
		- comb_flag = False: performs normal CQR over a single property 
		- comb_flag = True: combine the prediction intervals of the CQR of two properties
	'''
	def __init__(self, Lc, Xc, cal_loader, num_cal_points, stl_property, cp_classifier, trained_generator, num_classes, opt, quantiles = [0.05, 0.95], plots_path='', load=False, nsamples=1):
		super(ClustCQR, self).__init__()

		self.Lc = Lc
		self.Xc = Xc
		self.cp_classifier = cp_classifier
		self.trained_generator = trained_generator
		self.cal_loader = cal_loader
		self.q =  Xc.shape[0]# number of points in the calibration set * num trajs
		self.quantiles = quantiles
		self.epsilon = 2*quantiles[0]
		self.M = len(quantiles) # number of quantiles
		self.col_list = ['yellow', 'orange', 'red', 'orange', 'yellow']
		self.plots_path = plots_path
		self.num_classes = num_classes
		self.stl_property = stl_property
		self.cal_robs = stl_property(self.Xc)
		self.num_cal_points = num_cal_points
		self.min = cal_loader.dataset.min
		self.max = cal_loader.dataset.max
		self.opt = opt
		self.load = load
		self.nsamples = nsamples

	def get_pred_interval(self, inputs_loader,extra=''):
		'''
		Apply the trained QR to inputs and returns the QR prediction interval
		'''

		#[n_points, n_trajs, T, K]
		#gen_trajs = generate_trajectories(self.trained_generator,inputs_loader,nsamples=10)
		if self.load:
			with open(self.plots_path+f'/trajs_{extra}_nsamples={self.nsamples}.pickle', 'rb') as file:
				D = pickle.load(file)
			gen_trajs,real_t = D['gen_trajs'],D['real_t']
		else:
			gen_trajs,real_t = light_evaluate(self.trained_generator, inputs_loader, nsample=self.nsamples,foldername=self.plots_path,  ds_id = extra)
		
			results = {'gen_trajs': gen_trajs, 'real_t': real_t}
			with open(self.plots_path+f'/trajs_{extra}_nsamples={self.nsamples}.pickle', 'wb') as file:
				pickle.dump(results, file)
		
		
		QQ = torch.empty((self.num_classes, gen_trajs.shape[0], 2))
		for i in range(gen_trajs.shape[0]):
			rescaled_realtrajs_i = self.min+(real_t[i]+1)*(self.max-self.min)/2 
			
			rescaled_trajs_i = self.min+(gen_trajs[i]+1)*(self.max-self.min)/2 
			rescaled_trajs_i[:,:int(-self.opt.testmissingratio)] = rescaled_realtrajs_i[0,:int(-self.opt.testmissingratio)]
			class_cpi_i = self.cp_classifier.get_prediction_region(rescaled_trajs_i.detach().numpy()) #[n_trajs,num_classes]

			#class_cpi_i = self.cp_classifier.get_prediction_region(gen_trajs[i].detach().numpy()) #[n_trajs,num_classes]
			for g in range(self.num_classes):
				kept_trajs_ig = []
				for j in range(gen_trajs.shape[1]):
					if class_cpi_i[j,g] == 1:
						#kept_trajs_ig.append(gen_trajs[i,j])
						kept_trajs_ig.append(rescaled_trajs_i[j])
				if len(kept_trajs_ig) > 0:
					KT = torch.stack(kept_trajs_ig, dim=0)
					
					robs_ig = self.stl_property(KT.detach().numpy())
					QQ[g,i,0] = torch.quantile(robs_ig, q = self.quantiles[0]) # qlo_ig
					QQ[g,i,1] = torch.quantile(robs_ig, q = self.quantiles[-1]) # qhi_ig
				else:
					QQ[g,i,0] = -math.inf # qlo_ig
					QQ[g,i,1] = math.inf # qhi_ig

		return QQ

	def get_calibr_nonconformity_scores(self):
		'''
		Compute the nonconformity scores over the calibration set
		if sorting = True the returned scores are ordered in a descending order
		'''

		print('self.calibr_quantiles: ', self.calibr_quantiles[0])
		print('self.listRc: ', self.listRc[0])
		ncm = []
		for g in range(self.num_classes):
			ncm_g = []
			for i in range(self.num_cal_points):
				ncm_ig = []
				for j in range(len(self.listRc[g][i])):
				
					# we compare g-specific quantiles, obtained by gen. trajectories having g
					# in the prediction region and compare it with real rob values of true cal trajs
					# belonging to class g
					ncm_ig.append(max(self.calibr_quantiles[g][i,0]-self.listRc[g][i][j], self.listRc[g][i][j]-self.calibr_quantiles[g][i,-1])) # pred_interval[i,0] = q_lo(x), pred_interval[i,1] = q_hi(x)
				ncm_g.append(ncm_ig)
			ncm_gg = sum(ncm_g,[])
			#print(f'g={g}, nb cal = {len(ncm_gg)}')
			ncm.append(np.sort(ncm_gg)[::-1])
		
		return ncm


	def get_scores_threshold(self):
		'''
		This method extract the threshold value tau (the quantile at level epsilon) from the 
		calibration nonconformity scores (computed by the 'self.get_calibr_nonconformity_scores' method)
		'''
		self.m = self.q//self.num_cal_points
		#print('Get calibr classif pred sets')
		#cal_pred_regions = self.cp_classifier.get_prediction_region(self.Xc)
		#cal_pred_regions_res = cal_pred_regions.reshape((self.num_cal_points,self.m, self.num_classes))
		
		cal_rob_res = self.cal_robs.reshape((self.num_cal_points,self.m))
		
		Lc_res = self.Lc.reshape((self.num_cal_points,self.m))
		self.listRc = []  # num_classes, num_cal_points, var_num_trajs
		
		for g in range(self.num_classes):
			Rc_g = []
			for i in range(self.num_cal_points):
				Rc_ig = []
				for j in range(self.m):
					#if cal_pred_regions_res[i,j,g] == 1:
					if Lc_res[i,j] == g:
						Rc_ig.append(cal_rob_res[i,j])
				Rc_g.append(Rc_ig)
			self.listRc.append(Rc_g)

		print('Get calibr generative pred interval')
		start_time = time.time()
		calibr_quantiles = self.get_pred_interval(self.cal_loader,extra='cal_')
		self.calibr_quantiles = calibr_quantiles.detach().cpu().numpy()
		print(' 		time for the calibr pred interval = ', time.time()-start_time)
		#self.calibr_quantiles = np.array([calibr_class_quantiles[int(self.Lc[i]),i] for i in range(calibr_class_quantiles.shape[1])])
		
		# nonconformity scores on the calibration set
		self.calibr_scores = self.get_calibr_nonconformity_scores()
		#print(self.calibr_scores)
		self.tau = np.empty(self.num_classes)
		for g in range(self.num_classes):
			print(f'g={g}, nb cal points ={len(self.calibr_scores[g])}')
			if len(self.calibr_scores[g]) > 0:
				Qg = (1-self.epsilon)*(1+1/len(self.calibr_scores[g]))
				self.tau[g] = np.quantile(self.calibr_scores[g], Qg)
				fig = plt.figure()
				plt.scatter(np.arange(len(self.calibr_scores[g])), self.calibr_scores[g], color='g')
				plt.scatter(self.epsilon*(1+1/len(self.calibr_scores[g]))*len(self.calibr_scores[g]), self.tau[g],color='r')
				plt.title('calibr ncm')
				plt.tight_layout()
				plt.savefig(self.plots_path+f'/calibr_nonconf_scores_cluster_g={g}_sol2.png')
				plt.close()
			else:
				self.tau[g] = math.inf
			
		print("self.tau: ", self.tau)



	def get_cpi(self, inputs_loader, pi_flag = False):
		'''
		Returns the conformalized prediction interval (cpi) by enlarging the 
		QR prediction interval by adding an subtracting tau from the lower and upper bound resp.
		'''

		
		res_path = self.plots_path+f'/sol2_PIs.pickle'

		if False:
			start_time = time.time()
			pis = self.get_pred_interval(inputs_loader,extra='test_') #[n_points,n_classes,2]
			print('-------- Time to compute the test PI = ', time.time()-start_time)

			pis_dict = {'pis': pis}
			with open(res_path, 'wb') as file:
				pickle.dump(pis_dict, file)
		else:
			with open(res_path, 'rb') as file:
				D = pickle.load(file)
			pis = D['pis']

		start_time = time.time()
		self.get_scores_threshold()
		print('-------- Time to get the taus = ', time.time()-start_time)


		cpis = []
		for g in range(self.num_classes):
			cpi = np.vstack((pis[g,:,0]-self.tau[g], pis[g,:,-1]+self.tau[g]))
			cpis.append(cpi.T)


		if pi_flag:
			return cpis, pis.detach().numpy()
		else:
			return cpis


	def get_coverage_efficiency(self, Ltest, Rtest, test_pred_intervals):
		'''
		Compute the empirical coverage and the efficiency of a prediction interval (test_pred_interval).
		y_test are the observed target values
		'''
		n_test_points = len(Ltest)//self.m
		test_rob_res = Rtest.reshape((n_test_points,self.m))
		
		Ltest_res = Ltest.reshape((n_test_points,self.m))

		self.listRtest = []  # num_classes, num_test_points, var_num_trajs
		
		for g in range(self.num_classes):
			Rt_g = []
			for i in range(n_test_points):
				Rt_ig = []
				for j in range(self.m):
					
					if Ltest_res[i,j] == g:
						Rt_ig.append(test_rob_res[i,j])
				Rt_g.append(Rt_ig)
			self.listRtest.append(Rt_g)

		
		cov = np.zeros(self.num_classes)
		eff = np.zeros(self.num_classes)

		for g in range(self.num_classes):
			#eff[g] = np.mean(np.abs(test_pred_intervals[g][:,-1]-test_pred_intervals[g][:,0]))
			den_gc = 0
			den_ge = 0
			for i in range(n_test_points):

				if len(self.listRtest[g][i]) > 0:

					eff[g] += np.abs(test_pred_intervals[g][i,-1]-test_pred_intervals[g][i,0])
					den_ge += 1

				den_gc += len(self.listRtest[g][i])
				for j in range(len(self.listRtest[g][i])):
					if self.listRtest[g][i][j] >= test_pred_intervals[g][i,0] and self.listRtest[g][i][j] <= test_pred_intervals[g][i,-1]:
						cov[g] += 1

			cov[g] = cov[g]/den_gc
			eff[g] = eff[g]/den_ge
		return cov, eff


	def get_global_coverage_efficiency(self, Ltest, Rtest, test_pred_intervals):
		'''
		Compute the empirical coverage and the efficiency of a prediction interval (test_pred_interval).
		y_test are the observed target values
		'''

		
		self.m = self.q//self.num_cal_points
		n_test_points = len(Ltest)//self.m
		test_rob_res = Rtest.reshape((n_test_points,self.m))
		
		Ltest_res = Ltest.reshape((n_test_points,self.m))

		self.listRtest = []  # num_classes, num_test_points, var_num_trajs
		
		for g in range(self.num_classes):
			Rt_g = []
			for i in range(n_test_points):
				Rt_ig = []
				for j in range(self.m):
					
					if Ltest_res[i,j] == g:
						Rt_ig.append(test_rob_res[i,j])
				Rt_g.append(Rt_ig)
			self.listRtest.append(Rt_g)

		
		cov = np.zeros(self.num_classes)
		
		for g in range(self.num_classes):
			#eff[g] = np.mean(np.abs(test_pred_intervals[g][:,-1]-test_pred_intervals[g][:,0]))
			den_gc = 0

			for i in range(n_test_points):


				den_gc += len(self.listRtest[g][i])
				for j in range(len(self.listRtest[g][i])):
					if self.listRtest[g][i][j] >= test_pred_intervals[g][i,0] and self.listRtest[g][i][j] <= test_pred_intervals[g][i,-1]:
						cov[g] += 1

			cov[g] = cov[g]/den_gc

		eff = 0
		for i in range(n_test_points):
			intervals = []
			for g in range(self.num_classes):
				intervals.append([test_pred_intervals[g][i,0],test_pred_intervals[g][i,-1]])

			eff += self.sum_intervals(intervals)/n_test_points


		return cov.mean(), eff

	def merge_intervals(self, intervals):
	    # Ordina gli intervalli in base all'inizio
	    intervals.sort(key=lambda x: x[0])
	    merged = []

	    for interval in intervals:
	        # Se la lista merged è vuota o l'intervallo corrente non si sovrappone all'ultimo intervallo unito
	        if not merged or merged[-1][1] < interval[0]:
	            merged.append(interval)
	        else:
	            # Altrimenti, c'è sovrapposizione, quindi unisci gli intervalli
	            merged[-1][1] = max(merged[-1][1], interval[1])

	    return merged

	def sum_intervals(self, intervals):
	    # Unisce gli intervalli sovrapposti
	    merged_intervals = self.merge_intervals(intervals)
	    # Calcola la lunghezza totale degli intervalli uniti
	    total_length = sum(end - start for start, end in merged_intervals)
	    return total_length

	def get_eqr(self, Rtest):

		n = Rtest.shape[0]
		eqr = 0
		for i in range(n):

			eqr += (np.quantile(Rtest[i],self.quantiles[-1])-np.quantile(Rtest[i],self.quantiles[0]))/n

		return eqr
	def compute_accuracy_and_uncertainty(self, test_pred_interval, S_test):
		'''
		Computes the number of correct, uncertain and wrong prediction intervals and the number of false positives.
		S_test is the sign of the observed quantile interval (-1: negative, 0: uncertain, +1: positive)
		'''
		n_points = len(S_test)

		correct = 0
		wrong = 0
		uncertain = 0
		fp = 0

		for i in range(n_points):
			
			if S_test[i,2]: # sign +1
				if test_pred_interval[i,0] >= 0 and test_pred_interval[i,-1] > 0:
					correct += 1
				elif test_pred_interval[i,0] <= 0 and test_pred_interval[i,-1] >= 0:
					uncertain += 1
				else:
					wrong +=1
			elif S_test[i,1]: # sign 0
				if test_pred_interval[i,0] <= 0 and test_pred_interval[i,-1] >= 0:
					correct += 1
				else:
					wrong +=1
					if test_pred_interval[i,0] > 0:
						fp+= 1
			else: # sign -1
				if test_pred_interval[i,-1] <= 0 and test_pred_interval[i,0] < 0:
					correct += 1
				elif test_pred_interval[i,-1] >= 0 and test_pred_interval[i,0] <= 0:
					uncertain += 1
				else:
					wrong +=1
					fp += 1

		return correct/n_points, uncertain/n_points, wrong/n_points, fp/n_points


	def plot_errorbars(self, y, qr_interval, cqr_interval, title_string, plot_path, extra_info = ''):
		'''
		Create barplots
		'''
		n_points_to_plot = 30

		
		self.test_hist_size = y.shape[1]

		n_test_points = y.shape[0]

		ind = np.zeros(n_test_points, dtype=bool)
		ind[:n_points_to_plot//3] = 1
		ind[50:50+n_points_to_plot//3] = 1
		ind[100:100+n_points_to_plot//3] = 1
		y_resh = y[ind]
		yq = []
		yq_out = []
		xline_rep = []
		xline_rep_out = []
		for i in range(n_points_to_plot):
			
			lower_yi = np.quantile(y_resh[i], self.epsilon/2)
			upper_yi = np.quantile(y_resh[i], 1-self.epsilon/2)
			for j in range(self.test_hist_size):
				if y_resh[i,j] <= upper_yi and y_resh[i,j] >= lower_yi:
					yq.append(y_resh[i,j])
					xline_rep.append(i)
				else:
					yq_out.append(y_resh[i,j])
					xline_rep_out.append(i)					

		n_quant = qr_interval[0].shape[1]

		xline = np.arange(n_points_to_plot)
		xline0 = np.arange(n_points_to_plot)+0.2
		xline1 = np.arange(n_points_to_plot)+0.3
		xline2 = np.arange(n_points_to_plot)+0.4

		fig = plt.figure(figsize=(20,4))
		plt.scatter(xline_rep_out, yq_out, c='peachpuff', s=6, alpha = 0.25)
		plt.scatter(xline_rep, yq, c='orange', s=6, alpha = 0.25,label='test')
		
		plt.plot(xline, np.zeros(n_points_to_plot), '-.', color='k')
		
		if cqr_interval[0][0,-1] < math.inf and cqr_interval[0][0,-1] == cqr_interval[0][0,-1]:
			cqr_med = (cqr_interval[0][ind,-1]+cqr_interval[0][ind,0])/2
			cqr_dminus = cqr_med-cqr_interval[0][ind,0]
			cqr_dplus = cqr_interval[0][ind,-1]-cqr_med
			plt.errorbar(x=xline0, y=cqr_med, yerr=[cqr_dminus,cqr_dplus],  color = 'blue', fmt='none', capsize = 4,label='0')
		
		if cqr_interval[1][0,-1] < math.inf and cqr_interval[1][0,-1] == cqr_interval[1][0,-1]:
			cqr_med = (cqr_interval[1][ind,-1]+cqr_interval[1][ind,0])/2
			cqr_dminus = cqr_med-cqr_interval[1][ind,0]
			cqr_dplus = cqr_interval[1][ind,-1]-cqr_med
			plt.errorbar(x=xline1, y=cqr_med, yerr=[cqr_dminus,cqr_dplus],  color = 'violet', fmt='none', capsize = 4,label='1')

		if cqr_interval[2][0,-1] < math.inf and cqr_interval[2][0,-1] == cqr_interval[2][0,-1]:
			cqr_med = (cqr_interval[2][ind,-1]+cqr_interval[2][ind,0])/2
			#cqr_med = cqr_interval[2][:n_points_to_plot,1]
			cqr_dminus = cqr_med-cqr_interval[2][ind,0]
			cqr_dplus = cqr_interval[2][ind,-1]-cqr_med
			plt.errorbar(x=xline2, y=cqr_med, yerr=[cqr_dminus,cqr_dplus],  color = 'darkviolet', fmt='none', capsize = 4,label='2')

		plt.ylabel('robustness')
		plt.title(title_string)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		fig.savefig(plot_path+"/cluster_"+extra_info+"_errorbar_merged.png")
		plt.close()

	def plot_multimodal_errorbars(self, y, qr_interval, cqr_interval, title_string, plot_path, extra_info = '', model_name = ''):
		'''
		Create barplots
		'''
		n_points_to_plot = 10
		
		self.test_hist_size = y.shape[1]

		n_test_points = y.shape[0]
		

		y_resh = y[:n_points_to_plot]
		yq = []
		yq_out = []
		x_rep = []
		xline_rep = []
		xline_rep_out = []
		li = []
		for i in range(n_points_to_plot):

			lower_yi = np.quantile(y_resh[i], self.quantiles[0])
			upper_yi = np.quantile(y_resh[i], self.quantiles[-1])
			for j in range(self.test_hist_size):
				x_rep.append(i)
				if y_resh[i,j] <= upper_yi and y_resh[i,j] >= lower_yi:
					yq.append(y_resh[i,j])
					xline_rep.append(i)
				else:
					yq_out.append(y_resh[i,j])
					xline_rep_out.append(i)					

		n_quant = qr_interval[0].shape[1]

		xlines = [np.arange(n_points_to_plot) + 0.15*i for i in range(self.num_classes+1)]

		fig = plt.figure(figsize=(20,4))
		#plt.scatter(xline_rep_out, yq_out, c='peachpuff', s=6, alpha = 0.25)
		#plt.scatter(xline_rep, yq, c='orange', s=6, alpha = 0.25,label='test')
		plt.scatter(x_rep, y_resh.flatten(), c='orange', s=6, alpha = 0.25,label='test')
		
		plt.plot(xlines[0], np.zeros(n_points_to_plot), '-.', color='k')
		
		colors =['cyan','blue','darkviolet','violet']
		for k in range(self.num_classes):
			if cqr_interval[k][0,-1] < math.inf and cqr_interval[k][0,-1] == cqr_interval[k][0,-1]:
				cqr_med = (cqr_interval[k][:n_points_to_plot,-1]+cqr_interval[k][:n_points_to_plot,0])/2
				cqr_dminus = cqr_med-cqr_interval[k][:n_points_to_plot,0]
				cqr_dplus = cqr_interval[k][:n_points_to_plot,-1]-cqr_med
				plt.errorbar(x=xlines[k+1], y=cqr_med, yerr=[cqr_dminus,cqr_dplus],  color = colors[k], fmt='none', capsize = 4,label=str(k+1))

		plt.ylabel('robustness')
		plt.title(title_string)
		plt.legend(fontsize=24)
		plt.grid(True)
		plt.tight_layout()
		fig.savefig(plot_path+f"/{model_name}_cluster_"+extra_info+"_multimodal_errorbar_sol2_10.png")
		plt.close()
