import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import stl

import numpy as np
print("CUDA availabe = " + str(torch.cuda.is_available()))
device = torch.device("cuda") 

def eval_crossroad_property(signal, prop_idx = 1):

	signal = signal.transpose(0,2,1)
	n_timesteps = signal.shape[2]
	
	if prop_idx == 1:
		x_barrier = 37

		signal_t = torch.tensor(signal)
	
		atom1 = stl.Atom(var_index=0, threshold=x_barrier, lte=True) # lte = True is <=
		glob1 = stl.Globally(atom1, unbound=False, time_bound=n_timesteps-1)
	
		satisf = glob1.quantitative(signal_t)
	else:
		safe_dist = 5
		moving_car = np.vstack((np.linspace(35,50,n_timesteps)[::-1],33*np.ones(n_timesteps)))

		dist_signal = np.expand_dims([np.linalg.norm(signal[i]-moving_car,2,axis=0) for i in range(signal.shape[0])],axis=1)

		dist_signal_t = torch.tensor(dist_signal)

		atom2 = stl.Atom(var_index=0, threshold=safe_dist, lte=False) # lte = True is <=
		glob2 = stl.Globally(atom2, unbound=False, time_bound=n_timesteps-1)
	
		satisf = glob2.quantitative(dist_signal_t)

	return satisf





