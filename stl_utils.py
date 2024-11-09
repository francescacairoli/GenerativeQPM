import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import stl
import numpy as np


print("CUDA availabe = " + str(torch.cuda.is_available()))
device = torch.device("cuda") 

def eval_signal_property(signal):

	signal = signal.transpose(0,2,1)
	n_timesteps = signal.shape[2]
	

	atom = stl.Atom(var_index=1, threshold=17.5, lte=False) # lte = True is <=
	
	glob = stl.Globally(atom, unbound=True, time_bound=n_timesteps-1)

	formula = stl.Eventually(glob, unbound=False, time_bound=n_timesteps-1)

	satisf = formula.quantitative(torch.tensor(signal))
	return satisf

def eval_crossroad_multiagent_property(signal, ped_gen, input_loader, prop_idx=3):

	from csdi_utils import gen_batch_trajs
	#gen_trajs,real_t = light_evaluate(ped_gen, input_loader, nsample=1,foldername='',  ds_id = 'test', save = False)
	#rescaled_realtrajs = input_loader.dataset.min+(real_t[0].cpu().numpy()+1)*(input_loader.dataset.max-input_loader.dataset.min)/2 
	#pedestrian = input_loader.dataset.min+(gen_trajs[0].cpu().numpy()+1)*(input_loader.dataset.max-input_loader.dataset.min)/2 
	#pedestrian[:,:1] = rescaled_realtrajs_i[0,:1]
	signal = signal.transpose(0,2,1)
	n_timesteps = signal.shape[2]
	
	
	pedestrian = gen_batch_trajs(ped_gen, input_loader)
	
	pedestrian = pedestrian.permute(0,2,1)[:,:,:n_timesteps]

	print('signal = ', signal.shape)
	print('pedestrian = ', pedestrian.shape)
	x_barrier = 37

	safe_dist = 5
	ped_dist = 1

	signal_t = torch.tensor(signal)

	moving_car = np.vstack((np.linspace(35,50,n_timesteps)[::-1],33*np.ones(n_timesteps)))
	dist_car = np.expand_dims([np.linalg.norm(signal[i]-moving_car,2,axis=0) for i in range(signal.shape[0])],axis=1)
	dist_car_t = torch.tensor(dist_car)

	
	dist_ped_t = torch.linalg.norm(signal_t-pedestrian[:signal_t.shape[0]],2,axis=1).unsqueeze(1)
	
	#dist_ped_t = torch.tensor(dist_ped)

	# avoid right turns
	atom_rt = stl.Atom(var_index=0, threshold=x_barrier, lte=True) # lte = True is <=
	
	atom_car = stl.Atom(var_index=1, threshold=safe_dist, lte=False) # lte = True is <=
	atom_ped = stl.Atom(var_index=2, threshold=ped_dist, lte=False) # lte = True is <=
	
	obj_form = stl.And(atom_car, atom_ped)
	multi_form = stl.And(atom_rt, obj_form)

	glob3 = stl.Globally(multi_form, unbound=False, time_bound=n_timesteps-1)

	concat_signal_t = torch.cat((signal_t, dist_car_t, dist_ped_t), dim=1)

	satisf = glob3.quantitative(concat_signal_t)
		

	return satisf

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


def _signal_norm(x, center):

	return torch.norm(x-center.unsqueeze(2),np.inf, dim=1).unsqueeze(1).to(device)


def eval_navigator_property(signal, prop_idx=1):#, mins, maxs

	signal = signal.transpose(0,2,1)
	n_timesteps = signal.shape[2]
	
	C1, R1 = [[7.5,22.5]], 2.5
	C2, R2 = [[13., 13.]], 3.
	C3, R3 = [[22.5,7.5]], 2.5
	C4, R4 = [[19.,21.]], 2.

	safety_1 = stl.Atom(var_index=2, threshold=R1, lte = False)
	safety_2 = stl.Atom(var_index=3, threshold=R2, lte = False)
	safety_3 = stl.Atom(var_index=4, threshold=R3, lte = False)
	safety_4 = stl.Atom(var_index=5, threshold=R4, lte = False)
	
	and_12 = stl.And(safety_1, safety_2)
	and_34 = stl.And(safety_3, safety_4)
	safe_and = stl.And(and_12, and_34)

	and_24 = stl.And(safety_2, safety_4)

		
	atom_xu = stl.Atom(var_index=0, threshold=30, lte=True)
	atom_xl = stl.Atom(var_index=0, threshold=0, lte=False)
	atom_yu = stl.Atom(var_index=1, threshold=30, lte=True)
	atom_yl = stl.Atom(var_index=1, threshold=0, lte=False)

	and_x = stl.And(atom_xl, atom_xu)
	and_y = stl.And(atom_yl, atom_yu)
	and_xy = stl.And(and_x, and_y)

	
	
	if prop_idx == 1:
		and_all = stl.And(and_xy, safe_and)
	elif prop_idx == 2:
		and_all = stl.And(and_xy, and_24)
	else:
		and_all = stl.And(and_xy, safety_1)

	formula = stl.Globally(and_all, unbound=True)
	rescaled_signal = torch.tensor(signal,device=device)
	
	sign_obs1 = _signal_norm(rescaled_signal,torch.tensor(C1,device=device))
	sign_obs2 = _signal_norm(rescaled_signal,torch.tensor(C2,device=device))
	sign_obs3 = _signal_norm(rescaled_signal,torch.tensor(C3,device=device))
	sign_obs4 = _signal_norm(rescaled_signal,torch.tensor(C4,device=device))

	concat_signal = torch.cat((rescaled_signal, sign_obs1, sign_obs2, sign_obs3, sign_obs4), dim=1)

	return formula.quantitative(concat_signal)



