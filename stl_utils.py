import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import stl
import numpy as np
import math


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

def eval_crossroad_multiagent_property(signal, ped_gen, input_loader, prop_idx, nsample, extra):

	from csdi_utils import gen_batch_trajs
	
	signal = signal.transpose(0,2,1)
	n_timesteps = signal.shape[2]
	
	
	pedestrian = gen_batch_trajs(ped_gen, input_loader, nsample, extra).to(device)
	
	pedestrian = pedestrian.permute(0,2,1)[:,:,:n_timesteps]

	'''
	ped_np = pedestrian.detach().cpu().numpy()
	for j in range(signal.shape[0]):
		plt.plot(ped_np[j,0], ped_np[j,1], 'r')
		plt.plot(signal[j,0], signal[j,1], 'b')
		plt.plot(np.linspace(35,50,n_timesteps)[::-1],33*np.ones(n_timesteps), 'g')
	plt.title(extra)
	idx = np.random.randint(0,1000,size=1)
	if idx < 100:
		plt.savefig('save/pedestrian/vis/crossroad_multiagent_'+extra+f'_{idx}.png')
	plt.close()
	'''
	print('----' , extra)
	print('signal = ', signal.shape)
	print('pedestrian = ', pedestrian.shape)
	x_barrier = 37

	safe_dist = 5
	ped_dist = 5

	signal_t = torch.tensor(signal).to(device)

	moving_car = np.vstack((np.linspace(35,50,n_timesteps)[::-1],33*np.ones(n_timesteps)))
	dist_car = np.expand_dims([np.linalg.norm(signal[i]-moving_car,2,axis=0) for i in range(signal.shape[0])],axis=1)
	dist_car_t = torch.tensor(dist_car).to(device)

	
	dist_ped_t = torch.linalg.norm(signal_t-pedestrian[:signal_t.shape[0]],2,axis=1).unsqueeze(1)
	
	#dist_ped_t = torch.tensor(dist_ped)

	# avoid right turns
	atom_rt = stl.Atom(var_index=0, threshold=x_barrier, lte=True) # lte = True is <=
	
	atom_car = stl.Atom(var_index=2, threshold=safe_dist, lte=False) # lte = True is <=
	atom_ped = stl.Atom(var_index=3, threshold=ped_dist, lte=False) # lte = True is <=
	
	obj_form = stl.And(atom_car, atom_ped)
	multi_form = stl.And(atom_rt, obj_form)

	glob3 = stl.Globally(multi_form, unbound=False, time_bound=n_timesteps-1)
	#glob3 = stl.Globally(atom_ped, unbound=False, time_bound=n_timesteps-1)

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



