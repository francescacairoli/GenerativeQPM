import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams.update({'font.size': 40})
import itertools

'''
train_path = 'data/crossroad_data.pickle'
with open(train_path, "rb") as f:
	train_data = pickle.load(f)

x_train = train_data['trajs']


# property 1
cal_path= 'data/crossroad_calibr_set_phi1.pickle'
with open(cal_path, "rb") as f:
	x_cal, t_cal= pickle.load(f)
'''


def load_train_data(modelname):
	path = f'data/{modelname}/{modelname}_data_train_w_labels.pickle'
	with open(path, "rb") as f:
		data = pickle.load(f)

	x = data['trajs'].astype(np.float32)
	y = data['labels'].astype(np.int64)

	return x, y

def load_calibr_data(modelname):
	path = f'data/{modelname}/{modelname}_data_calibr_w_labels.pickle'
	with open(path, "rb") as f:
		data = pickle.load(f)

	x = data['trajs'].astype(np.float32)
	y = data['labels'].astype(np.int64)

	return x, y

def load_test_data(modelname):
	path = f'data/{modelname}/{modelname}_data_test_w_labels.pickle'
	with open(path, "rb") as f:
		data = pickle.load(f)

	x = data['trajs'].astype(np.float32)
	y = data['labels'].astype(np.int64)

	return x, y

'''
def load_test_fixed_data(modelname):
	path = f'data/{modelname}_data_test_fixed_froms.pickle'
	with open(path, "rb") as f:
		data = pickle.load(f)

	x = data['trajs'].astype(np.float32)
	y = data['labels'].astype(np.int64)

	return x, y
'''

def navigator_partition(full_trajs):

	n = full_trajs.shape[0]
	labels = np.empty(n)

	for i in range(n):

		if np.any((full_trajs[i,:,0] <= 7.5)*(full_trajs[i,:,1] > 22.5)):
			labels[i] = 0
		elif np.any((full_trajs[i,:,0] >22.5)*(full_trajs[i,:,1] <= 7.5)):
			labels[i] = 3
		elif np.any((full_trajs[i,:,0] >7.5)*(full_trajs[i,:,0] <= 15)*(full_trajs[i,:,1] > 15)*(full_trajs[i,:,1] <= 22.5)):
			labels[i] = 1
		elif np.any((full_trajs[i,:,0] >15)*(full_trajs[i,:,0] <= 22.5)*(full_trajs[i,:,1] > 7.5)*(full_trajs[i,:,1] <= 15)):
			labels[i] = 2
		else:

			labels[i] = 4
		
	return labels

def signal_partition(full_trajs):

	n = full_trajs.shape[0]
	labels = np.empty(n)
	for i in range(n):
		if full_trajs[i,-1,1] > 0 and full_trajs[i,-1,1] < 5:
			labels[i] = 0
		elif full_trajs[i,-1,1] > 10 and full_trajs[i,-1,1] < 15:
			labels[i] = 1
		elif full_trajs[i,-1,1] > 20 and full_trajs[i,-1,1] < 25:
			labels[i] = 2
		else:
			labels[i] = 3


	return labels


def crossroad_partition(trajs):

	n = trajs.shape[0]
	classes = np.empty(n)
	for i in range(n):

		if trajs[i,-1,0] < 25:
			classes[i] = 0
		elif trajs[i,-1,0] <= 38 and trajs[i,-1,0] >= 25:
			classes[i] = 1
		elif trajs[i,-1,0] > 38:
			classes[i] = 2
		else:
			classes[i] = 4


	return classes


def plot_robustness_histogram(Xs, Rs, foldername, model_name, prop_idx):

	print(Rs.shape)
	fig,ax = plt.subplots(figsize=(6,6))
	plt.hist(Rs[0], bins=50, color='teal', edgecolor='black') 
	plt.xlabel('STL Robustness')
	plt.title(model_name)
	plt.tight_layout()
	plt.savefig(foldername+'/'+model_name+f'_robustness_hist_prop={prop_idx}.png')
	plt.close()


def plot_trajs_with_robustness(Xs, Rs, foldername, model_name, prop_idx,n_trajs_to_plot=100, gridmap = None):

	print('Robustness: ', Rs)

	Xs = Xs-.5
	permutations = np.random.randint(Xs.shape[0], size=(n_trajs_to_plot))
	#list(itertools.permutations(np.arange(Xs.shape[0])))
	Rs = Rs.detach().cpu().numpy()
	alpha = 2  # Puoi aumentare il valore di alpha per enfatizzare maggiormente gli estremi
	scalars_normalized = (Rs - min(Rs)) / (max(Rs) - min(Rs))
	scaled_scalars = np.sqrt(scalars_normalized)

	
	colors = cm.get_cmap('RdYlBu')(scaled_scalars)
	

	fig,ax = plt.subplots(figsize=(10,10))
	if gridmap is not None:
		ax.imshow(gridmap, cmap='gray', origin='lower')
		ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

	for i in range(n_trajs_to_plot):#(Xs.shape[0]):
		ii = permutations[i]
		ax.plot(Xs[ii,:,0], Xs[ii,:,1], color = colors[ii],linewidth=2)
	
	sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=plt.Normalize(vmin=min(Rs), vmax=max(Rs)))
	sm.set_array([])
	#sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=min(scalars), vmax=max(scalars)))
	#sm.set_array([])

	cbar = plt.colorbar(sm, ax=ax, label='robustness')
	
	tick_positions = np.linspace(min(Rs), max(Rs), num=5)

	cbar.set_ticks(tick_positions)  # Imposta i tick sui valori originali
	cbar.set_ticklabels([f'{val:.1f}' for val in tick_positions])  # Mostra i valori originali sui tick
	plt.title(model_name, fontsize=45)
	plt.tight_layout()

	fig.savefig(foldername+'/'+model_name+f'_trajs_with_robustness_prop={prop_idx}_grid.png')
	plt.close()


def plot_partition(Xs, partition_fnc, foldername, model_name, prop_idx, n_trajs_to_plot = 100,gridmap=None):

	colors = ['cyan','blue','darkviolet','violet']
	Xtrain, Xcal, Xtest = Xs
	Xtest = Xtest-.5
	'''
	train_classes = partition_fnc(Xtrain)

	
	fig = plt.figure()
	for i in range(Xtrain.shape[0]):
		plt.plot(Xtrain[i,:,0], Xtrain[i,:,1], c=col[int(train_classes[i])])
	fig.savefig(foldername+'/'+model_name+'train_partition.png')
	plt.title('train set')
	plt.close()

	cal_classes = partition_fnc(Xcal)

	
	fig = plt.figure()
	for i in range(Xcal.shape[0]):
		plt.plot(Xcal[i,:,0], Xcal[i,:,1], c=col[int(cal_classes[i])])
	fig.savefig(foldername+'/'+extra+'calibration_partition.png')
	plt.title('calibration set')
	plt.close()
	'''
	test_classes = partition_fnc(Xtest)

	permutations = np.random.randint(Xtest.shape[0], size=(n_trajs_to_plot))
	#list(itertools.permutations(np.arange(Xtest.shape[0])))

	fig, ax = plt.subplots(figsize=(6,6))
	if gridmap is not None:
		ax.imshow(gridmap, cmap='gray', origin='lower')
		ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

	leg_flag = np.ones(len(colors))
	for i in range(n_trajs_to_plot):
		if leg_flag[int(test_classes[permutations[i]])] == 1:
			ax.plot(Xtest[permutations[i],:,0], Xtest[permutations[i],:,1], c=colors[int(test_classes[permutations[i]])], label=str(int(test_classes[permutations[i]]+1)))
			leg_flag[int(test_classes[permutations[i]])] = 0
		else:
			ax.plot(Xtest[permutations[i],:,0], Xtest[permutations[i],:,1], c=colors[int(test_classes[permutations[i]])])
	plt.title(model_name)
	plt.legend()
	plt.tight_layout()
	fig.savefig(foldername+'/'+model_name+f'_test_partition_prop={prop_idx}_grid.png')
	plt.close()
