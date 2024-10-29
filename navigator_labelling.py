import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

flag ='calibr'
model = 'navigator'
fn = f'data/{model}/{model}_data_'+flag+'.pickle'

with open(fn, "rb") as f:
	data = pickle.load(f)

h = 16 # fix number of steps 

full_trajs = data['trajs']

print(data['trajs'].shape)
n = full_trajs.shape[0]
d = full_trajs[0].shape[1]



labels = np.empty(n)
c = 0
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
		c += 1
		labels[i] = 4
print(c)
col = ['r', 'b', 'g', 'm', 'y']
fig = plt.figure()
for i in range(n):
	plt.plot(full_trajs[i][:,0],full_trajs[i][:,1], c=col[int(labels[i])])

plt.savefig(f'data/{model}/{flag}_labels_navigator.png')
plt.close()
print(labels)
path = f'data/{model}/{model}_data_'+flag+'_w_labels.pickle'
data_dict = {'trajs': full_trajs, 'labels': labels}
with open(path, "wb") as f:
	pickle.dump(data_dict, f)
