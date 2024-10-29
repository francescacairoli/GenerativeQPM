import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

flag ='calibr'
model = 'signal'

fn = f'data/{model}/{model}_data_'+flag+'.pickle'

with open(fn, "rb") as f:
	data = pickle.load(f)

m = 22 # fix number of steps 

full_trajs = data['trajs']

print(data['trajs'].shape)
n = full_trajs.shape[0]
d = full_trajs[0].shape[1]

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

col = ['r', 'b', 'g', 'm', 'y']
fig = plt.figure()
for i in range(n):
	plt.plot(full_trajs[i][:,0],full_trajs[i][:,1], c=col[int(labels[i])])

plt.savefig(f'data/{model}/plots/{model}_{flag}_labels.png')
plt.close()

path = f'data/{model}/{model}_data_'+flag+'_w_labels.pickle'
data_dict = {'trajs': full_trajs, 'labels': labels}
with open(path, "wb") as f:
	pickle.dump(data_dict, f)
