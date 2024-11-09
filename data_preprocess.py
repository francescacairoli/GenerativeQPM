import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

flag ='test'
modelname = 'pedestrian'
#fn = f'data/{modelname}/{modelname}_'+flag+'_trajs_big.mat'
fn = f'data/{modelname}/{modelname}_'+flag+'_trajs.mat'

data = sio.loadmat(fn)


full_trajs = data['list_trajs'][0]

n = full_trajs.shape[0]
d = full_trajs[0].shape[1]

lengths = np.empty(n)
for j in range(n):
	lengths[j] = full_trajs[j].shape[0]
	
max_len = 23#int(np.max(lengths))

print('max len = ', max_len)

print('n = ', n)
print('d = ', d)

X = np.empty((n,max_len,d))

if d==2:
	fig = plt.figure()
for i in range(n):

	D = full_trajs[i]
	h = D.shape[0]

	if h > max_len:
		h = max_len
	X[i,:h] = D[:h].copy()

		
	for j in range(h, max_len):
		Dcopy = D.copy()
		X[i,j] = D[-1]
		
		if d==2:		
			plt.plot(X[i,:,0], X[i,:,1], c='b')

			fig.savefig('./data/'+modelname+'/plots/'+flag+f'_repeated_trajs_{modelname}_map.png')
			plt.close()


print(X)

path = 'data/'+modelname+'/'+modelname+'_data_'+flag+'.pickle'
data_dict = {'trajs': X}
with open(path, "wb") as f:
	pickle.dump(data_dict, f)
	
