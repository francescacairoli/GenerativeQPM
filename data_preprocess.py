import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

flag ='test'
fn = 'data/'+flag+'_crossroad_trajs.mat'

data = sio.loadmat(fn)

m = 31 # fix number of steps 

full_trajs = data['trajs'][0]

print(data['trajs'][0][0].shape)
n = full_trajs.shape[0]
d = full_trajs[0].shape[1]

X = np.empty((n,m,d))

fig = plt.figure()
for i in range(n):

	D = full_trajs[i]
	h = D.shape[0]
	if  h >= m:
		X[i] = D[:m]
	else:
		c = 0
		for j in range(m-h):
			Dcopy = D.copy()
			X[i,c] = D[j]
			X[i,c+1] = (D[j]+D[j+1])/2
			c = c+2
		X[i,c:] = D[j+1:]	

	
	plt.plot(X[i,:,0,], X[i,:,1], c='b')

fig.savefig('data/'+flag+'_cropped_trajs.png')
plt.close()

labels = np.zeros(n)
labels[n//3:2*n//3] = 1
labels[2*n//3:] = 2
path = 'data/crossroad_data_'+flag+'.pickle'
data_dict = {'trajs': X, 'labels': labels}
with open(path, "wb") as f:
	pickle.dump(data_dict, f)
