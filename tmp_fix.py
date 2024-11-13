import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

flag ='test'
modelname = 'pedestrian'
path = 'data/'+modelname+'/'+modelname+'_data_'+flag+'.pickle'

with open(path, "rb") as f:
	data = pickle.load(f)

full_trajs = data['trajs']
print(full_trajs.shape)

X = np.vstack((full_trajs, full_trajs, full_trajs))
print(X.shape)

newpath = 'data/'+modelname+'/'+modelname+'_data_calibr.pickle'
data_dict = {'trajs': X}
with open(newpath, "wb") as f:
	pickle.dump(data_dict, f)
	
