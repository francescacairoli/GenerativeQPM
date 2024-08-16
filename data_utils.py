import pickle
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


def load_train_data():
	path = 'data/crossroad_data_train.pickle'
	with open(path, "rb") as f:
		data = pickle.load(f)

	x = data['trajs']
	y = data['labels']

	return x, y

def load_calibr_data():
	path = 'data/crossroad_data_calibr.pickle'
	with open(path, "rb") as f:
		data = pickle.load(f)

	x = data['trajs']
	y = data['labels']

	return x, y

def load_test_data():
	path = 'data/crossroad_data_test.pickle'
	with open(path, "rb") as f:
		data = pickle.load(f)

	x = data['trajs']
	y = data['labels']

	return x, y