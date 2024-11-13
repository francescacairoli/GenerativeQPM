import numpy as np

def get_model_details(opt):

	if opt.model_name == "crossroad":
		opt.species_labels = ["x", "y"]
		opt.target_dim = 2
		opt.eval_length = 20
		n = 51;
		w = 15;
		M = np.ones((n,n))
		M[:w,:w] = 0
		M[n-w:,:w] = 0
		M[:w,n-w:] = 0
		M[n-w:,n-w:] = 0

		M[:w,n//2] = 0
		M[n-w:n,n//2] = 0
		M[n//2,:w] = 0
		M[n//2,n-w:] = 0
		opt.gridmap = M
	elif opt.model_name == "navigator":
		opt.species_labels = ["x", "y"]
		opt.target_dim = 2
		opt.eval_length = 16
		n = 30
		M = np.ones((n,n))

		M[n-10:n-4,4:10] = 0
		M[n-20:n-13, 9:16] = 0
		M[n-12:n-7,17:22] = 0
		M[n-25:n-19,19:25] = 0

		opt.gridmap = M
	elif opt.model_name == "signal":
		opt.species_labels = ["t", "x"]
		opt.target_dim = 2
		opt.eval_length = 22
		opt.gridmap = None
	elif opt.model_name == "pedestrian":
		opt.species_labels = ["x", "y"]
		opt.target_dim = 2
		opt.eval_length = 23
		opt.gridmap = None
	else:
		opt.species_labels = []
		opt.target_dim = 0
		opt.eval_length = 0

	return opt
