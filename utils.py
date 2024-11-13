def import_filenames(model_name, H, dim=2):

	n_train_states = 2000
	n_cal_states = 1000
	n_test_states = 25
	cal_hist_size = 10
	test_hist_size = 1000

	trainset_fn = "../data/"+model_name+"/"+model_name+"_train_set_H={}_{}x{}.pickle".format(H, n_train_states, cal_hist_size)
	testset_fn = "../data/"+model_name+"/"+model_name+"_test_set_H={}_{}x{}.pickle".format(H, n_test_states, test_hist_size)
	calibrset_fn = "../data/"+model_name+"/"+model_name+"_calibr_set_H={}_{}x{}.pickle".format(H, n_cal_states, cal_hist_size)

	traintraj_fn = "../data/"+model_name+"/"+model_name+"_train_trajs_H={}_{}x{}.pickle".format(H+1, n_train_states, cal_hist_size)
	testtraj_fn = "../data/"+model_name+"/"+model_name+"_test_trajs_H={}_{}x{}.pickle".format(H+1, n_test_states, test_hist_size)
	calibrtraj_fn = "../data/"+model_name+"/"+model_name+"_calibration_trajs_H={}_{}x{}.pickle".format(H+1, n_cal_states, cal_hist_size)

	return (trainset_fn, calibrset_fn, testset_fn), (traintraj_fn, calibrtraj_fn, testtraj_fn), (n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size)


def save_results_to_file(results_list, filepath, extra_info= ""):

	f = open(filepath+"/results"+extra_info+".txt", "w")
	for i in range(len(results_list)):
		f.write(results_list[i])
	f.close()


def get_model_details(opt):

	if opt.model_name == "crossroad":
		opt.species_labels = ["x", "y"]
		opt.target_dim = 2
		opt.x_dim = opt.target_dim
		opt.traj_len = 19
		opt.eval_length = opt.traj_len+1
	elif opt.model_name == "navigator":
		opt.species_labels = ["x", "y"]
		opt.target_dim = 2
		opt.x_dim = opt.target_dim
		opt.traj_len = 15
		opt.eval_length = opt.traj_len+1
	elif opt.model_name == "signal":
		opt.species_labels = ["t", "x"]
		opt.target_dim = 2
		opt.x_dim = opt.target_dim
		opt.traj_len = 21
		opt.eval_length = opt.traj_len+1
	elif opt.model_name == "pedestrian":
		opt.species_labels = ["x", "y"]
		opt.target_dim = 2
		opt.x_dim = opt.target_dim
		opt.traj_len = 22
		opt.eval_length = opt.traj_len+1
	else:
		opt.species_labels = []
		opt.target_dim = 1
		opt.x_dim = opt.target_dim
		opt.traj_len = 1
		opt.eval_length = opt.traj_len+1

	return opt
