
def get_model_details(opt):

	if opt.model_name == "MM":
		opt.species_labels = ["x", "y"]
		opt.target_dim = 2
		opt.eval_length = 31
	
	else:
		opt.species_labels = []
		opt.target_dim = 0
		opt.eval_length = 0

	return opt