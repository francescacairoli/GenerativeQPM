import os
import sys
import dill as pickle
import argparse
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
import torch
import torch.autograd as autograd
from torch.autograd import Variable

from dataset import get_dataloader
import json
import yaml


from model_details import *
from data_utils import *
from main_model import absCSDI
from csdi_utils import *

from NNClassifier import *
from Partition_CQR_bis import *

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--model_name", type=str, default="crossroad")
parser.add_argument("--unconditional", type=eval, default=False)#, action="store_true"
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--property_idx", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--load", default=False, type=eval)
parser.add_argument("--calload", default=False, type=eval)
parser.add_argument("--classifier", default=False, type=eval, help=" True = NN classif, False = kmeans") 
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--nb_trajs_to_plot", type=int, default=100)

args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)

args = get_model_details(args)

print(args)
path = "config/" + args.config
with open(path, "r") as f:
	config = yaml.safe_load(f)

config["train"]["epochs"] = args.nepochs
config["train"]["batch_size"] = args.batch_size
config["train"]["lr"] = args.lr

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))
print(config)

foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"

Xtrain, train_classes = load_train_data(args.model_name)
Xcal, cal_classes = load_calibr_data(args.model_name)
Xtest, test_classes = load_test_data(args.model_name)


train_classes = np.zeros(train_classes.shape)
cal_classes = np.zeros(cal_classes.shape)
test_classes = np.zeros(test_classes.shape)

print('data loaded !')

#CQR with GENERATIVE MODEL
Nclasses = 1
partition_fnc = lambda x: np.zeros(x.shape[0])
		

# load datasets and dataloaders
train_loader, test_loader, cal_loader = get_dataloader(
	model_name=args.model_name,
	eval_length=args.eval_length,
	target_dim=args.target_dim,
	seed=args.seed,
	nfold=args.nfold,
	batch_size=config["train"]["batch_size"],
	missing_ratio=config["model"]["test_missing_ratio"],
	scaling_flag=args.scaling_flag,
)


model = absCSDI(config, args.device,target_dim=args.target_dim).to(args.device)

print(f'Loading the pre-trained model with id {args.modelfolder}..')
model.load_state_dict(torch.load(foldername+ "model.pth"))



if args.model_name == 'crossroad':
	stl_fnc = lambda trajs: eval_crossroad_property(trajs, prop_idx = args.property_idx)
elif args.model_name == 'navigator':
	stl_fnc = lambda trajs: eval_navigator_property(trajs, prop_idx = args.property_idx)
else:  #signal
	stl_fnc = lambda trajs: eval_signal_property(trajs)


Ncal = 600
Ntest = 200
Ntrajs = 300

Rtest = stl_fnc(Xtest)
Rtest_res = Rtest.reshape((Ntest,Ntrajs)).detach().cpu().numpy()


res_path = foldername+f'{args.model_name}_base_results_property={args.property_idx}_class={args.classifier}.pickle'
if not args.calload:
	

	cqr = PartitionCQR(cal_classes, Xcal, cal_loader, num_cal_points=Ncal, stl_property = stl_fnc, partition_fnc=partition_fnc, trained_generator=model, num_classes=Nclasses, opt = args, quantiles = [args.epsilon/2, 1-args.epsilon/2], plots_path=foldername, load=args.load, nsamples=args.nsample)

	cpis, pis = cqr.get_cpi(test_loader, pi_flag = True)

	cov, eff = cqr.get_coverage_efficiency(test_classes, Rtest, cpis)
	print('CQR Coverage = ', cov)
	print('CQR Efficiency = ', eff)

	results = {'Rtest': Rtest, 'Rtest_res': Rtest_res,'calibr_scores': cqr.calibr_scores, 'pis': pis, 'cpis': cpis, 'cov': cov, 'eff': eff}

	with open(res_path, 'wb') as file:
		pickle.dump(results, file)

	cqr.plot_multimodal_errorbars(Rtest_res, pis, cpis, 'unimodal cqr - baseline', foldername, extra_info=str(args.property_idx)+f'_class={args.classifier}_base', model_name=args.model_name)


else:

	with open(res_path, 'rb') as file:
		D = pickle.load(file)

	cqr = PartitionCQR(cal_classes, Xcal, cal_loader, num_cal_points=Ncal, stl_property = stl_fnc, partition_fnc=partition_fnc, trained_generator=model, num_classes=Nclasses, opt = args, quantiles = [args.epsilon/2, 1-args.epsilon/2], plots_path=foldername, load=args.load, nsamples=args.nsample)
	
	eqr = cqr.get_eqr(Rtest_res)
	print('EQR = ', eqr)
	pi_cov, pi_eff = cqr.get_global_coverage_efficiency(test_classes, Rtest, D['pis'])
	print('PI Coverage = ', pi_cov)
	print('PI Efficiency = ', pi_eff)
	c_cov, c_eff = cqr.get_coverage_efficiency(test_classes, Rtest, D['pis'])
	cc_cov, cc_eff = cqr.get_coverage_efficiency(test_classes, Rtest, D['cpis'])
	print('class spec pi cov = ', c_cov, cc_cov)
	print('class spec pi eff = ', c_eff, cc_eff)
	cpi_cov, cpi_eff = cqr.get_global_coverage_efficiency(test_classes, Rtest, D['cpis'])
	print('CPI Coverage = ', cpi_cov)
	print('CPI Efficiency = ', cpi_eff)
	
	cqr.plot_multimodal_errorbars(D['Rtest_res'], D['pis'], D['cpis'], 'unimodal cqr - baseline', foldername, extra_info=str(args.property_idx)+f'_class={args.classifier}_base', model_name=args.model_name)
