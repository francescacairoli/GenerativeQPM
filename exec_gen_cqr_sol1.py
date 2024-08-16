import os
import sys
import pickle
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


from Partition_CQR_bis import *

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--model_name", type=str, default="MM")
parser.add_argument("--unconditional", type=eval, default=False)#, action="store_true"
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--property_idx", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--load", default=True, type=eval)
parser.add_argument("--epsilon", type=float, default=0.1)
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

args = get_model_details(args)

path = "config/" + args.config
with open(path, "r") as f:
	config = yaml.safe_load(f)

config["train"]["epochs"] = args.nepochs
config["train"]["batch_size"] = args.batch_size
config["train"]["lr"] = args.lr

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"


# load datasets and dataloaders
train_loader, test_loader, cal_loader = get_dataloader(
	model_name=args.model_name,
	eval_length=args.eval_length,
	target_dim=args.target_dim,
	seed=args.seed,
	nfold=args.nfold,
	batch_size=config["train"]["batch_size"],
	missing_ratio=config["model"]["test_missing_ratio"],
	scaling_flag=args.scaling_flag
)



model = absCSDI(config, args.device,target_dim=args.target_dim).to(args.device)
print(f'Loading the pre-trained model with id {args.modelfolder}..')
model.load_state_dict(torch.load(foldername+ "model.pth"))


Xtrain, _ = load_train_data()
Xcal, _ = load_calibr_data()
Xtest, _ = load_test_data()
Xtest_fixed, _ = load_test_fixed_data()

stl_fnc = lambda trajs: eval_crossroad_property(trajs, prop_idx = args.property_idx)

Rtest = stl_fnc(Xtest)
Rtest_fixed = stl_fnc(Xtest_fixed)

print(Rtest.shape)
print(Rtest_fixed.shape)

#Rtest_res = Rtest.reshape((150,200)).detach().numpy()

Rtest_fixed_res = Rtest_fixed.reshape((50,600)).detach().numpy()

print('Avg rob mode 0', np.mean(Rtest_fixed_res[-1,:200]))
print('Avg rob mode 1', np.mean(Rtest_fixed_res[-1,200:400]))
print('Avg rob mode 2', np.mean(Rtest_fixed_res[-1,400:]))

if False:
	plot_partition((Xtrain,Xcal,Xtest), crossroad_partition, foldername)


#CQR with GENERATIVE MODEL
cal_classes = crossroad_partition(Xcal)
test_classes = crossroad_partition(Xtest)

cqr = PartitionCQR(cal_classes, Xcal, cal_loader, num_cal_points=120, stl_property = stl_fnc, partition_fnc=crossroad_partition, trained_generator=model, num_classes=3, opt = args, quantiles = [args.epsilon/2, 1-args.epsilon/2], plots_path=foldername)

cpis, pis = cqr.get_cpi(test_loader, pi_flag = True)

cov, eff = cqr.get_coverage_efficiency(test_classes, Rtest, cpis)
print('CQR Coverage = ', cov)
print('CQR Efficiency = ', eff)
cqr.plot_multimodal_errorbars(Rtest_fixed_res, pis, cpis, 'multimodal cqr', foldername, extra_info=str(args.property_idx))

#cqr.plot_errorbars(Rtest_res, pis, cpis, 'multimodal cqr', foldername, extra_info=str(args.property_idx))
