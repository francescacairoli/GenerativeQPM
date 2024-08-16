import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import time
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from tqdm import tqdm
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from utils import * # import-export methods
from dataset import get_dataloader
import json
import yaml


from model_details import *

from main_model import absCSDI
from csdi_utils import *



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
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--load", default=False, type=eval)

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
if args.modelfolder == "":
    args.modelfolder = str(np.random.randint(0,500))
    foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
else:

    foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"
    os.makedirs(foldername, exist_ok=True)


# load datasets and dataloaders
train_loader, test_loader, _ = get_dataloader(
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


if not args.load:
	print('Training...')
	st = time.time()
	train(
		model,
		config["train"],
		train_loader,
		valid_loader=test_loader,
		foldername=foldername,
	)
	print('Training time: ', time.time()-st)

else:
	print(f'Loading the pre-trained model with id {args.modelfolder}..')
	model.load_state_dict(torch.load(foldername+ "model.pth"))

# Evaluate over the test set
if not args.load:
	print('Evaluating...')
	#evaluate(model, test_loader, nsample=1, scaler=1, foldername=foldername, ds_id = 'test')
	
	_ = light_evaluate(model, test_loader, nsample=1, foldername=foldername, ds_id = 'test')

plot_rescaled_crossroads(opt=args, foldername=foldername, dataloader=test_loader, nsample=args.nsample, idx='test')
