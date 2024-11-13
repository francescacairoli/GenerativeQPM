import argparse
import torch
import datetime
import time
import json
import yaml
import sys
import os


#sys.path.append(".")
#current_dir = os.path.dirname(os.path.realpath(__file__))
#parent_dir = os.path.dirname(current_dir)
#sys.path.append(parent_dir)

import torch_two_sample 
from main_model import absCSDI
from dataset import *
from csdi_utils import *

from utils import *

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
#parser.add_argument("--target_dim", type=int, default=2)
#parser.add_argument("--eval_length", type=int, default=33)
parser.add_argument("--model_name", type=str, default="Crossroad")
parser.add_argument("--unconditional", default=False)#, action="store_true"
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--ntrajs", type=int, default=10)
parser.add_argument("--nepochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--q", type=float, default=0.9)
parser.add_argument("--load", type=eval, default=False)
parser.add_argument("--rob_flag", type=eval, default=False)
args = parser.parse_args()


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
if args.modelfolder == "":
    args.modelfolder = str(np.random.randint(0,500))
    foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
else:
    foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"


train_loader, valid_loader, test_loader = get_dataloader(
    model_name=args.model_name,
    eval_length=args.eval_length,
    target_dim=args.target_dim,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    scaling_flag=args.scaling_flag
)

print('-----', train_loader.dataset.observed_values.shape)
model = absCSDI(config, args.device,target_dim=args.target_dim).to(args.device)

if not args.load:
    st = time.time()
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
    print('Training time: ', time.time()-st)

else:
    model.load_state_dict(torch.load(foldername+ "model.pth"))

# Evaluate over the test set
if not args.load:
    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'test')

plot_rescaled_planners(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample, idx = 'test')
