import argparse
from CGVAE import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default='ba')
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--num_epochs",type=int,default=10)
parser.add_argument("--hidden_dim",type=int,default=5)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--kl_tradeoff",type=float,default=0.3)
parser.add_argument("--optstep",type=int,default=0)

args = parser.parse_args()

model_args = {'dataset':args.dataset,
              'batch_size':args.batch_size,
              'num_epochs':args.num_epochs,
              'hidden_size':args.hidden_dim,
              'lr':args.lr,
              'kl_trade_off_lambda':args.kl_tradeoff,
              'optimization_step':args.optstep,
              }

model = DenseGGNNChemModel(model_args,)

model.train()

