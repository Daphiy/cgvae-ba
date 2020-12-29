import argparse
from typing import Sequence, Any
#from docopt import docopt
from collections import defaultdict, deque
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json
import os
from GGNN_core import ChemModel
import utils
from utils import *
import pickle
import random
from numpy import linalg as LA
from rdkit import Chem
from copy import deepcopy
from rdkit.Chem import QED
import os
import time
from data_augmentation import *
from CGVAE import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default='ba')
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--num_epochs",type=int,default=10)
parser.add_argument("--hidden_dim",type=int,default=10)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--kl_tradeoff",type=float,default=0.3)
parser.add_argument("--optstep",type=int,default=0)

args = parser.parse_args()

model_args = {}
model = DenseGGNNChemModel(model_args,dataset=args.dataset,
batch_size=args.batch_size,
num_epochs=args.num_epochs,
hidden_size=args.hidden_dim,
lr=args.lr,
kl_trade_off_lambda=args.kl_tradeoff,
optimization_step=args.optstep)

model.train()

