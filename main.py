import argparse
import os
import random
import numpy as np
import torch
from run import Run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fakesv', help='fakesv/fakett')
parser.add_argument('--mode', default='train')
parser.add_argument('--epoches', type=int, default=30)
parser.add_argument('--batch_size', type = int, default=128)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--gpu', default='0')
parser.add_argument('--lr', type=float,default=5e-5)
parser.add_argument('--alpha',type=float,default=0.5)
parser.add_argument('--beta',type=float,default=0.1)
parser.add_argument('--gamma',type=float,default=3.0)
parser.add_argument('--path_ckp', default= './checkpoints/')
parser.add_argument('--path_tb', default= './tensorboard/')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING']='1'
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print (args)

config={
    'dataset':args.dataset,
    'mode':args.mode,
    'epoches':args.epoches,
    'batch_size':args.batch_size,
    'early_stop':args.early_stop,
    'device':args.gpu,
    'lr':args.lr,
    'alpha':args.alpha,
    'beta':args.beta,
    'gamma':args.gamma,
    'path_ckp':args.path_ckp,
    'path_tb':args.path_tb
}

if __name__ == '__main__':
    Run(config = config
        ).main()
