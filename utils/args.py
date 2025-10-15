import argparse
import torch as th
from enum import Enum

class Model(Enum):
    lightgrand = 'lightgrand'
    gcn = 'gcn'
    mlp = 'mlp'

    def __str__(self):
        return self.value

def argument():
    parser = argparse.ArgumentParser(description='lightGRAND')

    parser.add_argument("--data_input", default="./dataset", help='inputs of the component.')
    parser.add_argument("--data_output", default="output", help='model output address in KP.')

    # data source params
    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    parser.add_argument('--device', type=str, default="cpu", help='GPU index. Default: -1, using CPU.')
    # training params
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
    parser.add_argument('--patience', type=int, default=200, help='Training patience.')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed for np, torch and cuda')
    parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 reg.')
    # choose model
    parser.add_argument('--model', type=Model, choices=list(Model), default="lightgrand")
    
    # model params
    parser.add_argument("--hid_dim", type=int, default=32, help='Hidden layer dimensionalities.')
    #parser.add_argument('--dropnode_rate', type=float, default=0.5, help='Dropnode rate (1 - keep probability).')
    parser.add_argument('--input_droprate', type=float, default=0.5, help='dropout rate of input layer')
    parser.add_argument('--hidden_droprate', type=float, default=0.5, help='dropout rate of hidden layer')
    parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
    parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
    parser.add_argument('--lam', type=float, default=1., help='Coefficient of consistency regularization')
    parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
    parser.add_argument('--beta', type=float, default=1., help='Coefficient of consistency regularization')

    parser.add_argument('--num_partitions', type=int, default=2)
    parser.add_argument("--partitions_path_name", type=str, default="partitions2", help="partition num.")
    #parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--conf', type=int, default=7, help='n_classes')
   
    parser.add_argument('--global_period', type=int, default=1, help='1 / global frenquency')
    parser.add_argument('--global_noise', type=float, default=0.5, help='global random drop node for global noise')
    parser.add_argument('--R', type=int, default=3, help=" order of features(list) for training")
    #parser.add_argument('--trails1_global_mean', type=bool, default=False)   period 

    parser.add_argument('--local_period', type=int, default=1, help='1 / local frenquency')
    parser.add_argument('--local_noise', type=float, default=0.5, help='local random drop node for local noise')
    parser.add_argument('--order', type=int, default=8, help='Propagation step')

    args = parser.parse_args()
    
    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
    #if th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    args.partitions_path_name = "partitions"+str(args.num_partitions)

    
    #args.trails1 = False
    print("-- Using  Device", args.device)

    return args
