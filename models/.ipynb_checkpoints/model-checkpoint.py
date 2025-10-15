import numpy as np
import torch as th
import torch.nn as nn
#import dgl.function as fn
import torch.nn.functional as F
import math
import random

from utils.graphs import propagate 


def soft_consis_loss(sample, logits, batch_train_mask, batch_labels, args):
    loss_sup = 0
    for k in range(sample):
        loss_sup += F.nll_loss(logits[k][batch_train_mask], batch_labels[batch_train_mask])
            
    loss_sup = loss_sup / sample

    loss_consis = consis_loss(logits, args.tem)
    
    return loss_sup + args.lam * loss_consis, loss_sup, loss_consis


def consis_loss(logps, temp):
    ps = [th.exp(p) for p in logps]
    ps = th.stack(ps, dim = 2)
    
    avg_p = th.mean(ps, dim = 2)
    #sharp_p = (th.pow(avg_p, 1./temp) / th.sum(th.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    sharp_avg_p = th.pow(avg_p, 1./temp)
    sharp_p = (sharp_avg_p / th.sum(sharp_avg_p, dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = th.mean(th.sum(th.pow(ps - sharp_p, 2), dim = 1, keepdim=True))

    return loss


def drop_node(feats, drop_rate, training):
    drop_rates = th.FloatTensor(np.ones( feats.shape[0] ) * drop_rate)
    
    if training:
        masks = th.bernoulli(1. - drop_rates).unsqueeze(1).to(feats.device)
        feats = masks * feats
    else:
        feats = feats * (1. - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn =False):
        super(MLP, self).__init__()

        #print("----===(nfeat, nhid:, ", nfeat, nhid, nclass)
        #print("----===(nfeat, nhid:, ", type(nfeat), type(nhid), type(nclass))
        
        self.layer1 = nn.Linear(nfeat, nhid, bias = True)
        self.layer2 = nn.Linear(nhid, nclass, bias = True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
    
    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        
    def forward(self, x):
        if self.use_bn: 
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))
        
        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        #return x   
        return th.log_softmax( x , -1)


class lightGRAND_NX(nn.Module):
    r"""
    with global-local consistency regularization
    ALEX: 测试forward不要动态Parms, 都先在Init初始化好
    """
    def __init__(self, in_dim, hid_dim, n_class, S = 1, O = 3, # node_dropout=0.0,
        input_droprate = 0.0, hidden_droprate = 0.0, args = None):

        super(lightGRAND_NX, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.S = S
        self.O = O # order...
        self.n_class = n_class
        
        self.mlp = MLP(in_dim, hid_dim, n_class, input_droprate, hidden_droprate, args.use_bn)
        
        self.args = args
        self.drop_feat_dict = dict([(cid, []) for cid in range(args.num_partitions)]) 

    def forward(self, graph, feats, ax, training=True, epoch=-1, cid=-1):
        if training: # Training Mode
            output_list = []
            for s in range(self.S):

                if epoch % self.args.local_period == 0:
                    drop_feat = drop_node(feats, self.args.local_noise, True)  # Drop node
                    self.drop_feat_dict[cid] = drop_feat
                else:
                    drop_feat = self.drop_feat_dict[cid]

                feat = GRANDConv_NX(drop_feat, graph.weight, self.O) ### 
                output_list.append(th.log_softmax(self.mlp(feat), dim=-1))  # Prediction
            output_list.append(th.log_softmax(self.mlp(ax), dim=-1))

            return output_list
        else:   # Inference Mode
            drop_feat = drop_node(feats, self.args.local_noise, False)  # Drop node
            feat = GRANDConv_NX(drop_feat, graph.weight, self.O)
            res = th.log_softmax(self.mlp(feat), dim = -1)

            return res    


def GRANDConv_NX(feats, weight, order):
    x = feats
    y = 0+feats
    for i in range(order):
      #x = th.spmm(weight, x).detach_()
      x = th.spmm(weight, x)
      y.add_(x)
   
    return y / (order+1.0)
