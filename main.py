import os
import numpy as np
import torch as th
import torch.nn.functional as F

from models.model import * 
from utils.preprocess import *
from utils.graphs import *
from utils.args import * 
from utils.data_load import *
from utils.record import *


def run(args):
    print("data intput: ", args.data_input)
    print("ARGS:  ", args)

    print("Load Data") 
    graph, features, labels, train_idx, val_idx, test_idx, idx_unlabel, n_classes, A = load_data(dataset_str=args.dataname, args=args)
    print(" Graph : Nodes={} Edges={}".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("To Device: ", args.device)
    graph.graph['device'] = args.device

    train_mask = torch.full((graph.number_of_nodes(),), False).index_fill_(0, train_idx, True).to(args.device)
    val_mask = torch.full((graph.number_of_nodes(),), False).index_fill_(0, val_idx, True).to(args.device)
    test_mask = torch.full((graph.number_of_nodes(),), False).index_fill_(0, test_idx, True).to(args.device)
    features = features.to(args.device)
    labels = labels.to(args.device)
    A = A.to(args.device)
    print("Matrix A: ", type(A))

    print("Init Sub Graphs") 
    #if check_subgraphs(args):
    #    Subgraphs = load_subgraphs(args)
    #else:
    #    Subgraphs = partition_graph_org_metis(graph, args) 
    #    save_subgraphs(Subgraphs, args)
    Subgraphs = init_subgraphs(graph, args) 

    ## lightGRAND
    print("Global Feature") 

    print("A  features: ", type(A), type(features), "features: ", features.shape)
    feat_list = preprocess_nx(features, A, args.R)
    print("feat_list: ", len(feat_list))
    feats = th.cat(feat_list, dim=1).to(args.device)

    n_features = feats.shape[-1] 
    print(f"features dim: {n_features}")
    #切换为使用metis做partition
    split_ndata_to_subgraphs(Subgraphs, features, feats, labels, train_mask, val_mask, test_mask)
    M = Metric(train_idx, val_idx, test_idx, 
               graph.number_of_nodes(), n_classes, labels, args)
    M.cid2nid = Subgraphs.cid2nid
    print("Init Model") 

    ####################
    ##初始化模型
    ####################
    print(" ------ ====== Using Model: ", args.model, type(args.model), args.model.value)
    model = lightGRAND_NX(n_features, args.hid_dim, n_classes, args.sample, args.order,
                  args.input_droprate, args.hidden_droprate, args).to(args.device)
    print("  Model :", model)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #RPD = RecorderPd(args, model, M, SubGraphs.cid2nid, SW)

    print("Training Model") 

    for epoch in range(args.epochs):
        model.train()
        M.reset() 

        if epoch % args.global_period == 0: 
            drop_feat = drop_node(features, args.global_noise, True) 
            feat_list = preprocess_nx(drop_feat, A, args.R)
            feats = th.cat(feat_list, dim=1).to(args.device)

        for subgraph in Subgraphs.subgraphs:
            input_nodes = Subgraphs.cid2nid[subgraph.cid]
            batch_input = feats[input_nodes]
            batch_train_mask = subgraph.ndata["train_mask"]
            batch_labels = subgraph.ndata["labels"]

            loss_sup = 0
            logits = model(subgraph, batch_input, batch_input, True, epoch, subgraph.cid)
            sample = args.sample + 1 

            for k in range(sample):
                loss_sup += F.nll_loss(logits[k][batch_train_mask], batch_labels[batch_train_mask])
            
            loss_sup = loss_sup / sample

            loss_consis = consis_loss(logits, args.tem)
            loss_train = loss_sup + args.lam * loss_consis 

            M.total_loss += loss_train.item(); M.count += 1
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            #th.cuda.empty_cache()

            M.counts(logits[0], batch_labels, batch_train_mask, "train") 
            M.record(epoch, subgraph.cid, loss_sup.item(), loss_consis.item(), loss_train.item())

        model.eval()   
        with th.no_grad():
            for subgraph in Subgraphs.subgraphs:
                input_nodes = Subgraphs.cid2nid[subgraph.cid]
                batch_input = feats[input_nodes]

                val_logits = model(subgraph, batch_input, batch_input, False)

                M.counts(val_logits, subgraph.ndata["labels"], subgraph.ndata["val_mask"], "val", subgraph.cid ) 
                M.counts(val_logits, subgraph.ndata["labels"], subgraph.ndata["test_mask"], "test", subgraph.cid) 
        M.bad_counts(epoch)
        M.print_metric(epoch)

        if M.bad_count == args.patience:
            print('Early stop! Min loss: ', M.loss_mn, ', Max accuracy: ', M.test_acc)
            print('Early stop model validation loss: ', M.val_acc_count / M.len_val_mask, ', accuracy: ', M.acc_best_test)
            break

    print("End Training") 
    return M.test_acc 

if __name__ == '__main__':
    args = argument()
    acc_best_test = run(args)
