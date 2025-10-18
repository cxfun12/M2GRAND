import networkx as nx
import time
import os
try:
  import metis
except ImportError:
  import pymetis as metis
import scipy.sparse as sp
import torch as th
import torch.nn.functional as F
import numpy as np
import pickle
from torch_geometric.utils import to_networkx
#from torchsummary import summary

#import dgl
#from dgl.transforms import metis_partition as metis_partition_dgl 
#from dgl import backend as FF
from collections import namedtuple

### TODO: 属性中不需要的也可以删除掉, 譬如norm?  nxgraph?
## graphinfo 是subgraphs的结构定义
GraphInfo = namedtuple('GraphInfo', ['nxgraph', 'norm', 'adj', 'cid', 'ndata', 'weight'])
SubGraphs = namedtuple('SubGraphs', ["cid2nid", #partition(cluster) id 转 graph的nid
                                    "nid2cid", #graph的nid到partition id 映射关系
                                    "subgraphs"
                                    ]) 
#print("partition graph ", partitions_path)
#part_adj, parts = partition_graph(adj, list(range(adj.shape[0])), partitions_num)
## G.nodes[0]["color"]
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def partition_graph_metis(graph, num_partitions):
    # type(graph) = networkx graph
    #idx_nodes = list(range(adj.shape[0])) # adj的顺序就是 node ID， 这里可能有问题
    if num_partitions <= 1:
      return 0, [0]*graph.number_of_nodes()

    if metis.__name__ == 'metis':
      edgecuts, parts = metis.part_graph(graph, num_partitions, )
    else:
      edgecuts, parts = metis.part_graph(num_partitions, graph)
    
    return edgecuts, parts 
  
def init_subgraphs(graph, args): 
    print(" Graph : Nodes={} Edges={}".format(graph.number_of_nodes(), graph.number_of_edges()))
    num_partitions = args.num_partitions

    #if check_subgraphs(args):
    #    edgecuts, parts = load_subgraphs(args)
    #else:
    #    edgecuts, parts = partition_graph_metis(graph, num_partitions) 
    #    save_subgraphs((edgecuts, parts), args)
    edgecuts, parts = partition_graph_metis(graph, num_partitions) 

    subgraphs = SubGraphs(
           cid2nid = dict([(cid, []) for cid in range(num_partitions)]), #partition id 转 graph的nid {0:[], 1:[]}
           nid2cid = parts, #graph的nid到partition id 映射关系
           subgraphs = []
    )

    #初始化cid, partitionid 到 nodes id的映射
    for node_id, cluster_id in enumerate(parts):  
      subgraphs.cid2nid[cluster_id].append(node_id)

    cut_edges_num = 0
    for cid in range(num_partitions):
      node_num = len(subgraphs.cid2nid[cid])
      subgraph = graph.subgraph( subgraphs.cid2nid[cid] )
      print("Subgraph {}: node_num = {} edge_num = {}".format(
            cid, 
            node_num, 
            subgraph.number_of_edges() )
      )
      cut_edges_num += subgraph.number_of_edges() 
      # TODO：查看加self loop的论文 add_self loop 是否需要？subgraph = dgl.add_self_loop( dgl.remove_self_loop( subgraph ) )
      #norm = subgraph.adj 
      #remove_self_loop, then add_self_loop 3partition下epoch 70时test acc由0.55 提升到0.65
      subgraph_tmp = subgraph.copy() 
      subgraph_tmp.remove_edges_from(nx.selfloop_edges(subgraph_tmp))
      subgraph_tmp.add_edges_from([(node, node) for node in subgraph_tmp.nodes])
     
      #正好对应？？GINN的论文， 需要加self_loop? 
      diag = np.array( subgraph_tmp.degree)[:,1]
      diag_square = np.power(diag, -0.5)
      norm = th.from_numpy(diag_square)
     
      #TODO: adj应该是0，1 不带权重的; 已经测试 adj[adj > 0] = 1 但是变化不大,因为weight_value[i] = norm[ indices[0][i] ] * norm[ indices[1][i] ] 间接做了
      adj = nx.to_scipy_sparse_array(subgraph_tmp)
      adj[adj > 0] = 1
      adj = sparse_mx_to_torch_sparse_tensor(adj).to(graph.graph['device'])
      #如果不nx.Graph， 就是freezen graph, 因为subgraph是grap的一个view, 不方便后面添加删除边
      #weight = norm.repeat(1, node_num) * adj * norm.t().repeat(node_num, 1)  
      #https://stackoverflow.com/questions/50666440/column-row-slicing-a-torch-sparse-tensor
      adj = adj.coalesce(); indices = adj.indices(); value =  adj.values()
      weight_value = th.zeros_like(value)
      #print("==== partition_graph_org_metis -> norm.shape, diag_square.shape, value.shape: ", norm.shape, diag_square.shape, value.shape)

      for i in range(indices.shape[1]):
        #row = indices[0] , indices[1]  
        weight_value[i] = norm[ indices[0][i] ] * norm[ indices[1][i] ]   

      weight = th.sparse_coo_tensor(indices, weight_value, adj.size()).coalesce()
      
      graphinfo = GraphInfo (
                nxgraph = None,# nx.Graph(subgraph), 
                              #norm = norm, adj = adj, 
                              norm = None, adj = None, 
                              weight= weight.to(graph.graph['device']), 
                              cid = cid, ndata={})
      subgraphs.subgraphs.append(graphinfo)


    print("--- ==== Cut_Edges_Num: ", edgecuts, cut_edges_num, indices.shape[1], 1 - cut_edges_num*1.0/graph.number_of_edges() )

    #print(" -------------------  ", type(subgraphs.cid2nid), subgraphs.cid2nid)
      
    return subgraphs

def _set_graphs_device(subgraphs, args):
    for i, graphinfo in enumerate(subgraphs.subgraphs):
        subgraphs.subgraphs[i] = graphinfo._replace(weight= graphinfo.weight.to(args.device))

def save_subgraphs(subgraphs, args):
    file_path = os.path.join(args.data_input, args.dataname, "partitions","{}.pkl".format(args.num_partitions))
    print(" --- save_subgraphs path : ", file_path)

    os.makedirs(os.path.join(args.data_input, args.dataname, "partitions"), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(subgraphs, f)

def load_subgraphs(args):
    file_path = os.path.join(args.data_input, args.dataname, "partitions","{}.pkl".format(args.num_partitions))
    print(" --- load_subgraphs path : ", file_path)
    with open(file_path, 'rb') as f:
        #edgecuts, parts = pickle.load(f)
        subgraphs = pickle.load(f)
    _set_graphs_device(subgraphs, args)
    #return (edgecuts, parts)
    return subgraphs 

def check_subgraphs(args):
    file_path = os.path.join(args.data_input, args.dataname, "partitions","{}.pkl".format(args.num_partitions))
    print(" Check {} exists {}", file_path, os.path.exists(file_path))
    return os.path.exists(file_path)

def partition_graph_dgl_metis(graph, num_partitions):
    subgraphs = SubGraphs(cid2nid = dict([(cid, []) for cid in range(num_partitions)]), #partition id 转 graph的nid {0:[], 1:[]}
                    nid2cid = [0]*graph.number_of_nodes(), #graph的nid到partition id 映射关系
                    nid2subnid = [0]*graph.number_of_nodes(), #graph的nid到 subgraph 的_nid； 与dgl.NID(_nid)不是同一类，dgl.NID放在ndata里， 自定义 
                    subgraphs = [],
                    vitural_id = [])
    
    #for cid, val in metis_partition_dgl(graph, num_partitions).items():
    #    nids = FF.asnumpy( val.ndata[dgl.NID] ).tolist()
    #    subgraphs.cid2nid[cid].extend( nids )

    #  ##construct subgraphs
    #    subgraph = graph.subgraph(nids)
    #    #subgraph = subgraph.remove_self_loop().add_self_loop()
    #    subgraph = subgraph.to_networkx()
    #    #subgraph = dgl.to_networkx( graph.subgraph(nids) )

    #    degs = graph.in_degrees().float().clamp(min=1)
    #    norm = th.pow(degs, -0.5).to(graph.device).unsqueeze(1)

    #    adj = graph.adjacency_matrix().to(graph.device)


    #    graphinfo = GraphInfo (nxgraph = subgraph, norm = norm, adj = adj, weight = None, cid = cid, ndata={})
    #    subgraphs.subgraphs.append(graphinfo)

    #    for subnid, node_id in enumerate(subgraphs.cid2nid[cid]): 
    #        subgraphs.nid2subnid[node_id] = subnid 
    #        subgraphs.nid2cid[node_id] = cid 

    return subgraphs

def split_ndata_to_subgraphs(subgraphs, features, feats, labels, train_mask, val_mask, test_mask):
    for  subgraph in subgraphs.subgraphs:
      input_nodes = subgraphs.cid2nid[subgraph.cid]

      subgraph.ndata["train_mask"] = train_mask[input_nodes] 
      subgraph.ndata["val_mask"] = val_mask[input_nodes] 
      subgraph.ndata["test_mask"] = test_mask[input_nodes] 
      subgraph.ndata["labels"] = labels[input_nodes] 

      #### TODO: 可能不需要赋值features & feats， 可以直接注释掉
      ####
      #subgraph.ndata["features"] = features[input_nodes] # raw node data   
      #subgraph.ndata["feats"] = feats[input_nodes] # progation node data 

    print(" subgraph.ndata[train_mask]  Device ", subgraph.ndata["train_mask"].device)


def partition_graph_METIS(adj, idx_nodes, num_clusters):
  start_time = time.time()
  num_nodes = len(idx_nodes)
  num_all_nodes = adj.shape[0]

  neighbor_intervals = []
  neighbors = []
  edge_cnt = 0
  edge_cut = 0
  edge_save = 0
  neighbor_intervals.append(0)
  #train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolinl()
  train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
  train_ord_map = dict()
  train_adj_lists = [[] for _ in range(num_nodes)]
  for i in range(num_nodes):
    rows = train_adj_lil[i].rows[0]
    # self-edge needs to be removed for valid format of METIS
    if i in rows:
      rows.remove(i)
    train_adj_lists[i] = rows
    neighbors += rows
    edge_cnt += len(rows)
    neighbor_intervals.append(edge_cnt)
    train_ord_map[idx_nodes[i]] = i

  if num_clusters > 1:
    if metis.__name__ == 'metis':
      _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
      _, groups = metis.part_graph(num_clusters, train_adj_lists)
  else:
    groups = [0] * num_nodes

  part_row = []
  part_col = []
  part_data = []
  parts = [[] for _ in range(num_clusters)]
  for nd_idx in range(num_nodes):
    gp_idx = groups[nd_idx]
    nd_orig_idx = idx_nodes[nd_idx]
    parts[gp_idx].append(nd_orig_idx)
    for nb_orig_idx in adj[nd_orig_idx].indices:
      nb_idx = train_ord_map[nb_orig_idx]
      if groups[nb_idx] == gp_idx:
        part_data.append(1)
        part_row.append(nd_orig_idx)
        part_col.append(nb_orig_idx)
        edge_save += 1
      else:
        edge_cut += 1 
  part_data.append(0)
  part_row.append(num_all_nodes - 1)
  part_col.append(num_all_nodes - 1)
  part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

  #print('Partitioning done. %f seconds.  minimal cut %f, num_clust=%d, cnt= %d cut=%d save=%d', time.time() - start_time,  edge_cut*1.0 / edge_cnt, num_clusters, edge_cnt, edge_cut, edge_save)
  print("Partitioning done. {} seconds.  minimal cut {}, num_clust={}, cnt= {} cut={} save={}".format(time.time() - start_time,  
                                                                                                      edge_cut*1.0 / edge_cnt, num_clusters, 
                                                                                                      edge_cnt, edge_cut, edge_save))
  return part_adj, parts

def preprocess_nx(features, A, R):
    with th.no_grad():
      x = features
      res = []
      res.append(features)

      for i in range(1, R+1):
        x = th.spmm(A, x).detach_()
        x = x + res[i-1]
        res.append(x) ##??

      return res 

def propagate(feature, A, order, is_layer=False):
  #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = []
    y_ = feature
    for i in range(order):
      x = th.spmm(A, x).detach_()
        #print(y.add_(x))
      y_.add_(x)
   
    if is_layer:
        return y_.div_(order+1.0).detach_()
    else:
        return th.cat(y, dim=1).detach_()        

def get_graph_diameter(graph):
    diameter = None 
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
        print("graph diameter: ", diameter)
    else:
        print("graph is not connected")
    return diameter

def print_layer_details(model):
    """
    打印模型每一层的参数量和形状
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'层名: {name}, 形状: {param.shape}, 参数量: {param.numel():,}')
            
    # 最后再打印一次总量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('-' * 50)
    print(f'总可训练参数量: {total_params:,}')
    
def print_model_parm_nums(model):
    """
    打印模型参数量和模型大小
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'模型总参数量: {total_params:,}')
    print(f'模型可训练参数量: {trainable_params:,}')
    
    # 计算模型大小 (假设参数类型为 float32)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'模型大小: {size_all_mb:.3f} MB')
    

def print_model(model):
    print_layer_details(model)
    print_model_parm_nums(model)
    
    


## 便利DGL的边与A进行逐元素对比
#s, d = graph.edges()
#num_err = 0
#for i in range(s.size()[0]): 
#	if s[i].item() == 591:
#		break
#	edge_id = graph.edge_ids(s[i], d[i])
#	edge_weight = graph.edata["weight"][edge_id]
#	if edge_weight != A[s[i],d[i]]:
#		num_err += 1
#		print("[", s[i].item(), ",", d[i].item(), "] = ", edge_weight == A[s[i],d[i]] , edge_weight.item(),  A[s[i],d[i]].item())
##
