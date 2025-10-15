import numpy as np
import time
import torch as th
import pandas as pd
import scipy.sparse as sp
#import dgl.sparse as dglsp
import fnmatch
import os


from collections import Counter, namedtuple
from ogb.nodeproppred import Evaluator

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize


def to_binary_bag_of_words(features):
    """Converts TF/IDF features to binary bag-of-words features."""
    features_copy = features.tocsr()
    features_copy.data[:] = 1.0
    return features_copy


def normalize_adj(A):
    """Compute D^-1/2 * A * D^-1/2."""
    # Make sure that there are no self-loops
    A = eliminate_self_loops(A)
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def renormalize_adj(A):
    """Renormalize the adjacency matrix (as in the GCN paper)."""
    A_tilde = A.tolil()
    A_tilde.setdiag(1)
    A_tilde = A_tilde.tocsr()
    A_tilde.eliminate_zeros()
    D = np.ravel(A.sum(1))
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def row_normalize(matrix):
    """Normalize the matrix so that the rows sum up to 1."""
    return normalize(matrix, norm='l1', axis=1)


def add_self_loops(A, value=1.0):
    """Set the diagonal."""
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.
    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.
    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.
    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.
    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].
    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.
    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.
    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def remove_underrepresented_classes(g, train_examples_per_class, val_examples_per_class):
    """Remove nodes from graph that correspond to a class of which there are less than
    num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.
    Those classes would otherwise break the training procedure.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)


#def symmetric_normalize_adjacency(graph):
#    """Symmetric normalize graph adjacency matrix."""
#    indices = th.stack(graph.edges())
#    n = graph.num_nodes()
#    adj = dglsp.spmatrix(indices, shape=(n, n))
#
#    #adj_neg = 1 - adj.to_dense() - np.eye(graph.num_nodes())
#    adj_neg = ( 1 - adj.to_dense() + np.eye(graph.num_nodes()) ).to_sparse() #需要由torch.sparse_coo 变成 dgl.sparse.sparse_matrix.SparseMatrix
#    dst = adj_neg.indices()[0]; src  = adj_neg.indices()[1];
#    adj_neg = dglsp.from_coo(dst, src)
#
#    #https://docs.dgl.ai/en/2.0.x/generated/dgl.sparse.from_coo.html#dgl.sparse.from_coo
#    #torch.sparse_coo.to_coo()
#    #dst = torch.tensor([1, 1, 2])
#   #src = torch.tensor([2, 4, 3])
#   #A = dglsp.from_coo(dst, src)
#
#    deg_invsqrt = dglsp.diag(adj.sum(0)) ** -0.5
#    deg_invsqrt_neg = dglsp.diag(adj_neg.sum(0)) ** -0.5
#
#    return deg_invsqrt @ adj @ deg_invsqrt

class IterableQueue(object): 
    def __init__(self , psize = 5, sample = 4):
        self.maxsize = psize
        self.sample = sample
        self.queue = [[object for j in range(sample)] for i in range(self.maxsize)] 
        self.idx_sample = [ 0 for i in range(self.maxsize)] 

    def put(self, idx, item):
        self.queue[idx][self.idx_sample[idx]] = item
        self.idx_sample[idx] = (self.idx_sample[idx]+1) % self.sample

    def get(self, idx, slice):
        return self.queue[idx][:slice]

def list_files_with_keyword(directory, keywords):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue
            if not any(keyword in file for keyword in keywords):
                continue
            #print(" ===== ", file_path)
            file_list.append(file_path)
    return file_list

class Metric(object):
    # https://github.com/benedekrozemberczki/ClusterGCN/blob/master/src/clustergcn.py#L104
    def __init__(self, train_idx, val_idx, test_idx
                 ,node_num, n_classes, labels, args):
        #self.len_train_mask = sum(graph.ndata["train_mask"]).item();
        #self.len_val_mask = sum(graph.ndata["val_mask"]).item();
        #self.len_test_mask = sum(graph.ndata["test_mask"]).item();
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.len_train_mask = len(train_idx)
        self.len_val_mask = len(val_idx)
        self.len_test_mask = len(test_idx)
        self.acc_best_test = 0

        self.bad_count = 0
        self.loss_mn = np.inf
        self.best_epoch = 0

        #其实就是graph的节点数目
        self.predicts = th.zeros(node_num, dtype=th.bool).to(args.device) 
        self.bottelnecks = None
        self.cid2nid = None
        ### 注意 命名与使用不一致, 测试新加一个bottelnecks acc是否影响老record读取
        self.cid2nidWObn = None# self.filter_bottelnecks_from_cid2nid() 实际被用来记录bottelneck的test acc精度, 是一个list [分母是bottelneck的节点数， 分母是总节点数]
        self.bottelnecks_acc = [0.0, 0.0]
        self.df = pd.DataFrame(columns=['epoch', 'cid', 'loss_sup', 
                                        'loss_consis', 'loss_train', 'train_acc',
                                        'train_loss', 'val_acc', 'test_acc',
                                        'best_test_acc' ])

    
        self.evaluator = Evaluator(name= "-".join(args.dataname.split("_"))) if "ogbn" in args.dataname else None
        self.pred_ogb = th.zeros([node_num, n_classes]).to(args.device) 
        self.labels = labels

        self.reset()

    def reset(self):
        self.accumulated_training_loss = 0
        self.train_acc_count = 0
        self.val_acc_count = 0
        self.test_acc_count = 0
        self.total_loss=0
        self.count=0
#train_acc_count += th.sum(logits[0][batch_train_mask].argmax(dim=1) == batch_labels[batch_train_mask]).item()# / len(train_idx)


# M.counts(logits[0], batch_labels, batch_train_mask, "train") #
    def counts(self, p,l,m, type="train", cid=-1): 
        res = p[m].argmax(dim=1) == l[m]
        acc_num = th.sum(res).item()

        if type == "train": 
            self.train_acc_count += acc_num
        if type == "val": 
            self.val_acc_count += acc_num
        if type == "test": 
            self.test_acc_count += acc_num

        #在 print_metric 打印值
        if cid < 0:
            return 0 
        #print(" **** ", res.shape,  len(self.cid2nid[cid]), m.shape, m)
        tmp_m = th.zeros_like(m, dtype=th.bool)
        tmp_m[m] = res
        self.predicts[self.cid2nid[cid]] = tmp_m 


    def filter_bottelnecks_from_cid2nid(self):
        cid2nidWObn = dict([(key, []) for key in  self.cid2nid.keys()]) 
        cid2nid_set = dict([(key, set(self.cid2nid[key])) for key in  self.cid2nid.keys()]) 
        bottelnecks_set = dict([(key, set(self.bottelnecks[key])) for key in  self.cid2nid.keys()]) 

        for key in self.cid2nid.keys():
            cid2nid_without_bn = cid2nid_set[key] - bottelnecks_set[key] &  cid2nid_set[key] 
            cid2nidWObn[key].extend( list(cid2nid_without_bn) )
        
        return cid2nidWObn

        #return " ".join(result)

    # M.record(epoch, subgraph.cid, loss_sup.item(), loss_consis.item(), 
    # loss_train.item(), M.total_loss)
    def record(self, epoch, cid, loss_sup, loss_consis, loss_train):
    #self.df = pd.DataFrame(columns=['epoch', 'cid', 'loss_sup', 'loss_consis', 'loss_train', 
                                    #'train_acc',
                                     #   'train_loss', 
                                     # 'val_acc', 
                                     # 'test_acc',
                                      #  'best_test_acc' ])

        self.df.loc[len(self.df)] = [epoch, cid, loss_sup, loss_consis, loss_train,  
                                     self.train_acc_count / self.len_train_mask,
                                     self.total_loss/self.count,
                                     self.val_acc_count / self.len_val_mask,
                                     self.test_acc_count / self.len_test_mask,
                                     self.acc_best_test]

    #def bad_counts(self, loss, acc, epoch): 
    def bad_counts(self, epoch): 
        #需要 acc_best_test 和 test_acc 所以放到print_metric之前， 才能执行
        self.test_acc = self.test_acc_count / self.len_test_mask
        self.is_best_acc = self.test_acc >= self.acc_best_test 

        ##记录一下val_acc和test_acc 避免都为0
        #self.df.loc[len(self.df)-1]['val_acc'] = self.val_acc_count / self.len_val_mask
        #self.df.loc[len(self.df)-1]['test_acc'] = self.test_acc
        self.df.loc[len(self.df)-1, 'val_acc'] = self.val_acc_count / self.len_val_mask
        self.df.loc[len(self.df)-1, 'test_acc'] = self.test_acc

        if self.total_loss <= self.loss_mn or self.is_best_acc:
            self.best_epoch = epoch
            self.acc_best_test = max(self.test_acc, self.acc_best_test)
            self.loss_mn = min( self.total_loss, self.loss_mn )
            self.bad_count = 0
        else:
            self.bad_count += 1

    def print_metric(self, epoch):
        #print("met: val_acc={}/{}, test_acc={}/{}".format(self.val_acc_count, self.len_val_mask, self.test_acc_count, self.len_test_mask))
        print("In epoch {}, Train Acc: {:.4f} | Train Loss: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}, Best Test Acc: {:.4f}".
              format(epoch, self.train_acc_count / self.len_train_mask, 
                     self.total_loss/self.count, 
                     self.val_acc_count / self.len_val_mask,
                     self.test_acc,
                     self.acc_best_test))

        #if self.acc_best_test >  self.bottelnecks_acc[0]:
        #     self.bottelnecks_acc = [self.acc_best_test, bottelnecks_num, bottelnecks_acc, bottelnecks_acc_total]
        #     
        #print("\t" + " ".join(list(map(str, self.bottelnecks_acc))))



Watch = namedtuple('Watch', ['t', 'tag_name'])

class StopWatch(object):
    def __init__(self, type=""):
        self.watchs = []
        watch = Watch(t=time.time(), tag_name="Start") 
        self.watchs.append(watch)
        self.preprocess_time = None
        self.total_time = None

    def add_stop_watch_tag(self, tag_name="watch_name"):
        watch = Watch(t=time.time(), tag_name=tag_name) 
        self.watchs.append(watch)

    def show_watchs(self):
        watch_list_str = []

        for i, watch in enumerate(self.watchs):
            if i == 0: continue

            watch_list_str.append("{} time elapsed: {:.4f}s ".format(self.watchs[i-1].tag_name, (watch.t - self.watchs[i-1].t)))

        self.preprocess_time = self.watchs[-2].t - self.watchs[0].t  
        self.total_time = self.watchs[-1].t - self.watchs[0].t  

        print("\n".join(watch_list_str))
        print("Preprocess time elapsed: {:.4f}s \n --------".format(self.preprocess_time))
        print("Total time elapsed: {:.4f}s".format(self.total_time))

def find_files(directory, patterns):
    ''' 定义要搜索的目录
        search_directory = '/home/alex/research/GNN/lightgrand/dataset/records/'
        定义文件名模式
        patterns = ['*cora_85*', '*cora_86*', '*cora_87*', '*cora_848*', '*cora_849*']
    '''
    # 创建一个空列表来保存匹配的文件路径
    matched_files = []
    # 遍历指定目录及其所有子目录
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 检查文件名是否匹配任意一个给定的模式
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    # 如果匹配，将文件的完整路径添加到列表中
                    matched_files.append(os.path.join(root, filename))
                    break  # 匹配到任何一个模式后，跳出内层循环

        for dir_ in dirs:
            matched_files.extend(find_files(dir_, patterns))

    return matched_files
# 调用函数并获取匹配的文件列表
#matching_files = find_files(search_directory, patterns)
# 打印匹配的文件路径
#for file_path in matching_files:
#    print(file_path)



# 获取实例 A 和 B 的属性字典
# dict_A = vars(A)
# dict_B = vars(B)

# 比较两个实例的属性字典，找出不同的属性
# diff_properties = [key for key in dict_A if dict_A[key] != dict_B[key]]
# print("实例 A 和 B 的属性值不同的属性：", diff_properties)

#dict_A = vars(RecorderPd.loader(kaola[1]).args)
#dict_B = vars(RecorderPd.loader(kaola[2]).args)
#diff_properties = [(key,dict_A[key],":",dict_B[key]) for key in dict_A if dict_A[key] != dict_B[key]];print(diff_properties)
#diff_properties = ["{}  {}:{}".format(key,dict_A[key],dict_B[key]) for key in dict_A if dict_A[key] != dict_B[key]];print(diff_properties)
