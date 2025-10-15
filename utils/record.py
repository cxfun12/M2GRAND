import numpy as np
import torch as th
import pandas as pd
from datetime import datetime
import os
import sys
import copy

try:
    import cPickle as pickle
except:
    import pickle 
from collections import namedtuple

StepStrut = namedtuple("StepStrut",["loss_sup", "loss_consis", "total_loss", "train_acc_count", "num_nodes"])
EpochStrut = namedtuple("EpochStrut", ["epoch_id","Steps", "acc_train", "total_loss", "count", "acc_val", "acc_test", "acc_best_test"])
#EpochStrut = namedtuple("Epoch",["Clusters", "Eval", ["acc_train", "total_loss", "count", "acc_val", "acc_test", "acc_best_test"]])

class Recorder(object):
    def __init__(self, args=None, model=None, metric=None):
        self.args = args
        self.model = model
        self.metric = metric
        self.epochs = []
        self.best_test_acc = np.inf
        #self.save_path=  os.path.join("dataset" , args.dataname, "records") 
        self.save_path=  os.path.join("dataset" , "records") 


    def get_best_test_acc(self):
        if len(self.epochs) < 1:
            print(" Not found epoch records !!! ")
            return  0.001 
        
        self.best_test_acc = self.epochs[-1].acc_best_test 
        return self.best_test_acc

    def get_train_val_loss_from_test(self, test_acc):
        res = [(e.acc_train, e.total_loss, e.acc_val, e.acc_test) for e in self.epochs if e.acc_test == test_acc]
        print(res)
        return res[0]

        
    def epochToDataFrame(self):
        return pd.DataFrame(self.epochs)

    def save_model(self, model_save_dir="models"):
        model_name = "{}_{}.pkl".format( self.args.dataname , 
                                        int(self.get_best_test_acc()*1000))

        path = os.path.join("dataset" , self.args.dataname, model_save_dir, model_name)
        th.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model = th.load(path)
        return self.model

    def update(self, tmp_dict):
        self.__dict__.update(tmp_dict) 

    def save(self):
        if len(self.epochs) < 1:
            return
        
        file_name = "record_{}_{}.pkl".format( self.args.dataname, int(self.get_best_test_acc()*100000))

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        print(" ===== ", self.save_path, file_name, )
        with open(os.path.join(self.save_path, file_name), 'wb') as f:
            pickle.dump(self.__dict__, f)
            #pickle.dump(self.epochs[-1] , f)
        #f = open( os.path.join(self.save_path, file_name), 'wb')
        #pickle.dump(self.__dict__, f, 2)
            f.close()
    
    def drop_model(self):
        self.model = None

    @classmethod
    def loader(cls, file_path):
        with open(file_path, 'rb') as f:
            re = Recorder(); 
            tmp = pickle.load(f) 
            re.update(tmp)
            #re.update(pickle.load(f))

            return re

class RecorderPd(object):
    def __init__(self, args=None, model=None, metric=None, partition=None, stopwatch=None):
        self.args = args
        self.model = model
        self.partitions = partition #TODO: 暂时没有用， 要不要删除?
        self.metric = metric
        self.stopwatch = stopwatch
        if self.args != None and self.args.data_output!= None: 
            self.save_path=  os.path.join("dataset" , "records", self.args.data_output) 
        self.record_file_path = None
        self.best_model = None

    def robust_path(self, path):
        prefix, suffix =  path.split(".")
        dir_path =  os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if os.path.exists(path): 
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            path = "{}_{}.{}".format(prefix,timestamp, suffix)
        return path

    def save_best_model(self):
        #model_name = "{}_{}.pkl".format( self.args.dataname , int(self.metric.acc_best_test*1000))
        #path = os.path.join("dataset" , self.args.dataname, model_save_dir, model_name)
        #path = self.robust_path(path)
        #self.record_file_path = path
        #th.save(self.model.state_dict(), path)
        self.best_model = copy.deepcopy(self.model)
        th.cuda.empty_cache()

    def load_model(self, path):
        #model.load_state_dict(torch.load(model_path))
        #model.eval()  # 设置为评估模式
        self.model = th.load(path)
        return self.model

    def update(self, tmp_dict):
        self.__dict__.update(tmp_dict) 

    def save(self, use_robust_path=True, file_name=None):
        if file_name == None:
            file_name = "record_{}_{}.pkl".format( self.args.dataname, int(self.metric.acc_best_test*100000))

        path = os.path.join(self.save_path, file_name)

        if use_robust_path:
            path = self.robust_path(path)

        tag = "{}:{}".format( sys._getframe().f_code.co_name , sys._getframe().f_lineno) 
        print("Save Recoder ({}) at {} ".format(tag, path))

        with open(os.path.join(path), 'wb') as f:
            pickle.dump(self.__dict__, f)
            #pickle.dump(self.epochs[-1] , f)
        #f = open( os.path.join(self.save_path, file_name), 'wb')
        #pickle.dump(self.__dict__, f, 2)
            f.close()
    
    def drop_model(self):
        self.model = None

    @classmethod
    def loader(cls, file_path):
        with open(file_path, 'rb') as f:
            rpd = RecorderPd(); 
            rpd.update(pickle.load(f) )
            #re.update(pickle.load(f))
            return rpd 
