"""
    We refer to and reuse the cross-subgraph meta-learning framework G-Meta.
    link: https://github.com/mims-harvard/G-Meta
    Paper: Huang K, Zitnik M. Graph meta learning via local subgraphs[J]. Advances in Neural Information Processing Systems, 2020, 33: 5862-5874.

"""

import copy
from meta import Meta
from subgraph_data import Subgraphs
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import glob
import scipy.io as sio
import matplotlib.pyplot as plt


def collate(samples):
        graphs_spt, labels_spt, fea_spt, graph_qry, labels_qry, fea_qry = map(list, zip(*samples))
        return graphs_spt, labels_spt, fea_spt, graph_qry, labels_qry, fea_qry

def load_graph(root):
    #load meta-training graph and meta-testing graph
        train_file = glob.glob(root + "*0.mat")
        test_file = glob.glob(root + "*1.mat")

        print('train_graph: ', train_file)
        print('test_graph: ', test_file)
      
        train_graph = [sio.loadmat(str(train_file[0]))]
        target_graph = [sio.loadmat(str(test_file[0]))]

        return train_graph, target_graph

def main():
    dataset_name = args.dataset_name
    print("dataset: ", dataset_name)
    root = args.root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roc_list = []

    for run in range(10):
        train_graph, target_graph = load_graph(root)

        db_train= Subgraphs(args, 'meta-train', train_graph, device)
        db_test = Subgraphs(args, 'meta-test', target_graph, device)
        
        in_dim = db_train.in_dim
        maml = Meta(args, in_dim, device)
        print(maml)

        max_acc = 0
        model_max = copy.deepcopy(maml)
        print('training')
        
        for epoch in range(args.epoch):
            train_data = DataLoader(db_train, shuffle = True, num_workers = 0, pin_memory = False, collate_fn = collate) #load data

            accs_all_train = []
            for step, (x_spt, y_spt, fea_spt, x_qry, y_qry, fea_qry) in enumerate(train_data):

                rocs = maml(x_spt[0], fea_spt[0], y_spt[0], x_qry[0], fea_qry[0], y_qry[0])
                accs_all_train.append(rocs)
                print('Epoch: ', epoch+1, 'Step: ', step, 'Train roc: ', rocs)
            
            rocs = np.array(accs_all_train).mean(axis=0).astype(np.float16)

            if rocs[-1] > max_acc:
                max_acc = rocs[-1]
                model_max = copy.deepcopy(maml)
        
        test_data = DataLoader(db_test, shuffle = True, num_workers = 0, pin_memory = False, collate_fn = collate)
        rocs_all_test = []
        
        for i, (x_spt, y_spt, fea_spt, x_qry, y_qry, fea_qry) in enumerate(test_data):
            rocs, _, _ = model_max.finetune(x_spt[0], fea_spt[0], y_spt[0], x_qry[0], fea_qry[0], y_qry[0])
            rocs_all_test.append(rocs)

            print(i, ' Test roc: ', rocs)
        
        rocs_mean = np.array(rocs_all_test).mean(axis=0).astype(np.float16) 


        print(run, 'run Test roc: ', str(rocs_mean.mean(axis = 0)))

        roc_list.append(rocs_mean.mean(axis = 0))
    

    print('10 runs mean roc:', np.array(roc_list).mean(axis = 0).astype(np.float16))
        


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_name', type=str, help='name of dataset', default="pubmed")
    argparser.add_argument('--root', type=str, help='dataset path', default="/root/LHML/dataset/pubmed/")
    argparser.add_argument('--task_num', type=int, help='number of meta task', default=4)
    argparser.add_argument('--nb_batch', type=int, help='batch_size', default=20)
    argparser.add_argument('--batch_size', type=int, help='number of nodes for every batch', default=20)
    argparser.add_argument('--k_shot', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--sample_nodes', type=int, help='sample nodes if above this number of nodes', default=1000)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40)
    argparser.add_argument('--nb_batch_maml', type=int, help='maml batch', default=64)
    argparser.add_argument('--nb_item', type=int, help='maml batch', default=5)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine tune', default=5)
    argparser.add_argument('--hid_dim', type=int, help='hidden dim', default=64)
    argparser.add_argument('--h', type=int, help='neighborhood', default=2)
    args = argparser.parse_args()

    main()