"""
    We refer to and reuse the cross-subgraph meta-learning framework G-Meta.
    link: https://github.com/mims-harvard/G-Meta
    Paper: Huang K, Zitnik M. Graph meta learning via local subgraphs[J]. Advances in Neural Information Processing Systems, 2020, 33: 5862-5874.

"""

import torch
from torch.utils.data import Dataset
import glob
import random
import scipy.io as sio
import numpy as np
import itertools
import dgl
from utils import *
from sklearn import decomposition

class Subgraphs(Dataset):
    def __init__(self, args, mode, graphs, device) -> None:
        super().__init__()
        self.nb_batch = args.nb_batch #number of batch
        self.batch_size = args.batch_size 
        self.graph_l = []
        self.attr_l = []
        self.label_l = []
        self.target_label = None
        self.k_shot = args.k_shot #size of support set or query set
        self.nb_graphs = len(graphs)
        self.graphs = graphs
        self.device = device
        self.h = args.h # hop size of a selected starting node for generating a subgraph
        self.mode = mode # training or testing

        self.labeled_idx_l = []
        self.unlabeled_idx_l = []
        self.sample_nodes = args.sample_nodes #maximum size of a subgraph
        self.in_dim = 0 #dimension of the input attributes
    
        self.subgraph = []
        for _ in range(self.nb_graphs):
            self.subgraph.append({})

        for f in self.graphs:
            data = f
            adj = data['Network'] if ('Network' in data) else data['A']
            adj = dgl.from_scipy(adj)
            attr = data['Attributes'].astype(np.float) if ('Attributes' in data) else data['X']


            if type(attr) is np.ndarray:
                attr = torch.FloatTensor(attr)
            else:
                attr = torch.FloatTensor(attr.toarray())
            label = data['Label'].astype(np.float)  if ('Label' in data) else data['gnd']

            self.graph_l.append(adj)
            self.attr_l.append(attr)
            self.label_l.append(label)

        self.sampleAnomaly()

        if self.mode == 'meta-train':
            self.create_batch()

        elif self.mode == 'meta-test':
            self.create_batch_test()

        self.in_dim = self.attr_l[0].shape[1]

    def sampleAnomaly(self):
        # print statistcs
        for i in range(self.nb_graphs):
            label_tmp = self.label_l[i]

            print(self.mode, ' number of all nodes: ',len(self.label_l[i]))
            idx_ano = np.nonzero(label_tmp == 1)[0]
            idx_nor = np.nonzero(label_tmp == 0)[0]
            print(self.mode, ' number of anomalies:',len(idx_ano))
            print(self.mode, ' number of normal data: ',len(idx_nor))

            idx_labeled = np.array(idx_ano)
            self.labeled_idx_l.append(idx_labeled)

            idx_unlabeled = np.array(idx_nor)
            self.unlabeled_idx_l.append(idx_unlabeled)

    #create batch for training
    def create_batch(self):
        self.support_x_batch = [] #list every support set batch in training
        self.query_x_batch = [] #list every query set batch in training

        for _ in range(self.nb_batch):
            support_x = [] #store the selected node indexs for support set in every batch
            query_x = [] #store the selected nodes indexs for query set in every batch

            for i in range(self.nb_graphs):
                #20 nodes(subgraphs) for every batch, 15 anomalies, 5 normal nodes
                #meta-training: 10 anomalies in support set, 5 anomalies and 5 normal nodes in query set
                ano_idx = np.random.choice(self.labeled_idx_l[i], int(self.batch_size / 4 * 3), False) # 15 anomalies index
                nor_idx = np.random.choice(self.unlabeled_idx_l[i], int(self.batch_size / 4), False) # 5 normal nodes index
                np.random.shuffle(ano_idx)
                np.random.shuffle(nor_idx)
            
                spt_idx = np.array(ano_idx[:self.k_shot]) #10 anomalies in support set
                qry_idx = np.append(np.array(ano_idx[self.k_shot:]), np.array(nor_idx)) # 5 anomalies and 5 normal nodes
                support_x.append(spt_idx)
                query_x.append(qry_idx)

            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)

    #create batch for testing
    def create_batch_test(self):
        self.test_spt_batch = []
        self.test_qry_batch = []

        for _ in range(self.nb_batch):
            support = []
            query = []

            for i in range(self.nb_graphs):

                ano_idx = np.random.choice(self.labeled_idx_l[i], int(self.batch_size / 4 * 3), False)
                nor_idx = np.random.choice(self.unlabeled_idx_l[i], int(self.batch_size / 4), False)
                np.random.shuffle(ano_idx)
                np.random.shuffle(nor_idx)

                spt_idx = np.array(ano_idx[:self.k_shot])
                qry_idx = np.append(np.array(ano_idx[self.k_shot:]), np.array(nor_idx))
                support.append(spt_idx)
                query.append(qry_idx)
            
            self.test_spt_batch.append(support)
            self.test_qry_batch.append(query)

    def generate_subgraph(self, adjs, graph_idx, item):
        if item in self.subgraph:
            return self.subgraph[graph_idx][item]

        else:
            if self.h == 2:
                f_hop = [n.item() for n in adjs[graph_idx].in_edges(item)[0]]
                n_2 = [[n.item() for n in adjs[graph_idx].in_edges(i)[0]] for i in f_hop]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_2)) + f_hop + [item]))).numpy()
            elif self.h == 1:
                f_hop = [n.item() for n in adjs[graph_idx].in_edges(item)[0]]
                h_hops_neighbor = torch.tensor(list(set(f_hop + [item]))).numpy()
            elif self.h == 3:
                f_hop = [n.item() for n in adjs[graph_idx].in_edges(item)[0]] 
                n_2 = [[n.item() for n in adjs[graph_idx].in_edges(i)[0]] for i in f_hop]
                n_3 = [[n.item() for n in adjs[graph_idx].in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [item]))).numpy()

        if h_hops_neighbor.reshape(-1,).shape[0] > self.sample_nodes:
            h_hops_neighbor = np.random.choice(h_hops_neighbor, self.sample_nodes, replace=False)
            h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [item]))
        
        sub = dgl.node_subgraph(adjs[graph_idx], h_hops_neighbor, store_ids = True)
        nodes = sub.ndata[dgl.NID]

        self.subgraph[graph_idx][item] = (sub, nodes)
        return sub, nodes

    def __getitem__(self, index):
        
        batched_graph_spt = []
        support_feature = []
        support_y = []

        batched_graph_qry = []
        query_feature = []
        query_y = []

        if self.mode == 'meta-train':
            for task in range(self.nb_graphs):
                task_info_spt = [self.generate_subgraph(self.graph_l, task, item) for item in self.support_x_batch[index][task]] 
                graphs_spt = [i for i, j in task_info_spt] #generated subgraph list for support set
                nodes_spt = [j for i, j in task_info_spt] #generated subgraph node indexs for support set
                
                batched_graph_spt.append(dgl.batch(graphs_spt).to(self.device)) # subgraph batch
                feature_spt = np.vstack(list(self.attr_l[task][nodes_spt[i]] for i in range(self.k_shot)))
                y_spt = np.vstack(list(self.label_l[task][nodes_spt[i]] for i in range(self.k_shot)))

                support_feature.append(torch.FloatTensor(feature_spt).to(self.device)) # feature batch
                support_y.append(torch.LongTensor(y_spt).to(self.device)) # label batch


                task_info_qry = [self.generate_subgraph(self.graph_l, task, item) for item in self.query_x_batch[index][task]]
                graphs_qry = [i for i, j in task_info_qry]
                nodes_qry = [j for i, j in task_info_qry]

                batched_graph_qry.append(dgl.batch(graphs_qry).to(self.device))
                feature_qry = np.vstack(list(self.attr_l[task][nodes_qry[i]] for i in range(self.k_shot)))
                y_qry = np.vstack(list(self.label_l[task][nodes_qry[i]] for i in range(self.k_shot)))

                query_feature.append(torch.FloatTensor(feature_qry).to(self.device))
                query_y.append(torch.LongTensor(y_qry).to(self.device))
        
        elif self.mode == 'meta-test':
            for task in range(self.nb_graphs):
                task_info_spt = [self.generate_subgraph(self.graph_l, task, item) for item in self.test_spt_batch[index][task]]
                graphs_spt = [i for i, j in task_info_spt]
                nodes_spt = [j for i, j in task_info_spt]
                
                batched_graph_spt.append(dgl.batch(graphs_spt).to(self.device))
                feature_spt = np.vstack(list(self.attr_l[task][nodes_spt[i]] for i in range(self.k_shot)))
                y_spt = np.vstack(list(self.label_l[task][nodes_spt[i]] for i in range(self.k_shot)))

                support_feature.append(torch.FloatTensor(feature_spt).to(self.device))
                support_y.append(torch.LongTensor(y_spt).to(self.device))


                task_info_qry = [self.generate_subgraph(self.graph_l, task, item) for item in self.test_qry_batch[index][task]]
                graphs_qry = [i for i, j in task_info_qry]
                nodes_qry = [j for i, j in task_info_qry]

                batched_graph_qry.append(dgl.batch(graphs_qry).to(self.device))
                feature_qry = np.vstack(list(self.attr_l[task][nodes_qry[i]] for i in range(self.k_shot)))
                y_qry = np.vstack(list(self.label_l[task][nodes_qry[i]] for i in range(self.k_shot)))

                query_feature.append(torch.FloatTensor(feature_qry).to(self.device))
                query_y.append(torch.LongTensor(y_qry).to(self.device))

        return batched_graph_spt, support_y, support_feature, batched_graph_qry, query_y, query_feature

    def __len__(self):
        return self.nb_batch
