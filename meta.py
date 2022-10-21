"""
    MAML framework for meta-learning.
    
"""
from copy import deepcopy
import torch
from torch import nn
from torch import optim
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from learner import *
from utils import DropGrad


def loss_func(dev, y_true):
    alpha = 0.9
    loss = -torch.mean(alpha * y_true * torch.log(dev) + (1-alpha) * (1-y_true) * torch.log(1-dev)) # weighted cross-entropy loss
  
    return loss

def auc_func(y_pred, y_true):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.flatten()

    y_pred = y_pred.flatten()

    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    return roc_auc

class Meta(nn.Module):
    def __init__(self, args, in_dim, device):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr #subgraph-level updating rate
        self.meta_lr = args.meta_lr # meta-learning rate
        self.update_step = args.update_step 
        self.update_step_test = args.update_step_test
        self.in_dim = in_dim
        self.hid_dim = args.hid_dim
        self.device = device
        self.dropout = DropGrad(rate=0.5, schedule='constant') # gradient dropout

        self.net = Classifier(self.in_dim, self.hid_dim).to(self.device)
        self.meta_optim = optim.Adam(params=self.net.parameters(), lr=self.meta_lr, weight_decay=5e-4) # optimizer
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.meta_optim, T_0=5, T_mult=2, eta_min=0, last_epoch=-1) # learning rate decay


    def forward(self, x_spt, fea_spt, y_spt, x_qry, fea_qry, y_qry):
        nb_graphs = len(x_spt)
        
        losses_s = [0 for _ in range(self.update_step)] #losses for support sets
        losses_q = [0 for _ in range(self.update_step + 1)] #losses for query sets
        rocs = [0 for _ in range(self.update_step + 1)]

        for i in range(nb_graphs):
            dev, radius = self.net(x_spt[i], fea_spt[i], vars = None)

            loss = loss_func(dev, y_spt[i])
            losses_s[0] += loss

            grad = torch.autograd.grad(loss, self.net.parameters())
            droped_grad = [self.dropout(g) for g in grad]

            adapted_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(droped_grad, self.net.parameters())))

            with torch.no_grad():
                dev, radius = self.net(x_qry[i], fea_qry[i], vars = self.net.parameters()) # orginal loss
                loss_q = loss_func(dev, y_qry[i])
                roc_q = auc_func(dev, y_qry[i])
                losses_q[0] += loss_q
                rocs[0] += roc_q
            
            with torch.no_grad():
                dev, radius = self.net(x_qry[i], fea_qry[i], vars = adapted_weights) # updated loss after an updating
                loss_q = loss_func(dev, y_qry[i])
                roc_q = auc_func(dev, y_qry[i])
                losses_q[1] += loss_q
                rocs[1] += roc_q
            
            for k in range(1, self.update_step):
                dev, radius = self.net(x_spt[i], fea_spt[i], vars = adapted_weights)
                loss = loss_func(dev, y_spt[i])
                losses_s[k] += loss
                grad = torch.autograd.grad(loss, adapted_weights, retain_graph = True)
                droped_grad = [self.dropout(g) for g in grad]

                adapted_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(droped_grad, adapted_weights)))            

                dev_q, radius_q = self.net(x_qry[i], fea_qry[i], vars = adapted_weights)
                loss_q = loss_func(dev_q, y_qry[i])
                roc_q = auc_func(dev_q, y_qry[i])
                losses_q[k+1] += loss_q
                rocs[k+1] += roc_q
        
        loss_q = losses_q[-1] / nb_graphs
        acc_q = rocs[-1] / nb_graphs
        accs1 = np.array(rocs) / nb_graphs

        print('loss', loss_q)

        if torch.isnan(loss_q):
            pass
        else:
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.scheduler.step()
            self.meta_optim.step()

            

        return accs1
    
    def finetune(self, x_spt, fea_spt, y_spt, x_qry, fea_qry, y_qry):
        rocs = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)
        #net.training = False
        x_spt = x_spt[0]
        fea_spt = fea_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        fea_qry = fea_qry[0]
        y_qry = y_qry[0]

        dev, radius = net(x_spt, fea_spt)
        loss = loss_func(dev, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        adapted_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        with torch.no_grad():
            dev, radius = net(x_qry, fea_qry, vars=net.parameters())
            loss_q = loss_func(dev, y_qry)
            roc_q = auc_func(dev, y_qry)
            rocs[0] += roc_q
        
        with torch.no_grad():
            dev, radius = net(x_qry, fea_qry, vars=adapted_weights)
            loss_q = loss_func(dev, y_qry)
            roc_q = auc_func(dev, y_qry)
            rocs[1] += roc_q
        
        for k in range(1, self.update_step_test):
            dev, radius = net(x_spt, fea_spt, adapted_weights)
            loss = loss_func(dev, y_spt)

            grad = torch.autograd.grad(loss, adapted_weights, retain_graph = True)
            adapted_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapted_weights)))

            dev_q, radius_q = net(x_qry, fea_qry, vars=adapted_weights)
            loss_q = loss_func(dev_q, y_qry)
            roc_q = auc_func(dev_q, y_qry)
            rocs[k+1] += roc_q
        
        y_pred, _ = net(x_qry, fea_qry, adapted_weights)
        del net
        accs1 = np.array(rocs)

        return accs1, y_pred, y_qry

            








