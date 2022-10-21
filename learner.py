import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl
import dgl.function as fn
import math
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair

class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = True
        self._activation = activation
    
    def forward(self, graph, feat, weight, bias=None):
        graph = graph.local_var()
        #aggregate_fn = fn.copy_src(src='h', out = 'm')
        if self._norm:
            norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm
        
        if self._in_feats > self._out_feats:
            feat = torch.matmul(feat, weight)
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out = 'm'), fn.sum(msg = 'm', out='h'))
            rst = graph.ndata['h']
            rst = torch.matmul(rst, weight)
        
        rst = rst * norm
        if bias is not None:
            rst = rst + bias

        if self._activation is not None:
            rst = self._activation(rst)
        
        return rst

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self.bias = True
        self.padding_mode = 'zeros'

    def forward(self, input, weight, bias):
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv1d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self.bias = True
        self.padding_mode = 'zeros'

    def forward(self, input, weight, bias):
        out = F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out

def hypersphere(feat, ref_radius):
        o = torch.mean(feat, dim=0) # data centre
        radius = F.pairwise_distance(feat, o, p=2) 
        dis = torch.sub(radius, ref_radius) # distance between the real radius and reference radius
        dis = torch.clamp(dis, min=1e-4, max=(1-(1e-4))) #smoothing

        return dis, ref_radius


class Classifier(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Classifier, self).__init__()
        self.config = [ ('gcn', [in_dim, hid_dim]),
                        ('gcn', [hid_dim, hid_dim]),
                        #('conv2d', [1, 1, (3,3), 1, 0]),
                        ('conv1d', [1, 2, 3, 1, 0]),
                        ('hypersphere', [])]

        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.graph_conv = []
        self.conv2d_list = []
        self.conv1d_list = []
        self.training = True

        for name, param in self.config:
            if name == 'gcn':
                w = nn.Parameter(torch.Tensor(*param))
                init.xavier_normal_(w)
                self.vars.append(w)
                self.graph_conv.append(GraphConv(*param, activation=F.relu))

            elif name == 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            
            elif name == 'conv2d':
                w = nn.Parameter(torch.empty(param[1], param[0], *param[2]))
                init.kaiming_normal_(w, a=math.sqrt(5))
                self.vars.append(w)

                b = nn.Parameter(torch.empty(param[1]))
                fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(b, -bound, bound)

                self.vars.append(b)
                self.conv2d_list.append(Conv2d(param[0], param[1], param[2], param[3], param[4]))
            
            elif name == 'conv1d':
                w = nn.Parameter(torch.empty(param[1], param[0], param[2]))
                init.kaiming_normal_(w, a=math.sqrt(5))
                self.vars.append(w)

                b = nn.Parameter(torch.empty(param[1]))
                fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(b, -bound, bound)

                self.vars.append(b)
                self.conv1d_list.append(Conv1d(param[0], param[1], param[2], param[3], param[4]))

            elif name == 'hypersphere':
                ref_radius = nn.Parameter(torch.tensor([0.], requires_grad=True)) # the parameter that controls the decision boundary
                self.vars.append(ref_radius)      
    
    def forward(self, g, feat, vars = None):
        g = dgl.add_self_loop(g)
        if vars is None:
            vars = self.vars
        
        idx = 0
        idx_gcn = 0
        idx_conv2d = 0
        idx_conv1d = 0
        idx_bn = 0

        h = feat.float()
        for name, param in self.config:
            if name == 'gcn':
                w = vars[idx]
                conv = self.graph_conv[idx_gcn]
                h = conv(g,h,w)
                h = F.dropout(h, p=0.5, training=self.training)
                g.ndata['h'] = h

                idx += 1
                idx_gcn += 1

            elif name == 'conv2d':
                w, b = vars[idx], vars[idx+1]
                conv = self.conv2d_list[idx_conv2d]
                num_nodes = h.shape[0]
                h = h.view(-1,1,8,8)
                h = conv(h, w, b)
                h = h.view(num_nodes,-1)
                g.ndata['h'] = h

                idx += 2
                idx_conv2d += 1

            elif name == 'conv1d':
                w, b = vars[idx], vars[idx+1]
                conv = self.conv1d_list[idx_conv1d]
                num_nodes = h.shape[0]

                h = h.view(num_nodes,1,-1)
                h = conv(h, w, b)
                h = h.view(num_nodes,-1)

                idx += 2
                idx_conv1d += 1    

            elif name == 'bn':
                w, b = vars[idx], vars[idx+1]
                running_mean, running_var = self.vars_bn[idx_bn], self.vars_bn[idx_bn+1]
                h = F.batch_norm(h, running_mean, running_var, weight=w, bias=b, training=True)
                h = F.relu(h)
                g.ndata['h'] = h

                idx += 2
                idx_bn += 2
            
            elif name == 'hypersphere':
                h,r = hypersphere(h, vars[idx])
                return h,r

        #return h
    
    def zero_grad(self, vars = None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
    
    def parameters(self):
        return self.vars





        