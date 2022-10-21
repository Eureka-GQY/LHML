import networkx as nx
import scipy.io as sio
import metis
import numpy as np
import dgl

def partition(f, p):
    data = sio.loadmat(f)

    adj = data['Network']
    attr = data['Attributes']
    label = data['Label']


    G_nx = nx.Graph(adj)
    (edgecuts, parts) = metis.part_graph(G_nx, p)
    nodes = np.array(parts)
    
    graphs = []
    G = dgl.from_scipy(adj)

    for i in range(p):
        idx = np.nonzero(nodes == i)[0]
        sub = dgl.node_subgraph(G, idx, store_ids=False)
        sub = sub.adj(scipy_fmt='coo')

        fea = attr[idx]
        grt = label[idx]
        graph_dict = {'Network':sub, 'Attributes':fea, 'Label':grt}
        graphs.append(graph_dict)


    return graphs
