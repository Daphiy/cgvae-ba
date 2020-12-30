import tensorflow
from networkx import barabasi_albert_graph
import numpy as np
import json

def BA_2_CGVAE_format(num_train=2400,num_val=600,N=10,M=8):
    ba_graphs = []
    for k in range(num_train+num_val):
        ba_graphs.append(barabasi_albert_graph(N,M))

    CGVAE_format_graphs_train, CGVAE_format_graphs_val = [], []

    for ba in ba_graphs[:num_train]:
        graph = []
        for (u,v) in ba.edges.keys():
            graph.append([u, 1, v])
        CGVAE_format_graphs_train.append({
            "graph":graph,
            "node_features":[[1] for node in range(N)],
            "targets":[[np.average(ba.degree)]],
            "smiles":"C",
        })

    with open('molecules_%s_%s.json' % ('train', 'ba'), 'w') as f:
        json.dump(CGVAE_format_graphs_train, f)
    
    for ba in ba_graphs[num_train:]:
        graph = []
        for (u, v) in ba.edges.keys():
            graph.append([u, 1, v])
        CGVAE_format_graphs_val.append({
                "graph": graph,
                "node_features": [[1] for node in range(N)],
                "targets": [[np.average(ba.degree)]],
                "smiles": "C",
        })
    with open('molecules_%s_%s.json' % ('valid', 'ba'), 'w') as f:
        json.dump(CGVAE_format_graphs_val, f)


BA_2_CGVAE_format()