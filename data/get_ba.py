import tensorflow
from networkx import barabasi_albert_graph
import numpy as np
import json

def nx2CGVAE(ba,N,M):
    graph = []
    for (u, v) in ba.edges.keys():
        graph.append([u, 1, v])
    return { "graph": graph,
             "node_features": [[1] for _ in range(N)],
             "targets": [[M]],
             "smiles": "C",
             }

def BA_in_CGVAEformat(num_train=2400,num_validation=600):

    scopes = [('train',num_train), ('valid',num_validation)]

    for (name, num_graphs) in scopes:
        CGVAE_format_graphs = []

        for k in range(num_train):
            N = np.random.choice([6, 8, 10, 12])
            N = int(N)
            M = np.random.random_integers(1,N-1)
            M = int(M)
            ba = barabasi_albert_graph(N,M)
            CGVAE_format_graphs.append(nx2CGVAE(ba,N,M))

        with open('molecules_%s_%s.json' % (name, 'ba'), 'w') as f:
            json.dump(CGVAE_format_graphs, f)


BA_in_CGVAEformat()