"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():
    dataset = TUDataset(root='datasets', name='MUTAG')
    Gs = [to_networkx(data, to_undirected=True) for data in dataset]
    y = [data.y.item() for data in dataset]
    return Gs, y


Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(Gs_train), 4))
    
    random_nodes = []
    for G in Gs_train: #for each graph, sample n_samples sets of 3 nodes
        random_nodes_G = []
        nodes_list = list(G.nodes())
        if len(nodes_list) <3:
            replace = True
        else:
            replace = False
        for _ in range(n_samples):
            random_nodes_G.append(np.random.choice(nodes_list, size=3, replace=replace))
        random_nodes.append(random_nodes_G)

    for i in range(4):
        current_graphlet = graphlets[i]
        for j,G in enumerate(Gs_train):
            count = 0
            for nodes in random_nodes[j]:
                subgraph = G.subgraph(nodes)
                if nx.is_isomorphic(subgraph, current_graphlet):
                    count += 1
            phi_train[j,i] = count

    phi_test = np.zeros((len(Gs_test), 4))
    
    random_nodes_test = []
    for G in Gs_test:
        random_nodes_G = []
        nodes_list = list(G.nodes())
        if len(nodes_list) < 3:
            replace = True
        else:
            replace = False
        for _ in range(n_samples):
            random_nodes_G.append(np.random.choice(nodes_list, size=3, replace=replace))
        random_nodes_test.append(random_nodes_G)
    
    for i in range(4):
        current_graphlet = graphlets[i]
        for j,G in enumerate(Gs_test):
            count = 0
            for nodes in random_nodes[j]:
                subgraph = G.subgraph(nodes)
                if nx.is_isomorphic(subgraph, current_graphlet):
                    count += 1
            phi_test[j,i] = count

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test

print("Shortest Path Kernel computation...")
K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9

print("Graphlet Kernel computation...")
K_train_gk, K_test_gk = graphlet_kernel(G_train, G_test) #nsamples = 200 by default



############## Task 10

# Train and evaluate SVM with longuest path kernel
classifier_lp = SVC(kernel='precomputed')
classifier_lp.fit(K_train_sp, y_train)
y_pred_lp = classifier_lp.predict(K_test_sp)
accuracy_lp = accuracy_score(y_test, y_pred_lp)
print(f"Shortest Path Kernel SVM Accuracy: {accuracy_lp*100:.2f}%")

# Train and evaluate SVM with graphlet kernel
classifier_gk = SVC(kernel='precomputed')
classifier_gk.fit(K_train_gk, y_train)
y_pred_gk = classifier_gk.predict(K_test_gk)
accuracy_gk = accuracy_score(y_test, y_pred_gk)
print(f"Graphlet Kernel SVM Accuracy: {accuracy_gk*100:.2f}%")

print("Shortest path kernel has much better accuracy than graphlet kernel on MUTAG dataset.")