"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k, d=None):
    #if d is not provided, set d=k
    if d==None:
        d=k
    m = G.number_of_nodes()
    #1 Compute the ajacency matrix A of G
    A = nx.adjacency_matrix(G)

    #2 Compute the Laplacian matrix L = I - D^(-1)A
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = np.diag(1.0 / degrees)
    L_rw = np.eye(m) - D_inv @ A.toarray()

    #3 Compute the first d eigenvectors of L
    vals, vecs = np.linalg.eigh(L_rw)
    order = np.argsort(vals)
    U = vecs[:, order[:d]]  # m x d matrix with eigenvectors as columns

    #4 Run k-means on rows of U
    km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(U)
    nodes = list(G.nodes())
    clustering = {node: int(cluster) for node, cluster in zip(nodes, labels)}

    return clustering

############## Task 4
# We need to reload the giant connected component of the CA-HepTh dataset
path_data = "C:\\Users\\gamer\\Downloads\\Lab4\\datasets\\CA-HepTh.txt"
G = nx.read_edgelist(path_data)
components = np.array(list(nx.connected_components(G)))
largest_component = max(components, key=len)
G_largest = G.subgraph(largest_component)

# Apply spectral clustering on G_largest
k=50 #number of clusters
print("Performing spectral clustering...")
clustering = spectral_clustering(G_largest, k)
print("Spectral clustering completed !")
print(f"Clustering result (first 10 nodes): {dict(list(clustering.items())[:10])}")


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    m = G.number_of_edges()
    degrees = dict(G.degree()) #degree of each node
    n_c = max(clustering.values()) + 1 #number of clusters 
    clusters_nodes = [[node for (node, cluster) in clustering.items() if cluster == c] for c in range(n_c)]
    dc_list = np.array([np.sum([degrees[node] for node in clusters_nodes[c]]) for c in range(n_c)]) #list of dc values
    edges = G.edges()
    lc_list = np.array([len([edge for edge in edges if c == clustering[edge[0]] and c == clustering[edge[1]]]) for c in range(n_c)]) #list of lc values
    
    modularity = np.sum((lc_list/m) - (dc_list/(2*m))**2)
    return modularity



############## Task 6

mod_g_larget = modularity(G_largest, clustering)
print(f"Modularity of the clustering on the largest connected component: {mod_g_larget:.4f}")

random_clustering = {node: np.random.randint(0, k) for node in G_largest.nodes()}

m_random = modularity(G_largest, random_clustering)
print(f"Modularity of the random clustering on the largest connected component: {m_random:.4f}")



