"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

#Load the data into an undirected graph G
path_data = "C:\\Users\\gamer\\Downloads\\Lab4\\datasets\\CA-HepTh.txt"
G = nx.read_edgelist(path_data)


print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
############## Task 2

components = np.array(list(nx.connected_components(G)))
print(f"Number of connected components: {len(components)}")

if len(components) > 1: #the graph is not connected if there is more than 1 connected component
    largest_component = max(components, key=len)
    G_largest = G.subgraph(largest_component)
    print(f"Size of the largest connected component: {G_largest.number_of_nodes()} nodes, {G_largest.number_of_edges()} edges")
    print(f"Fraction of nodes in the largest component: {G_largest.number_of_nodes() / G.number_of_nodes():.2f}")
