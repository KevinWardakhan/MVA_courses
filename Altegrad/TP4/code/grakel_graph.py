
####### RENAMED THE FILE BECAUSE CONFLICT WHEN IMPORTING GRAKEL PACKAGE IF FILE NAME IS grakel.py #######
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = []
    for doc in docs:
        G = nx.Graph()
        
        # Process sliding windows
        for i in range(len(doc)):
            current_word = doc[i]
            if current_word not in vocab:
                continue
            
            current_idx = vocab[current_word]
            G.add_node(current_idx, label=current_word)
            
            # Look ahead within window
            for j in range(i + 1, min(i + window_size + 1, len(doc))):
                neighbor_word = doc[j]
                if neighbor_word in vocab:
                    neighbor_idx = vocab[neighbor_word]
                    G.add_edge(current_idx, neighbor_idx)
                    G.nodes[neighbor_idx]['label'] = neighbor_word
                    
        graphs.append(G)
    return graphs

# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

from grakel import Graph

def convert_nx_to_grakel(nx_graphs):
    grakel_graphs = []
    for G_nx in nx_graphs:
        edges = list(G_nx.edges())
        node_labels = {}
        for node in G_nx.nodes():
            if 'label' in G_nx.nodes[node]:
                node_labels[node] = G_nx.nodes[node]['label']
            else:
                node_labels[node] = str(node)
        
        g = Graph(edges, node_labels=node_labels)
        grakel_graphs.append(g)
    
    return grakel_graphs

# # Transform networkx graphs to grakel representations
# G_train = graph_from_networkx(G_train_nx, node_labels_tag='label') #generate parse input is empty, i have thus create a function to convert manually
# G_test = graph_from_networkx(G_test_nx, node_labels_tag='label')


G_train = convert_nx_to_grakel(G_train_nx)
G_test = convert_nx_to_grakel(G_test_nx)


# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_jobs=-1, normalize=False, n_iter=1, base_graph_kernel=VertexHistogram)

# Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

#Task 13
import time

# Train an SVM classifier and make predictions

print("Weisfeiler-Lehman Kernel computation...")
start_time = time.time()
svm_classifier = SVC(kernel='precomputed')
svm_classifier.fit(K_train, y_train)
y_pred = svm_classifier.predict(K_test)
end_time = time.time()
print(f"Weisfeiler-Lehman Kernel SVM training and prediction time: {end_time - start_time:.2f} seconds")

# Evaluate the predictions
print(f"Accuracy : {accuracy_score(y_pred, y_test)*100:.2f}%")


#Task 14

from grakel.kernels import PyramidMatch, ShortestPath, GraphletSampling, RandomWalk, Propagation, OddSth

# Pyramid Match Kernel is too heavy and raises memory error on my machine, so I comment it out

# print("Pyramid Match Kernel computation...")
#start_time = time.time()
# pm = PyramidMatch(n_jobs=-1, normalize=False)
# K_train_pm = pm.fit_transform(G_train)
# K_test_pm = pm.transform(G_test)

# svm_pm = SVC(kernel='precomputed')
# svm_pm.fit(K_train_pm, y_train)
# y_pred_pm = svm_pm.predict(K_test_pm)
# end_time = time.time()
# print(f"Pyramid Match Kernel SVM training and prediction time: {end_time - start_time:.2f} seconds")

# print(f"Accuracy for Pyramid Match Kernel: {accuracy_score(y_test, y_pred_pm)*100:.2f}%")


print("Shortest Path Kernel computation...")
sp = ShortestPath(n_jobs=-1, normalize=False)
start_time = time.time()
K_train_sp = sp.fit_transform(G_train)
K_test_sp = sp.transform(G_test)

svm_sp = SVC(kernel='precomputed')
svm_sp.fit(K_train_sp, y_train)
y_pred_sp = svm_sp.predict(K_test_sp)
end_time = time.time()
print(f"Shortest Path Kernel SVM training and prediction time: {end_time - start_time:.2f} seconds")
print(f"Accuracy for Shortest Path Kernel: {accuracy_score(y_test, y_pred_sp)*100:.2f}%")


print("Graphlet Kernel computation...")
gp = GraphletSampling(n_jobs=-1, normalize=False)
start_time = time.time()
K_train_gp = gp.fit_transform(G_train)
K_test_gp = gp.transform(G_test)
svm_gp = SVC(kernel='precomputed')
svm_gp.fit(K_train_gp, y_train)
y_pred_gp = svm_gp.predict(K_test_gp)
end_time = time.time()
print(f"Graphlet Sampling Kernel SVM training and prediction time: {end_time - start_time:.2f} seconds")
print(f"Accuracy for Graphlet Sampling Kernel: {accuracy_score(y_test, y_pred_gp)*100:.2f}%")

print("Random Walk Kernel computation...")
rw = RandomWalk(n_jobs=-1, normalize=False)
start_time = time.time()
K_train_rw = rw.fit_transform(G_train)
K_test_rw = rw.transform(G_test)
svm_rw = SVC(kernel='precomputed')
svm_rw.fit(K_train_rw, y_train)
y_pred_rw = svm_rw.predict(K_test_rw)
end_time = time.time()
print(f"Random Walk Kernel SVM training and prediction time: {end_time - start_time:.2f} seconds")
print(f"Accuracy for Random Walk Kernel: {accuracy_score(y_test, y_pred_rw)*100:.2f}%")


print("Propagation Kernel computation...")
prop = Propagation(n_jobs=-1, normalize=False)
start_time = time.time()
K_train_prop = prop.fit_transform(G_train)
K_test_prop = prop.transform(G_test)
svm_prop = SVC(kernel='precomputed')
svm_prop.fit(K_train_prop, y_train)
y_pred_prop = svm_prop.predict(K_test_prop)
end_time = time.time()
print(f"Propagation Kernel SVM training and prediction time: {end_time - start_time:.2f} seconds")
print(f"Accuracy for Propagation Kernel: {accuracy_score(y_test, y_pred_prop)*100:.2f}%")


print("OddSth Kernel computation...")
osth = OddSth(n_jobs=-1, normalize=False)
start_time = time.time()
K_train_osth = osth.fit_transform(G_train)
K_test_osth = osth.transform(G_test)
svm_osth = SVC(kernel='precomputed')
svm_osth.fit(K_train_osth, y_train)
y_pred_osth = svm_osth.predict(K_test_osth)
end_time = time.time()
print(f"OddSth Kernel SVM training and prediction time: {end_time - start_time:.2f} seconds")
print(f"Accuracy for OddSth Kernel: {accuracy_score(y_test, y_pred_osth)*100:.2f}%")


#The best model is the Weisfeiler-Lehman Kernel with an accuracy of 96.4%, lower than the one obtained with
# the sp kernel (96.8%), but the computation time is much lower (around 20 seconds vs more than 2 minutes for the sp kernel).