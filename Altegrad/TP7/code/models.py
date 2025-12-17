"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2*n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        
        ############## Task 1
        
        h = self.fc(x)        
        indices = adj.coalesce().indices()

        # Compute attention coefficients
        h_i = h[indices[0, :], :]
        h_j = h[indices[1, :], :]
        h_cat = torch.cat([h_i, h_j], dim=1)
        e = self.leakyrelu(self.a(h_cat))
        
        # Normalize attention coefficients using softmax
        e = torch.exp(e.squeeze())
        unique = torch.unique(indices[0,:])
        t = torch.zeros(unique.size(0), device=x.device)
        e_sum = t.scatter_add(0, indices[0,:], e) 
        e_norm = torch.gather(e_sum, 0, indices[0,:])
        alpha = torch.div(e, e_norm)
        
        # Create attention-weighted adjacency matrix
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)
        
        # Compute output by applying attention-weighted aggregation
        out = torch.sparse.mm(adj_att, h)

        return out, alpha


class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, x, adj):
        
        ############## Tasks 2 and 4
    
        x, _ = self.mp1(x, adj)
        x = self.relu(x) 
        x = self.dropout(x)

        x, alpha_2 = self.mp2(x, adj)
        x = self.relu(x)
        x = self.dropout(x)

        x= self.fc(x)

        return F.log_softmax(x, dim=1), alpha_2
