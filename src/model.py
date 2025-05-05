import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE  # or SAGEConv + custom
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import os
import networkx as nx
import numpy as np

sbert = SentenceTransformer('all-MiniLM-L6-v2')

def load_graph(graph_path):
    Gnx = nx.read_graphml(graph_path)

    mapping = {nid: i for i, nid in enumerate(Gnx.nodes())}
    edges = np.array([[mapping[u], mapping[v]] for u, v in Gnx.edges()]).T

    # x = torch.zeros((len(mapping), 384), dtype=torch.float)
    # data = Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long))
    # return data, mapping
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, mapping

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim=384, hidden=128, num_layers=2):
        super().__init__()
        # Initialize GraphSAGE with specified number of layers
        self.sage = GraphSAGE(in_channels=in_dim, hidden_channels=hidden, num_layers=num_layers)
        self.lin = torch.nn.Linear(hidden*2, 1)

    def forward(self, x, edge_index, pos_edge_label_index, neg_edge_label_index):
        h = self.sage(x, edge_index)
        # positive
        pos_h = torch.cat([h[pos_edge_label_index[0]], h[pos_edge_label_index[1]]], dim=1)
        neg_h = torch.cat([h[neg_edge_label_index[0]], h[neg_edge_label_index[1]]], dim=1)
        pos_score = self.lin(pos_h).view(-1)
        neg_score = self.lin(neg_h).view(-1)
        return pos_score, neg_score
