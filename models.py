import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim // 4, output_dim))
        self.relu = torch.nn.GELU()
 
 
    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.MLP(x)
        x = x.mean(dim=1)
        #print(x.shape)
        return x
