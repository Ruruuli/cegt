import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import sys

class Ortho_Trans(torch.nn.Module):
    def __init__(self, T=5, norm_groups=norm_groups, *args, **kwargs):
        super(Ortho_Trans, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-4

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)

        S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        return W.view_as(weight)


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_iterations):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.u = nn.Parameter(torch.randn(hidden_dim))
        nn.init.orthogonal_(self.W)

    def forward(self, x):
        u_temp = torch.tanh(torch.matmul(x, self.W))
        u_temp = torch.matmul(u_temp, self.u)
        alpha = F.softmax(u_temp, dim=1).unsqueeze(-1)
        r = (alpha * x).sum(dim=1, keepdim=True)

        for i in range(self.num_iterations):
            beta = F.softmax(torch.matmul(x, r.transpose(1, 2)).squeeze(-1), dim=1)
            beta = beta.unsqueeze(-1)
            r = (beta * x).sum(dim=1, keepdim=True)

        return r


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads,num_iterations=3)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(x))
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(2)])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Model(nn.Module):
    def __init__(self, input_dim,output_dim, hidden_dim,d_model, nhead,dropout, num_layers):
        super(Model, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.nhead = nhead
        self.conv_layers = torch.nn.ModuleList()
        self.dropout = dropout
        self.encoder = Encoder(d_model, nhead, hidden_dim, num_layers, dropout)
        self.gama = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        for _ in range(num_layers):
            if _ == 0:
                self.conv_layers.append(GraphConvolution(input_dim, hidden_dim))
            else:
                self.conv_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        self.conv_layers.append(GraphConvolution(hidden_dim,d_model))
        self.weight_normalization = Ortho_Trans(T=5,norm_groups=4)
        self.conv_layers[0].weight.data = self.weight_normalization(self.conv_layers[0].weight)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4 ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim // 4, d_model))
        self.relu = torch.nn.GELU()
        self.sigmod = torch.nn.Sigmoid()
        self.embedding = nn.Linear(input_dim, d_model)
        self.fc = nn.Linear(d_model,output_dim)

    def forward(self, x,edge_index,mask=None):
        for conv in self.conv_layers:
            x = self.relu(conv(x, edge_index))
        x = F.dropout(x,self.dropout, training=self.training)
        x = self.MLP(x)
        x = self.encoder(x, mask)
        x = self.fc(x)
        # x = torch.max(x, dim=1)[0].squeeze()
        x = x.mean(dim=1)
        return x

