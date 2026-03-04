import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool, BatchNorm 

class Fakenews_GNN(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels=64, 
                 out_channels=2, 
                 num_layers=2, 
                 conv_type="GCN", 
                 dropout=0.0, 
                 use_batchnorm=False, 
                 pooling="mean"):
        super(Fakenews_GNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # Choose convolution layer type
        conv_dict = {
            "GCN": GCNConv,
            "GraphSAGE": SAGEConv,
            "GAT": GATConv
        }
        Conv = conv_dict[conv_type]
        
        # Build GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() if use_batchnorm else None
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(Conv(in_ch, hidden_channels))
            if use_batchnorm:
                self.bns.append(BatchNorm(hidden_channels))
        
        # Pooling
        pooling_dict = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "add": global_add_pool
        }
        self.pool = pooling_dict[pooling]
        
        # Final classifier
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        x = self.fc(x)
        return x
