import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool


# GCN based model
class PPI_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(PPI_GCN, self).__init__()

        print('PPI_GCN Loading ...')
        self.n_output = n_output

        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro1_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro1_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro1_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro2_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro2_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro2_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_pro1, data_pro2):
        pro1_x, pro1_edge_index, pro1_batch = data_pro1.x, data_pro1.edge_index, data_pro1.batch
        pro2_x, pro2_edge_index, pro2_batch = data_pro2.x, data_pro2.edge_index, data_pro2.batch

        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)

        x = self.pro1_conv2(x, pro1_edge_index)
        x = self.relu(x)

        x = self.pro1_conv3(x, pro1_edge_index)
        x = gep(x, pro1_batch)  # global pooling

        # flatten
        x = self.relu(self.pro1_fc_g1(x))
        x = self.dropout(x)
        x = self.pro1_fc_g2(x)
        x = self.dropout(x)

        xt = self.pro2_conv1(pro2_x, pro2_edge_index)

        xt = self.relu(xt)

        xt = self.pro2_conv2(xt, pro2_edge_index)

        xt = self.relu(xt)

        xt = self.pro2_conv3(xt, pro2_edge_index)
        xt = self.relu(xt)

        xt = gep(xt, pro2_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro2_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro2_fc_g2(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        return out
