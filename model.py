# import torch
# import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout
from torch_geometric.nn import GCNConv, SortAggregation, SAGEConv, GATConv, SimpleConv
from torch_geometric.utils import remove_self_loops
import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)


class Model(nn.Module):
    def __init__(self, num_features, num_classes, hidden_layers, depth, option=0):
        super(Model, self).__init__()
        core = [GCNConv, SAGEConv, GATConv]

        self.convInitial = core[option](num_features, hidden_layers)
        self.convs = []
        for _ in range(depth - 2):
            self.convs += [core[option](hidden_layers, hidden_layers)]             
        
        self.convFinal = core[option](hidden_layers, 1)
        self.kernel = (depth  - 1) * hidden_layers + 1
        self.sort_pool = SortAggregation(k=30)
        self.conv1 = Conv1d(1, 512, self.kernel, self.kernel)
        self.conv2 = Conv1d(512, 256, kernel_size=5, stride=1, padding=48)       
        self.pool = MaxPool1d(2)
        self.classifier_1 = Linear(256*53, 128)
        self.drop_out = Dropout(0.2)        
        self.classifier_2 = Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        # print(self.kernel)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x.size())
        edge_index, _ = remove_self_loops(edge_index)
        lst = []
        x = torch.tanh(self.convInitial(x, edge_index))
        lst += [x]
        if self.convs != []:
            for conv in self.convs:
                x = torch.tanh(conv(x, edge_index))
                lst += [x]
        x = torch.tanh(self.convFinal(x, edge_index))
        lst += [x]
        # x_1 = torch.tanh(self.conv1(x, edge_index))
        # # print( 'GCN layers 1:',x_1.shape)
        # x_2 = torch.tanh(self.conv2(x_1, edge_index))
        # # print( 'GCN layers 2:',x_2.shape)
        # x_3 = torch.tanh(self.conv3(x_2, edge_index))
        # # print( 'GCN layers 3:',x_3.shape)
        # x_4 = torch.tanh(self.conv4(x_3, edge_index))
        # # print( 'GCN layers 4:',x_4.shape)
        x = torch.cat(lst, dim=-1)
        # print( 'GCN layers all:',x.shape)
        x = self.sort_pool(x, batch)        
        x = x.view(x.size(0), 1, x.size(-1))
        # print( 'sort pooling layers 1:',x.shape)
        x = self.relu(self.conv1(x))
        # print('relu1: ', x.shape)
        x = self.pool(x)
        # print('pool1: ', x.shape)
        x = self.relu(self.conv2(x))
        # print('relu2: ', x.shape)   
        x = self.pool(x)
        # print('pool2: ', x.shape)
        # x = self.relu(self.conv3(x))
        # print('relu3: ', x.shape)   
        # x = self.pool(x)  
        x = x.view(x.size(0), -1) 
        # print(x.shape)       
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        classes = F.log_softmax(self.classifier_2(out), dim=-1)        
        # print(classes)
        return classes