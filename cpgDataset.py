import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import json
import pickle
class cpgDataset(Dataset):
    def __init__(self, root, test=False, transform=None, pre_transform=None, pre_filter=None):               
        self.test = test        
        super(cpgDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return ['train.zip']
        return ['CWE89_main.txt']
        # pass

    @property
    def processed_file_names(self):
        self.data = {}             

        for i in range(0, 5400, 300):
            filename = f"./data/CPG/train_set_{i}.pkl"
            with open(filename, "rb") as f:
                chunk = pickle.load(f)            
            self.data.update(chunk)

        # for i in range(0, 1900, 300):
        #     filename = f"./data/CPG/CWE_89/train_set_{i}.pkl"
        #     with open(filename, "rb") as f:
        #         chunk = pickle.load(f)            
        #     self.data.update(chunk)
             

        if self.test:
            return [f'data_test_{i}.pt' for i in self.data]
        else:
            return [f'data_{i}.pt' for i in self.data]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):        
        self.data = {}   
        # for i in range(0, 1900, 300):
        #     filename = f"./data/CPG/CWE_89/train_set_{i}.pkl"
        #     with open(filename, "rb") as f:
        #         chunk = pickle.load(f)            
        #     self.data.update(chunk)   
        # for i in range(0, 2130, 300):
        #     filename = f"./data/CPG/CWE_89/train_set_{i}.pkl"
        #     with open(filename, "rb") as f:
        #         chunk = pickle.load(f)            
        #     self.data.update(chunk)
        for i in range(0, 5400, 300):
            filename = f"./data/CPG/train_set_{i}.pkl"
            with open(filename, "rb") as f:
                chunk = pickle.load(f)            
            self.data.update(chunk)
        for index in self.data:            
            # node_feature = torch.tensor(self.data[index]['nodes'], dtype=torch.float).reshape([-1, 1])
            
            node_feature = torch.tensor(self.data[index]['nodes'], dtype=torch.float)
            # edge_feature = torch.tensor(self.data[index]['edge_feature'], dtype=torch.float)
            edge_index = torch.tensor(self.data[index]['edges'], dtype=torch.float)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            label = torch.tensor([self.data[index]['label']])
            data = Data(x=node_feature, 
                        edge_index=edge_index,
                        # edge_attr=edge_feature,
                        y=label
                        ) 
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))
    def len(self):
        return len(self.data)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
    def num_node_features(self):
        return len(self.get(1).x[0])
    # def num_edge_features(self):
    #     return len(self.get(1).edge_attr[0])
    def num_classes(self):
        return 2

       
    