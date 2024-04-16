import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import visdom
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from cpgDataset import cpgDataset
from model import Model
# from utils import Indegree
from set_determ import set_determ
from sklearn.model_selection import KFold

def get_args():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_type', default='CPG', type=str,
                        choices=['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB', 'CPG'],
                        help='dataset type')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
    parser.add_argument('--seed', default=324, type=int, help='random seed')
    return parser.parse_args()

def train(dataloader, model, loss_fn, optimizer, device):
    """Training in one epoch. Return loss and accuracy*100."""        
    model.train()
    num_batches = len(dataloader)         
    num_samples = len(dataloader.dataset)    
    running_loss, correct = 0, 0
    
    for sample in dataloader:        
        data, y = sample.to(device), sample.y.to(device)    
        # print('nodes number', len(data.x))
        # print('feature length', len(data.x[0]))
        # print('feature', data.x)
        # print('feature size', data.x.size)
        # print('total', len(data.x) * len(data.x[0])) 
        # print(data)         
        pred = model(data)
       
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        correct += (pred.argmax(dim=1) == y).sum().item()

    return running_loss/num_batches, correct/num_samples*100

count = 0
def test(dataloader, model, loss_fn, device):
    """Test in one epoch. Return loss and accuracy*100."""

    model.eval()
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    running_loss, correct = 0, 0
    # confusion matrix
    y_true = []
    y_pred = []
  
    with torch.no_grad():
        for sample in dataloader:
            data, y = sample.to(device), sample.y.to(device)
            global count
            # if count == 0:
            #     print(data , '\n')
            #     print(y)
            #     count+=1
            pred = model(data)
            loss = loss_fn(pred, y)

            running_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()

            y_true.extend(y.tolist())
            _, predicted = torch.max(pred.data, 1)
            y_pred.extend(predicted.tolist())

    cm = confusion_matrix(y_true, y_pred)     
    
    return running_loss/num_batches, correct/num_samples*100, cm


if __name__ == '__main__':
    
    # ─── Initialization ───────────────────────────────────────────────────

    opt = get_args()
    set_determ(opt.seed)
    device = (
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    vis = visdom.Visdom(env=opt.data_type)  # To plot loss and accuracy
    # ============================= Dataset ================================================
    # data_set = TUDataset(
    #     f'data/{opt.data_type}',
    #     opt.data_type,
    #     pre_transform=Indegree(),
    #     use_node_attr=True,
    # )
    # print(f'{data_set.num_features=}, {data_set.num_classes=}')
    # print(data_set.get(2))
    # print(data_set.get(2).y)
    # # ─── 10-fold Cross Validation ─────────────────────────────────────────

    
    # data_set = cpgDataset(root="data/")
    
    # hidden_layers = [32, 64, 128]
    # depth = [2, 3, 4]
    # option = 1
    # core = ['GCNConv', 'SAGEConv', 'GATConv']
    # for i in hidden_layers:
    #     for j in depth:
    #         for t in range(3):               
    #             over_results = {'train_accuracy': [], 'test_accuracy': []}
    #             over_matrix = {'epoch': 0, 'matrix': [], 'max_acc': 0, 'fold': 0}
    #             # Define the K-fold Cross Validator
    #             print('Core layer: ', core[t], '--', 'hidden layer size: ', i, '--', 'depth: ', j)
    #             kfold = KFold(n_splits=5, shuffle=True)
    #             for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):
    #                 print(f'FOLD {fold + 1}')
    #                 train_set, test_set = data_set[train_ids], data_set[test_ids]   
                    
    #                 print('--------------------------------')
                    
    #                 model = Model(data_set.num_node_features(), data_set.num_classes(), i, j, t).to(device)
    #                 # model = GraphModel(3, data_set.num_node_features(), 32, 128, data_set.num_classes()).to(device)
    #                 loss_criterion = nn.NLLLoss()  # Set loss criterion to negative log likelihood loss
    #                 optimizer = Adam(model.parameters()) # Create Adam optimizer for model parameters
    #                 train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    #                 test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False)         

    #             #     # ─── Training Loop ────────────────────────────────────────────
                    
    #                 fold_results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
    #                 for epoch in range(1, opt.num_epochs+1):
    #                     train_loss, train_acc = train(train_loader, model, loss_criterion, optimizer, device)
    #                     test_loss, test_acc, cm = test(test_loader, model, loss_criterion, device)
    #                     if test_acc > over_matrix['max_acc']:
    #                         over_matrix['epoch'] = epoch
    #                         over_matrix['matrix'] = cm
    #                         over_matrix['max_acc'] = test_acc
    #                     fold_results['train_loss'].append(train_loss)
    #                     fold_results['train_accuracy'].append(train_acc)
    #                     fold_results['test_loss'].append(test_loss)
    #                     fold_results['test_accuracy'].append(test_acc)
    #                     vis.line(torch.tensor([train_loss]), torch.tensor([epoch]), win='Train Loss', update='append', name=f'Fold_{fold + 1}', opts={'title':'Train Loss', 'xlabel':'Epoch', 'ylabel':'NLL Loss'})
    #                     vis.line(torch.tensor([train_acc]), torch.tensor([epoch]), win='Train Accuracy', update='append', name=f'Fold_{fold + 1}', opts={'title':'Train Accuracy', 'xlabel':'Epoch', 'ylabel':'%'})
    #                     vis.line(torch.tensor([test_loss]), torch.tensor([epoch]), win='Test Loss', update='append', name=f'Fold_{fold + 1}', opts={'title':'Test Loss', 'xlabel':'Epoch', 'ylabel':'NLL Loss'})
    #                     vis.line(torch.tensor([test_acc]), torch.tensor([epoch]), win='Test Accuracy', update='append', name=f'Fold_{fold + 1}', opts={'title':'Test Accuracy', 'xlabel':'Epoch', 'ylabel':'%'})

    #                 # ─── Save To Files ────────────────────────────────────────────

    #                 torch.save(model.state_dict(), f'epochs/{opt.data_type}_{fold + 1}_{core[t]}_{i}_{j}.pth')
    #                 pd.DataFrame(data=fold_results, index=range(1, opt.num_epochs + 1)).to_csv(
    #                     f'statistics/{opt.data_type}_results_{fold + 1}_{core[t]}_{i}_{j}.csv', index_label='epoch')
                    
    #                 # ─── Save Overall Resultsrain_iter = tqdm(range(1, 11), desc='Training Model......')
    #             # print(fold_results)
    #             # train_iter = tqdm(range(1, 11), desc='Training Model......')
    #             # for fold_number in train_iter:

    #                 # ─── Model ─────────────────────────────────────

    #                 over_results['train_accuracy'].append(max(fold_results['train_accuracy']))
    #                 # print(over_results['test_accuracy'])
    #                 over_results['test_accuracy'].append(max(fold_results['test_accuracy']))

    #                 # ─── Print Progress Bar ───────────────────────────────────────

    #                 print(f'[{fold + 1}] Train Acc: {max(fold_results["train_accuracy"]):.2f}% Test Acc: {max(fold_results["test_accuracy"]):.2f}%')

    #             # ─── Save And Print Overall Result ────────────────────────────────────

    #             pd.DataFrame(data=over_results,index=range(1, 6)).to_csv(
    #                 f'statistics/{opt.data_type}_{core[t]}_{i}_{j}_results_overall.csv', index_label='fold')
    #             print('Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' %
    #                 (np.array(over_results['train_accuracy']).mean(), np.array(over_results['train_accuracy']).std(),
    #                 np.array(over_results['test_accuracy']).mean(), np.array(over_results['test_accuracy']).std()))
    #             cm = over_matrix['matrix']
    #             cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
    data_set = cpgDataset(root="data/", test=True)
    
    hidden_layers = [32, 64, 128]
    depth = [2, 3, 4]
    option = 1
    core = ['GCNConv', 'SAGEConv', 'GATConv']
    model_lst = []
    for i in hidden_layers:
        for j in depth:
            for t in range(3): 
                model = Model(data_set.num_node_features(), data_set.num_classes(), i, j, t).to(device)
                des = f'Core layer: {core[t]} -- hidden layer size: {i} -- depth: {j}'
                optimizer = Adam(model.parameters())
                model_lst += [[model, optimizer, des]]
    print(len(model_lst))
    kfold = KFold(n_splits=5, shuffle=True)
    kfold_data = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data_set)):
        train_set, test_set = data_set[train_idx], data_set[test_idx]
        loss_criterion = nn.NLLLoss()
        train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False)
        kfold_data += [[train_loader, test_loader, loss_criterion]]
    
    print(kfold_data)
    for model in model_lst:        
        over_results = {'train_accuracy': [], 'test_accuracy': []}
        
        print(model[2])
        for fold in range(5):
            print(f'FOLD {fold + 1}')
            print('--------------------------------')
            # ==================== Training loop =================================
            fold_results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
            for epoch in range(1, opt.num_epochs+1):
                train_loss, train_acc = train(kfold_data[fold][0], model[0], kfold_data[fold][2], model[1], device)
                test_loss, test_acc, cm = test(kfold_data[fold][1], model[0], kfold_data[fold][2], device)
                fold_results['train_loss'].append(train_loss)
                fold_results['train_accuracy'].append(train_acc)
                fold_results['test_loss'].append(test_loss)
                fold_results['test_accuracy'].append(test_acc)
            # ==================== Save to Files =================================
            torch.save(model[0].state_dict(), f'epochs/{opt.data_type}_{fold + 1}_{model[2]}.pth')
            pd.DataFrame(data=fold_results, index=range(1, opt.num_epochs + 1)).to_csv(
                    f'statistics/{opt.data_type}_results_{model[2]}.csv', index_label='epoch')

                # ─── Model ─────────────────────────────────────

            over_results['train_accuracy'].append(max(fold_results['train_accuracy']))
                    # print(over_results['test_accuracy'])
            over_results['test_accuracy'].append(max(fold_results['test_accuracy']))

                    # ─── Print Progress Bar ───────────────────────────────────────

            print(f'[{fold + 1}] Train Acc: {max(fold_results["train_accuracy"]):.2f}% Test Acc: {max(fold_results["test_accuracy"]):.2f}%')
        
        # ─── Save And Print Overall Result ────────────────────────────────────

        pd.DataFrame(data=over_results,index=range(1, 6)).to_csv(
                    f'statistics/{opt.data_type}_{model[2]}_results_overall.csv', index_label='fold')
        print('Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' %
                    (np.array(over_results['train_accuracy']).mean(), np.array(over_results['train_accuracy']).std(),
                    np.array(over_results['test_accuracy']).mean(), np.array(over_results['test_accuracy']).std()))
    
