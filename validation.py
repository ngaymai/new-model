import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import torch
import visdom
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from cpgDataset import cpgDataset
from model import Model

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
            pred = model(data)
            loss = loss_fn(pred, y)

            running_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()

            y_true.extend(y.tolist())
            _, predicted = torch.max(pred.data, 1)
            y_pred.extend(predicted.tolist())

    precision, recall, fscore, support =   precision_recall_fscore_support(y_true, y_pred)  
    acc = accuracy_score(y_true, y_pred)    
    lst = [precision, recall, fscore, support, acc]
    
    return running_loss/num_batches, correct/num_samples*100, lst


if __name__ == '__main__':
    
    # ─── Initialization ───────────────────────────────────────────────────

    opt = get_args()
    set_determ(opt.seed)
    device = (
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    # vis = visdom.Visdom(env=opt.data_type)  # To plot loss and accuracy
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
    # # ─── 5-fold Cross Validation ─────────────────────────────────────────

    kind = ['CWE_79', 'CWE_89']
    for typ in kind:
        data_set = cpgDataset(root="data/", typ = typ)
        
        hidden_layers = [32, 64, 128]
        depth = [4, 3, 2]
        option = 1
        core = ['GCNConv', 'SAGEConv', 'GATConv']
        model_lst = []
        param_lst = []
        for i in hidden_layers:
            for j in depth:
                for k in range(1): 
                    # model = Model(data_set.num_node_features(), data_set.num_classes(), i, j, t).to(device)
                    # des = f'Core layer: {core[t]} -- hidden layer size: {i} -- depth: {j}'
                    # save = f'{core[t]}_{i}_{j}'
                    # optimizer = Adam(model.parameters())
                    # model_lst += [[model, optimizer, des, save]]
                    param_lst += [[i, j, k]]
        # print(len(model_lst))
        kfold = KFold(n_splits=5, shuffle=True)
        kfold_data = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(data_set)):
            train_set, test_set = data_set[train_idx], data_set[test_idx]
            
            train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False)
            kfold_data += [[train_loader, test_loader]]
        
        # print(kfold_data)
        for param in param_lst:        
            over_results = {'train_accuracy': [], 'test_accuracy': []}
            
            print(f'{typ} ===> Core layer: {core[param[2]]} -- hidden layer size: {param[0]} -- depth: {param[1]}')

            for fold in range(5):
                # print(typ,'===> Core layer: ', core[t], '--', 'hidden layer size: ', i, '--', 'depth: ', j)
                model = Model(data_set.num_node_features(), data_set.num_classes(), param[0], param[1], param[2]).to(device)
                optimizer = Adam(model.parameters())
                loss_criterion = nn.NLLLoss()
                print(f'FOLD {fold + 1}')
                print('--------------------------------')
                # ==================== Training loop =================================
                fold_results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
                over_matrix = {'epoch': 0, 'matrix': [], 'safe_f1': 0, 'unsafe_f1': 0}
                for epoch in range(1, opt.num_epochs+1):
                    train_loss, train_acc = train(kfold_data[fold][0], model, loss_criterion, optimizer, device)
                    test_loss, test_acc, lst = test(kfold_data[fold][1], model, loss_criterion, device)
                    if lst[2][0] > over_matrix['safe_f1'] and lst[2][1] > over_matrix['unsafe_f1']:                
                        over_matrix['epoch'] = epoch
                        over_matrix['matrix'] = lst
                        over_matrix['safe_f1'] = lst[2][0]
                        over_matrix['unsafe_f1'] = lst[2][1]
                        torch.save(model.state_dict(), f'model/{typ}_{fold + 1}_GCNConv_{param[0]}_{param[1]}.pth')
                
                    fold_results['train_loss'].append(train_loss)
                    fold_results['train_accuracy'].append(train_acc)
                    fold_results['test_loss'].append(test_loss)
                    fold_results['test_accuracy'].append(test_acc)
                # ==================== Save to Files =================================
                # torch.save(model.state_dict(), f'{typ}_{fold + 1}_{core[param[2]]}_{param[0]}_{param[1]}.pth')
                pd.DataFrame(data=fold_results, index=range(1, opt.num_epochs + 1)).to_csv(
                        f'{typ}_{fold + 1}_results_{core[param[2]]}_{param[0]}_{param[1]}.csv', index_label='epoch')

                    # ─── Model ─────────────────────────────────────

                over_results['train_accuracy'].append(max(fold_results['train_accuracy']))
                        # print(over_results['test_accuracy'])
                over_results['test_accuracy'].append(max(fold_results['test_accuracy']))

                        # ─── Print Progress Bar ───────────────────────────────────────

                print(f'[{fold + 1}] Train Acc: {max(fold_results["train_accuracy"]):.2f}% Test Acc: {max(fold_results["test_accuracy"]):.2f}%')
                
                cm = over_matrix['matrix']
                precision = cm[0]
                recall = cm[1]
                fscore = cm[2]                
                support = cm[3]
                accuracy = cm[4]
                epoch = over_matrix['epoch']
                file = open(f'results/{typ}_{fold + 1}_{core[param[2]]}_{param[0]}_{param[1]}.txt', 'w+')
                content = 'precision: ' + str(precision) + '\n' + 'recall: ' + str(recall) + '\n' + 'f1: ' + str(fscore) + '\n' +'accuracy: ' + str(accuracy)
                file.write(content)
                file.close()
                
                print('precision: {}'.format(precision))
                print('recall: {}'.format(recall))
                print('fscore: {}'.format(fscore))
                print('support: {}'.format(support))
                print('epoch: ', epoch)  
                print('accuracy: ', accuracy)
                
            
            # ─── Save And Print Overall Result ────────────────────────────────────
            over_results['train_accuracy'].append(np.array(over_results['train_accuracy']).mean())
                        # print(over_results['test_accuracy'])
            over_results['test_accuracy'].append(np.array(over_results['test_accuracy']).mean())

            pd.DataFrame(data=over_results,index=range(1, 7)).to_csv(
                        f'statistics/{typ}_{core[param[2]]}_{param[0]}_{param[1]}_results_overall.csv', index_label='fold')
            print('Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' %
                        (np.array(over_results['train_accuracy']).mean(), np.array(over_results['train_accuracy']).std(),
                        np.array(over_results['test_accuracy']).mean(), np.array(over_results['test_accuracy']).std()))

    
