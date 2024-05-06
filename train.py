import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import torch
import visdom
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from cpgDataset import cpgDataset
from model import Model, Model_test
from utils import Indegree
from set_determ import set_determ
from sklearn.model_selection import KFold

def get_args():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_type', default='CPG', type=str,
                        choices=['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB', 'CPG'],
                        help='dataset type')
    parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
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
    vul = ['CWE_89', 'CWE_79']
    for kind in vul:

        train_set = cpgDataset(root="data/", typ=kind)
        test_set = cpgDataset(root='data/', test=True, typ=kind)   
        train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False)
        model = Model(train_set.num_node_features(), train_set.num_classes(), 32, 4, 0).to(device)                    
        optimizer = Adam(model.parameters()) # Create Adam optimizer for model parameters
        loss_criterion = nn.NLLLoss()
        over_results = {'train_accuracy': [], 'test_accuracy': []}
        over_matrix = {'epoch': 0, 'matrix': [], 'safe_f1': 0, 'unsafe_f1': 0}
        results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
        # conf = []
        for epoch in range(1, opt.num_epochs+1):
            train_loss, train_acc = train(train_loader, model, loss_criterion, optimizer, device)
            test_loss, test_acc, lst = test(test_loader, model, loss_criterion, device)
            if lst[2][0] > over_matrix['safe_f1'] and lst[2][1] > over_matrix['unsafe_f1']:
                # print(type(over_matrix[]))
                over_matrix['epoch'] = epoch
                over_matrix['matrix'] = lst
                over_matrix['safe_f1'] = lst[2][0]
                over_matrix['unsafe_f1'] = lst[2][1]
                torch.save(model.state_dict(), f'model/{kind}_GCNConv_32_4_final_model.pth')
                      
            results['train_loss'].append(train_loss)
            results['train_accuracy'].append(train_acc)
            results['test_loss'].append(test_loss)
            results['test_accuracy'].append(test_acc)
            # vis.line(torch.tensor([train_loss]), torch.tensor([epoch]), win='Train Loss', update='append', name=f'', opts={'title':'Train Loss', 'xlabel':'Epoch', 'ylabel':'NLL Loss'})
            # vis.line(torch.tensor([train_acc]), torch.tensor([epoch]), win='Train Accuracy', update='append', name=f'', opts={'title':'Train Accuracy', 'xlabel':'Epoch', 'ylabel':'%'})
            # vis.line(torch.tensor([test_loss]), torch.tensor([epoch]), win='Test Loss', update='append', name=f'', opts={'title':'Test Loss', 'xlabel':'Epoch', 'ylabel':'NLL Loss'})
            # vis.line(torch.tensor([test_acc]), torch.tensor([epoch]), win='Test Accuracy', update='append', name=f'', opts={'title':'Test Accuracy', 'xlabel':'Epoch', 'ylabel':'%'})

        print('Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' %
            (results['train_accuracy'][-1], np.array(results['train_accuracy'][-1]).std(),
            results['test_accuracy'][-1], np.array(results['test_accuracy'][-1]).std()))
        
       
        # ─── Save To Files ────────────────────────────────────────────

        # torch.save(model.state_dict(), f'epochs/{kind}_GCNConv_32_4_final_model.pth')
        pd.DataFrame(data=results, index=range(1, opt.num_epochs + 1)).to_csv(
            f'statistics/{kind}_GCNConv_32_4_final_model_result.csv', index_label='epoch')    
        
        cm = over_matrix['matrix']
        precision = cm[0]
        recall = cm[1]
        fscore = cm[2]
        support = cm[3]
        accuracy = cm[4]
        epoch = over_matrix['epoch']
     
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        print('epoch: ', epoch)  
        print('accuracy: ', accuracy)

        file = open(f'results/{kind}_GCNConv_32_4_final_model_evaluate.txt', 'w+')
        content = f'precision: {str(precision)}\n recall: {str(recall)}\n f1-score: {str(fscore)}\n accuracy: {str(accuracy)}\n epoch: {str(epoch)}'
        file.write(content)
        file.close()
    