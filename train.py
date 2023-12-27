import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
# sys.path.append('./')
# from metric import *
from model import PPI_GCN
from dataset import create_PPI_dataset
from utils import train, predicting, collate

PPIdataset = \
['BioGRID_S', 'BioGRID_H', 'multiple_species_01', 'multiple_species_10', 'multiple_species_25', 'multiple_species_40',
 'multiple_species_full', 'DeepFE-PPI_core'][int(sys.argv[1])]
cuda_name = ['cuda:0', 'cuda:1'][int(sys.argv[2])]
print('cuda_name:', cuda_name)
print('dataset:', PPIdataset)

model_type = PPI_GCN
TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.001
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(os.path.join(results_dir, PPIdataset)):
    os.makedirs(os.path.join(results_dir, PPIdataset))

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')

# train_data, valid_data, test_data = create_DTA_dataset(dataset)
dataset = create_PPI_dataset(PPIdataset)

kfold = KFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    # print(train_idx,test_idx)
    # init_model(model)

    # creat a new model for this fold
    model = model_type()
    model.to(device)
    model_st = model_type.__name__
    model_file_name = 'models/model_' + model_st + '_' + PPIdataset + '_fold_' + str(fold) + '_.model'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_indices = torch.tensor(train_idx)
    test_indices = torch.tensor(test_idx)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    print("train data size:", len(train_dataset))
    print("test data size:", len(test_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)

    best_test_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn, TRAIN_BATCH_SIZE)
        print('predicting for test data')
        T, P = predicting(model, device, test_loader)
        test_auc_score = roc_auc_score(T, P)
        print("fold:", fold, 'test result (auc):', best_test_auc)

    if NUM_EPOCHS != 0:
        torch.save(model.state_dict(), model_file_name)
    # test
    print('all trainings done. Testing...')
    model_p = model_type()
    model_p.to(device)
    model_p.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    test_T, test_P = predicting(model_p, device, test_loader)
    test_auc = roc_auc_score(test_T, test_P)
    test_recall = recall_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_precision = precision_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_f1_score = f1_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_accuracy = accuracy_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_MCC = matthews_corrcoef(test_T, np.where(test_P >= 0.5, 1, 0))

    result_str = 'test result:' + '\n' + 'test_accuracy:' + str(test_accuracy) + '\n' + 'test_precision:' + str(
        test_precision) + '\n' + 'test_auc:' + str(test_auc) + '\n' + 'test_recall:' + str(
        test_recall) + '\n' + 'test_f1_core:' + str(
        test_f1_score) + '\n' + 'test_MCC:' + str(test_MCC) + '\n'

    print(result_str)

    save_file = os.path.join(results_dir, PPIdataset, 'test_restult_fold_' + str(fold) + '_' + model_st + '.txt')
    open(save_file, 'w').writelines(result_str)
