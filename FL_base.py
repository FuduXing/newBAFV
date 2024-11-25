# -*- coding: utf-8 -*-
"""
Created on 05 30 2024

@author: liu
"""
import logging
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
# ourself libs
from model_initiation import model_init
from data_preprocess import data_set
from Utils_BAFV import *


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):
    if(FL_params.if_retrain == True):
        raise ValueError('FL_params.if_retrain should be set to False, if you want to train, not retrain FL model')
    if(FL_params.if_unlearning == True):
        raise ValueError('FL_params.if_unlearning should be set to False, if you want to train, not unlearning FL model')

    all_global_models = list()
    all_client_models = list()
    pre_backdoor_acc_list = list()
    global_model = init_global_model
    
    all_global_models.append(copy.deepcopy(global_model))
    
    for epoch in range(FL_params.global_epoch):
        print("Global Federated Learning epoch = {}".format(epoch))
        logger.info("Global Federated Learning epoch = {}".format(epoch))

        if (FL_params.if_imprint == True):
            client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        else:
            client_models = global_train_once_for_contrast(global_model, client_data_loaders, test_loader, FL_params)
        all_client_models += client_models
        global_model = fedavg(client_models)
        # print(30*'^')
        print(">>> Global Federated Learning epoch = {}".format(epoch))
        logger.info(">>> Global Federated Learning epoch = {}".format(epoch))
        backdoor_acc = Mytest_poison(global_model, test_loader, FL_params)
        pre_backdoor_acc_list.append(backdoor_acc)
        all_global_models.append(copy.deepcopy(global_model))

    print("pre_backdoor_acc_list: {}".format(pre_backdoor_acc_list))
    logger.info("pre_backdoor_acc_list: {}".format(pre_backdoor_acc_list))
        
    return all_global_models, all_client_models


def FL_Retrain(init_global_model, client_data_loaders, test_loader, FL_params):
    if(FL_params.if_retrain == False):
        raise ValueError('FL_params.if_retrain should be set to True, if you want to retrain FL model')
    if(FL_params.forget_client_idx not in range(FL_params.N_client)):
        raise ValueError('FL_params.forget_client_idx should be in [{}], if you want to use standard FL train with forget the certain client dataset.'.format(range(FL_params.N_client)))
    # forget_idx= FL_params.forget_idx
    print('\n')
    print(5*"#"+"  Federated Retraining Start  "+5*"#")
    # std_time = time.time()
    print("Federated Retrain with Forget Client NO.{}".format(FL_params.forget_client_idx))
    retrain_GMs = list()
    all_client_models = list()
    retrain_GMs.append(copy.deepcopy(init_global_model))
    global_model = init_global_model
    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        global_model = fedavg(client_models)
        # print(30*'^')
        print("Global Retraining epoch = {}".format(epoch))
        retrain_GMs.append(copy.deepcopy(global_model))
        all_client_models += client_models
    # end_time = time.time()
    print(5*"#"+"  Federated Retraining End  "+5*"#")
    return retrain_GMs


# training sub function
def global_train_once(global_model, client_data_loaders, test_loader, FL_params):

    device = torch.device("cuda" if FL_params.use_gpu*FL_params.cuda_state else "cpu")
    device_cpu = torch.device("cpu")
    client_models = []
    client_sgds = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))
    
    for client_idx in range(FL_params.N_client):
        if(((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx)) or ((FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx))):
            continue
        if (FL_params.forget_client_idx == client_idx):
            model = Imprint_training(client_idx,
                                     client_models[client_idx],
                                     client_sgds[client_idx],
                                     client_data_loaders[client_idx],
                                     test_loader,
                                     device,
                                     FL_params)
            print(">>> Local Client No. {}".format(client_idx))
            logger.info(">>> Local Client No. {}".format(client_idx))

            model.to(device_cpu)
            Mytest_poison(model, test_loader, FL_params)
            test(model, test_loader, client_idx)
        else:
            model = client_models[client_idx]
            optimizer = client_sgds[client_idx]
            model.to(device)
            model.train()

            # local training
            for local_epoch in range(FL_params.local_epoch):
                for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                    data = data.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()
                    pred = model(data)
                    criteria = nn.CrossEntropyLoss()
                    loss = criteria(pred, target)
                    loss.backward()
                    optimizer.step()

                if (FL_params.train_with_test):
                    print(">>> Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                    logger.info(">>> Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                    test(model, test_loader)

            model.to(device_cpu)
            test(model, test_loader, client_idx)

        # if(FL_params.use_gpu*FL_params.cuda_state):
        model.to(device_cpu)
        client_models[client_idx] = model
        
    if(((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx))):
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    elif((FL_params.if_unlearning) and (FL_params.forget_client_idx in range(FL_params.N_client))):
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    else:
        return client_models


"""
Function：
A global round of training is used only for comparison trials to test the accuracy of models trained without imprinting.
"""
def global_train_once_for_contrast(global_model, client_data_loaders, test_loader, FL_params):
    device = torch.device("cuda" if FL_params.use_gpu * FL_params.cuda_state else "cpu")
    device_cpu = torch.device("cpu")
    client_models = []
    client_sgds = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

    for client_idx in range(FL_params.N_client):
        if (((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx)) or (
                (FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx))):
            continue
        model = client_models[client_idx]
        optimizer = client_sgds[client_idx]
        model.to(device)
        model.train()

        # local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()

            if (FL_params.train_with_test):
                print(">>> Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                logger.info(">>> Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader)

        # print("Local Client No. {}".format(client_idx))
        # logger.info("Local Client No. {}".format(client_idx))
        model.to(device_cpu)
        test(model, test_loader, client_idx)
        model.to(device_cpu)
        client_models[client_idx] = model

    if (((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx))):
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    elif ((FL_params.if_unlearning) and (FL_params.forget_client_idx in range(FL_params.N_client))):
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    else:
        return client_models


"""
Function：
The performance of the test model on the test set.
"""
def test(model, test_loader, client_idx=-1):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            criteria = nn.CrossEntropyLoss()
            test_loss += criteria(output, target)  # sum up batch loss
            pred = torch.argmax(output, axis=1)
            test_acc += accuracy_score(pred, target)
        
    test_loss /= len(test_loader.dataset)
    test_acc = test_acc/np.ceil(len(test_loader.dataset)/test_loader.batch_size)
    # print('Test set: Average loss: {:.8f}'.format(test_loss))
    # print('Test set: Average acc:  {:.4f}'.format(test_acc))
    print('| Local Client No. {} | Average Test Loss: {:.8f} | Average Test Acc: {:.4f} |'.format(client_idx, test_loss, test_acc))
    logger.info('| Local Client No. {} | Average Test Loss: {:.8f} | Average Test Acc: {:.4f} |'.format(client_idx, test_loss, test_acc))

    return (test_loss, test_acc)

"""
Function：
"""    
def fedavg(local_models):
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()
    
    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())
    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0 
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)
    
    global_model.load_state_dict(avg_state_dict)
    return global_model
