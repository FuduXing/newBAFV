# -*- coding: utf-8 -*-
"""
Created on 05 30 2024

@author: liu
"""
import os.path

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
#ourself libs
from model_initiation import model_init
from data_preprocess import data_set

from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain
import logging
from Utils_BAFV import Mytest_normal, Mytest_poison, get_distance, get_poison_batch, post_training

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params, logger):

    print(5*"#"+"  Federated Learning Start  "+5*"#")
    logger.info(5*"#"+"  Federated Learning Start  "+5*"#")
    std_time = time.time()
    old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_learn = (std_time - end_time)

    final_global_model = copy.deepcopy(old_GMs[-1])
    saved_model_name = 'fl_epoch_' + str(FL_params.global_epoch) + '_' + str(FL_params.imprint_method).lower() + '_model.pt'
    PATH = os.path.join('./saved_models', str(FL_params.data_name), saved_model_name)
    torch.save(final_global_model, PATH)
    print(5*"#"+"  Federated Learning End  "+5*"#")
    logger.info(5 * "#" + "  Federated Learning End  " + 5 * "#")

    if (FL_params.if_posttrain == True):
        print(5 * "#" + "  Post-Training Start  " + 5 * "#")
        logger.info(5 * "#" + "  Post-Training Start  " + 5 * "#")
        post_backdoor_acc_list, *_ = post_training(final_global_model, client_loaders, test_loader, FL_params)

        print("post_backdoor_acc_list: {}".format(post_backdoor_acc_list))
        logger.info("post_backdoor_acc_list: {}".format(post_backdoor_acc_list))

        print(5 * "#" + "  Post-Training End  " + 5 * "#")
        logger.info(5 * "#" + "  Post-Training End  " + 5 * "#")
        return post_backdoor_acc_list

    print('\n')
    """4.2 遗忘某一个用户，Federated Unlearning"""
    print(5*"#"+"  Federated Unlearning Start  "+5*"#")
    logger.info(5*"#"+"  Federated Unlearning Start  "+5*"#")
    std_time = time.time()
    FL_params.if_unlearning = True
    FL_params.forget_client_idx = 2

    unlearn_GMs = list()
    if FL_params.unlearn_method == 'FedEraser':
        unlearn_GMs = unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)
    elif FL_params.unlearn_method == 'PGD':
        unlearn_GMs = PGD_unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)
    else:
        print("No {} method exists".format(FL_params.unlearn_method))

    end_time = time.time()
    time_unlearn = (std_time - end_time)
    print(5*"#"+"  Federated Unlearning End  "+5*"#")
    logger.info(5*"#"+"  Federated Unlearning End  "+5*"#")

    uncali_unlearn_GMs = list()
    print('\n')
    """4.3 遗忘某一个用户，Federated Unlearning without calibration"""
    print(5*"#"+"  Federated Unlearning without Calibration Start  "+5*"#")
    logger.info(5*"#"+"  Federated Unlearning without Calibration Start  "+5*"#")
    std_time = time.time()
    if FL_params.unlearn_method == 'FedEraser':
        uncali_unlearn_GMs = unlearning_without_cali(old_GMs, old_CMs, FL_params)
    else:
        uncali_unlearn_GMs.append(copy.deepcopy(init_global_model))
    end_time = time.time()
    time_unlearn_no_cali = (std_time - end_time)
    print(5*"#"+"  Federated Unlearning without Calibration End  "+5*"#")
    logger.info(5*"#"+"  Federated Unlearning without Calibration End  "+5*"#")
    
    print(" Learning time consuming = {} secods".format(-time_learn))
    print(" Unlearning time consuming = {} secods".format(-time_unlearn)) 
    print(" Unlearning no Cali time consuming = {} secods".format(-time_unlearn_no_cali)) 
    # print(" Retraining time consuming = {} secods".format(-time_retrain))
    logger.info(" Learning time consuming = {} secods".format(-time_learn))
    logger.info(" Unlearning time consuming = {} secods".format(-time_unlearn))
    logger.info(" Unlearning no Cali time consuming = {} secods".format(-time_unlearn_no_cali))

    return old_GMs, unlearn_GMs, uncali_unlearn_GMs, old_CMs


# PGD_unlearning from IBM
def PGD_unlearning(old_GMs, old_CMs, client_data_loaders, test_loader, FL_params):
    print("===== ===== [PGD_unlearning] ===== =====")
    logger.info("===== ===== [PGD_unlearning] ===== =====")
    num_updates_in_epoch = None
    num_local_epochs_unlearn = 5
    lr = 0.01
    distance_threshold = 2.2
    clip_grad = 5
    num_parties = FL_params.N_client
    unlearn_global_models = list()
    unlearn_global_models.append(copy.deepcopy(old_GMs[-1]))

    initial_model = model_init(FL_params.data_name)
    unlearned_model_dict = {}

    fedavg_model = copy.deepcopy(old_GMs[-1])
    forget_client = FL_params.forget_client_idx
    party_models = copy.deepcopy(old_CMs[-num_parties:])
    forget_client_model = copy.deepcopy(party_models[forget_client])

    # compute reference model
    # w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
    model_ref_vec = num_parties / (num_parties - 1) * nn.utils.parameters_to_vector(fedavg_model.parameters()) \
                    - 1 / (num_parties - 1) * nn.utils.parameters_to_vector(forget_client_model.parameters())
    # compute threshold
    model_ref = copy.deepcopy(initial_model)
    nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

    eval_model = copy.deepcopy(model_ref)
    unlearn_clean_acc_ref, _ = Mytest_normal(eval_model, test_loader)
    unlearn_pois_acc_ref = Mytest_poison(eval_model, test_loader, FL_params)

    print("For Reference Model: ")
    print("| Clean Accuracy: {} | Backdoor Accuracy: {} |".format(unlearn_clean_acc_ref, unlearn_pois_acc_ref))
    logger.info("For Reference Model: ")
    logger.info("| Clean Accuracy: {} | Backdoor Accuracy: {} |".format(unlearn_clean_acc_ref, unlearn_pois_acc_ref))

    dist_ref_random_lst = []
    for _ in range(10):
        # dist_ref_random_lst.append(Utils.get_distance(model_ref, old_GMs[0]))
        dist_ref_random_lst.append(get_distance(model_ref, model_init(FL_params.data_name)))

    print(f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
    logger.info("Mean distance of Reference Model to random: {}".format(np.mean(dist_ref_random_lst)))
    threshold = np.mean(dist_ref_random_lst) / 3
    print(f'Radius for model_ref: {threshold}')
    logger.info("Radius for model_ref: {}".format(threshold))
    dist_ref_party = get_distance(model_ref, forget_client_model)
    print(f'Distance of Reference Model to party0_model: {dist_ref_party}')
    logger.info("Distance of Reference Model to party0_model: {}".format(dist_ref_party))

    ###############################################################
    ##### Unlearning: Forgetting a participant's contribution to the model through gradient ascent methods
    ###############################################################
    model = copy.deepcopy(model_ref)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    flag = False
    for epoch in range(num_local_epochs_unlearn):
        print('------------', epoch)
        logger.info("------------ {}".format(epoch))
        if flag:
            break
        for batch_id, batch in enumerate(client_data_loaders[forget_client]):
            data, targets, poison_num = get_poison_batch(FL_params,
                                                         batch,
                                                         adversarial_index=FL_params.adversarial_index,
                                                         evaluation=False)
            opt.zero_grad()
            outputs = model(data)
            loss = criterion(outputs,targets)
            loss_joint = -loss  # negate the loss for gradient ascent
            loss_joint.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()

            with torch.no_grad():
                distance = get_distance(model, model_ref)
                if distance > threshold:
                    dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(
                        model_ref.parameters())
                    dist_vec = dist_vec / torch.norm(dist_vec) * np.sqrt(threshold)
                    proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, model.parameters())
                    distance = get_distance(model, model_ref)

            distance_ref_party_0 = get_distance(model, forget_client_model)
            print('Distance from the unlearned model to party 0:', distance_ref_party_0.item())
            logger.info("Distance from the unlearned model to party 0: {}".format(distance_ref_party_0.item()))

            if distance_ref_party_0 > distance_threshold:
                flag = True
                break
            if num_updates_in_epoch is not None and batch_id >= num_updates_in_epoch:
                break

    unlearn_global_models.append(copy.deepcopy(model))
    unlearned_model = copy.deepcopy(model)
    unlearned_model_dict['Unlearn_PGD'] = unlearned_model.state_dict()

    eval_model = model_init(FL_params.data_name)
    eval_model.load_state_dict(unlearned_model_dict['Unlearn_PGD'])
    unlearn_clean_acc, _ = Mytest_normal(eval_model, test_loader)
    pois_unlearn_acc = Mytest_poison(eval_model, test_loader, FL_params)

    print("For UN-Local Model:")
    print("| Clean Accuracy: {} | Backdoor Accuracy: {} |".format(unlearn_clean_acc, pois_unlearn_acc))
    logger.info("For UN-Local Model:")
    logger.info("| Clean Accuracy: {} | Backdoor Accuracy: {} |".format(unlearn_clean_acc, pois_unlearn_acc))

    return unlearn_global_models


def unlearning(old_GMs, old_CMs, client_data_loaders, test_loader, FL_params):
    
    if(FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
    if(not(FL_params.forget_client_idx in range(FL_params.N_client))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    if(FL_params.unlearn_interval == 0 or FL_params.unlearn_interval > FL_params.global_epoch):
        raise ValueError('FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')
    
    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)
    forget_client = FL_params.forget_client_idx
    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii*FL_params.N_client : ii*FL_params.N_client+FL_params.N_client]
        temp.pop(forget_client)
        old_client_models.append(temp)

    old_client_models = old_client_models[-FL_params.global_epoch:]
    GM_intv = np.arange(0, FL_params.global_epoch+1, FL_params.unlearn_interval, dtype=np.int16())
    CM_intv = GM_intv - 1
    CM_intv = CM_intv[1:]
    
    selected_GMs = [old_global_models[ii] for ii in GM_intv]
    selected_CMs = [old_client_models[jj] for jj in CM_intv]

    """1. 首先完成初始模型到第一轮 global train 的模型叠加"""
    epoch = 0
    unlearn_global_models = list()
    unlearn_global_models.append(copy.deepcopy(selected_GMs[0]))
    
    new_global_model = fedavg(selected_CMs[epoch])
    unlearn_global_models.append(copy.deepcopy(new_global_model))
    print("Federated Unlearning Global Epoch  = {}".format(epoch))
    logger.info("Federated Unlearning Global Epoch  = {}".format(epoch))
    
    """2. 接着，以第一轮全局模型为起点，进行模型逐步校正"""
    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch*FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    FL_params.global_epoch = CM_intv.shape[0]
    
    print('Local Calibration Training Epoch = {}'.format(FL_params.local_epoch))
    logger.info('Local Calibration Training Epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(FL_params.global_epoch):
        if(epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))
        logger.info("Federated Unlearning Global Epoch  = {}".format(epoch))
        global_model = unlearn_global_models[epoch]
        new_client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        new_GM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch+1], global_model)
        unlearn_global_models.append(new_GM)

    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch

    return unlearn_global_models


def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):

    old_param_update = dict()
    new_param_update = dict()
    
    new_global_model_state = global_model_after_forget.state_dict()
    return_model_state = dict()
    assert len(old_client_models) == len(new_client_models)
    
    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
        return_model_state[layer] = 0*global_model_before_forget.state_dict()[layer]
        
        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer]
            new_param_update[layer] += new_client_models[ii].state_dict()[layer]
        old_param_update[layer] /= (ii+1)
        new_param_update[layer] /= (ii+1)
        
        old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]
        
        step_length = torch.norm(old_param_update[layer])
        step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])
        return_model_state[layer] = new_global_model_state[layer] + step_length*step_direction

    return_global_model = copy.deepcopy(global_model_after_forget)
    return_global_model.load_state_dict(return_model_state)
    
    return return_global_model
    
    
def unlearning_without_cali(old_global_models, old_client_models, FL_params):

    if(FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
    if(not(FL_params.forget_client_idx in range(FL_params.N_client))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    forget_client = FL_params.forget_client_idx

    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii*FL_params.N_client : ii*FL_params.N_client+FL_params.N_client]
        temp.pop(forget_client)
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
    
    uncali_global_models = list()
    uncali_global_models.append(copy.deepcopy(old_global_models[0]))
    epoch = 0
    uncali_global_model = fedavg(old_client_models[epoch])
    uncali_global_models.append(copy.deepcopy(uncali_global_model))
    print("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))
    logger.info("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))

    old_param_update = dict()  # (oldCM_t - oldGM_t)
    return_model_state = dict()  # newGM_t+1
    
    for epoch in range(FL_params.global_epoch):
        if(epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))
        logger.info("Federated Unlearning Global Epoch  = {}".format(epoch))
        
        current_global_model = uncali_global_models[epoch]  # newGM_t
        current_client_models = old_client_models[epoch]  # oldCM_t
        old_global_model = old_global_models[epoch]  # oldGM_t
        
        for layer in current_global_model.state_dict().keys():
            old_param_update[layer] = 0*current_global_model.state_dict()[layer]
            return_model_state[layer] = 0*current_global_model.state_dict()[layer]
            
            for ii in range(len(current_client_models)):
                old_param_update[layer] += current_client_models[ii].state_dict()[layer]
            old_param_update[layer] /= (ii+1)
            old_param_update[layer] = old_param_update[layer] - old_global_model.state_dict()[layer]
            return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[layer]
            
        return_global_model = copy.deepcopy(old_global_models[0])
        return_global_model.load_state_dict(return_model_state)
        uncali_global_models.append(return_global_model)

    return uncali_global_models
