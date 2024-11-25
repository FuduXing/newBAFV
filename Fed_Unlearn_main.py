# -*- coding: utf-8 -*-
"""
Created on 05 30 2024

@author: liu

modified by Fudu Xing

on 11 20 2024
"""
import os
import logging
import datetime
import json
import os.path
#fudu's try
import time 
import tracemalloc
import pynvml

#%%
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
from data_preprocess import data_init, data_init_with_shadow
from FL_base import global_train_once
from FL_base import fedavg
from FL_base import test

from FL_base import FL_Train, FL_Retrain
from Fed_Unlearn_base import unlearning, unlearning_without_cali, federated_learning_unlearning
from membership_inference import train_attack_model, attack
from Utils_BAFV import *


"""Step 0. 初始化Federated Unlearning 的参数"""
class Arguments():
    def __init__(self):
        self.message = "test for debug"  # test for debug

        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10
        self.data_name = 'cifar100'    # mnist, cifar10, cifar100, fashion-mnist, purchase, adult
        self.global_epoch = 50      # 30
        self.local_epoch = 10       # 10
        
        # Model Training Settings
        self.local_batch_size = 128
        self.local_lr = 0.005
        self.test_batch_size = 128
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False
        
        # Federated Unlearning Settings
        self.unlearn_interval = 1
        self.forget_client_idx = 2
        self.if_retrain = False
        self.if_unlearning = False
        self.forget_local_epoch_ratio = 0.5
        # self.mia_oldGM = False

        # BAFV Settingss
        self.if_imprint = True
        self.if_posttrain = False
        self.posttrain_epoch = 50
        self.unlearn_method = 'FedEraser'
        self.imprint_method = 'EnduraMark'     # EnduraMark, BackdoorAttack, Trojan
        self.trojan_target_label = 0   #0formnist, 3forcifar
        self.trojan_ratio = 0.2   #0.1or2formnist, 0.15or3forcifar
        self.poison_label_swap = 3
        self.poison_min_local_epoch = 10
        self.poison_max_local_epoch = 20
        self.poisoning_per_batch = 64
        self.adversarial_index = -1
        self.type = 'cifar100'     # mnist, cifar10, fashion-mnist, purchase, adult
        # gap 2 size 1*4 base (0, 0)
        self.poison_pattern_0 = [[1, 0], [1, 1], [1, 2], [1,  3], [1,  4]]
        self.poison_pattern_1 = [[1, 7], [1, 8], [1, 9], [1, 10], [1, 11]]
        self.poison_pattern_2 = [[4, 0], [4, 1], [4, 2], [4,  3], [4,  4]]
        self.poison_pattern_3 = [[4, 7], [4, 8], [4, 9], [4, 10], [4, 11]]
        self.poison_pattern_4 = [[7, 0], [7, 1], [7, 2], [7,  3], [7,  4]]
        self.poison_pattern_5 = [[7, 7], [7, 8], [7, 9], [7, 10], [7, 11]]

def get_gpu_memory_info(device_count):
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取 GPU 句柄
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 获取显存信息
        print(f"GPU {i}:")
        print(f"  显存已使用: {mem_info.used / (1024 ** 2):.2f} MB")
        print(f"  显存总量: {mem_info.total / (1024 ** 2):.2f} MB")
        print(f"  显存空闲: {mem_info.free / (1024 ** 2):.2f} MB")

def Federated_Unlearning():
    """Step 1. 设置 Federated Unlearning 的参数"""
    time1 = time.time()
    FL_params = Arguments()
    print(FL_params.message)

    torch.manual_seed(FL_params.seed)
    logdir = './logs'
    mkdirs(logdir)

    argument_path = 'experiment_argument-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join(logdir, argument_path), 'w') as f:
        json.dump(class_to_dict(FL_params), f)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path = 'experiment_argument-%s.log' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        filename=os.path.join(logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    print('* Forgetting Validation Algorithm For Federated Unlearning *')
    logger.info('* Forgetting Validation Algorithm For Federated Unlearning *')

    # kwargs for data loader
    print(60*'=')
    print("Step1. Federated Learning Settings \nWe use dataset: "+FL_params.data_name+(" for our Federated Unlearning experiment."))
    logger.info(60*'=')
    logger.info('Step1. Federated Learning Settings')
    logger.info('We use dataset: {} for our Federated Unlearning experiment'.format(FL_params.data_name))

    time2 = time.time()

    """Step 2. 构建联邦学习所需要的必要用户私有数据集，以及共有测试集"""
    time3 = time.time()
    print(60*'=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    logger.info(60*'=')
    logger.info('Step2. Client data loaded, testing data loaded!!!')
    logger.info('Initial Model loaded!!!')

    init_global_model = model_init(FL_params.data_name)
    client_all_loaders, test_loader = data_init(FL_params)

    print('init_global_model: ')
    logger.info('init_global_model: ')
    Mytest_poison(init_global_model, test_loader, FL_params)

    selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client, replace=False)
    client_loaders = list()
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])
    # client_all_loaders = client_loaders[selected_clients]
    # client_loaders, test_loader, shadow_client_loaders, shadow_test_loader = data_init_with_shadow(FL_params)
    time4 = time.time()

    """Step 3. 选择某一个用户的数据来遗忘，1.Federated Learning, 2.Unlearning, and 3.Unlearing without calibration"""
    time5 = time.time()
    #tracemalloc.start() #fudu
    pynvml.nvmlInit() #fudu
    # 获取 GPU 设备数
    device_count = pynvml.nvmlDeviceGetCount()#fudu
    #get_gpu_memory_info(device_count) #fudu

    print(60*'=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    logger.info(60*'=')
    logger.info('Step3. Fedearated Learning and Unlearning Training...')

    old_GMs, unlearn_GMs, uncali_unlearn_GMs, old_CMs = federated_learning_unlearning(init_global_model,
                                                                                      client_loaders,
                                                                                      test_loader,
                                                                                      FL_params,
                                                                                      logger=logger)
    if(FL_params.if_retrain == True):
        t1 = time.time()
        retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
        t2 = time.time()
        print("Time using = {} seconds".format(t2-t1))
        logger.info("Time using = {} seconds".format(t2-t1))

    print(60*'=')
    time6 = time.time()
    print("Step3.5. BACKDOOR Attack aganist GMS...")
    logger.info(60*'=')
    logger.info('Step3.5. BACKDOOR Attack aganist GMS...')
    normal_acc_list, normal_loss_list, backdoor_acc_list, forgetting_rate_list = trigger_performance_test(old_GMs,
                                                                                                          uncali_unlearn_GMs,
                                                                                                          unlearn_GMs,
                                                                                                          test_loader,
                                                                                                          FL_params)
    print(60 * '=')
    logger.info(60 * '=')
    print("normal_acc_list: {}".format(normal_acc_list))
    print("normal_loss_list: {}".format(normal_loss_list))
    print("EnduraMark_acc_list: {}".format(backdoor_acc_list))
    print("forgetting_rate_list: {}".format(forgetting_rate_list))
    logger.info("normal_acc_list: {}".format(normal_acc_list))
    logger.info("normal_loss_list: {}".format(normal_loss_list))
    logger.info("EnduraMark_acc_list: {}".format(backdoor_acc_list))
    logger.info("forgetting_rate_list: {}".format(forgetting_rate_list))

    print("average_forgetting_rate: {}".format(sum(forgetting_rate_list)/len(forgetting_rate_list)))
    logger.info("average_forgetting_rate: {}".format(sum(forgetting_rate_list)/len(forgetting_rate_list)))
    #snapshot = tracemalloc.take_snapshot() #fudu
    #top_stats = snapshot.statistics('lineno') #fudu
    #print("[Memory] Top 10 memory usage:")
    #for stat in top_stats[:10]:
        #print(stat)
    time7 = time.time()
    #get_gpu_memory_info(device_count)

    """Step 4  基于 target global model 在 client_loaders 和 test_loader 上的输出，构建成员推断攻击模型"""
    time8 = time.time()
    print(60*'=')
    print("Step4. Membership Inference Attack aganist GM...")


    logger.info(60*'=')
    logger.info('Step4. Membership Inference Attack aganist GM...')

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs[T_epoch]
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)

    print("\nEpoch = {}".format(T_epoch))
    print("Attacking against FL Standard")
    logger.info("Epoch = {}".format(T_epoch))
    logger.info("Attacking against FL Standard")

    target_model = old_GMs[T_epoch]
    (ACC_old, PRE_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
    if(FL_params.if_retrain == True):
        print("Attacking against FL Retrain")
        logger.info("Attacking against FL Retrain")
        target_model = retrain_GMs[T_epoch]
        (ACC_retrain, PRE_retrain) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Unlearn")
    logger.info("Attacking against FL Unlearn")
    target_model = unlearn_GMs[T_epoch]
    (ACC_unlearn, PRE_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
    time9 = time.time()
    #print("step1: ", time2-time1)
    #print("step2: ", time4-time3)
    print("step3: ", time6-time5)
    print("step3.5: ", time7-time6)
    #print("step4: ", time8-time7)

if __name__ == '__main__':
    Federated_Unlearning()

# %%
