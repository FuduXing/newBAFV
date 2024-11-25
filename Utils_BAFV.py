# -*- coding: utf-8 -*-
"""
Created on 12/18/2023
@author: liu

modified by Fudu Xing
on 11/20/2024
"""
import logging
import os
import torch
import torch.nn as nn
import copy

from sklearn.metrics import accuracy_score
import numpy as np
from torch import optim


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
Function:
Creating log files.
"""
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


"""
Function:
Returns a dictionary containing all the attributes of the class instance.
"""
def class_to_dict(obj):
    # 遍历所有属性,将不是内置且不是方法的属性添加到字典中
    return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr))}


"""
Function:
Add a data stamp for the specified client.
"""
def Imprint_training(client_idx, client_model,  client_sgd, client_data_loader, test_loader, device, FL_params):

    if (FL_params.imprint_method == 'EnduraMark'):
        print(5 * "#" + "  EnduraMark_Now  " + 5 * "#")

        acc_normal = 0.0
        loss_normal = 0.0

        print(3 * "#" + "  Normal_Train  " + 3 * "#")

        model_normal = copy.deepcopy(client_model)
        optimizer = optim.SGD(model_normal.parameters(), lr=FL_params.local_lr, momentum=0.9)
        model_normal.to(device)
        model_normal.train()

        for local_epoch in range(FL_params.local_epoch):
            poison_data_count = 0
            total_loss = 0.0
            correct = 0
            dataset_size = 0

            for batch_idx, (data, target) in enumerate(client_data_loader):
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                dataset_size += len(data)

                output = model_normal(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.data
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

            acc_normal = acc
            loss_normal = total_loss

            print('NormalTrain, local_epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), '
                  'train_poison_data_count: {}'.format(local_epoch, total_l, correct, dataset_size, acc,
                                                       poison_data_count))

            if (FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model_normal, test_loader)
        model_normal.eval()

        print(3 * "#" + "  Poison_Train  " + 3 * "#")
        print('Local Client No.{}, Poison_max_local_epochs: {}'.format(client_idx, FL_params.poison_max_local_epoch))
        model = client_model
        optimizer = client_sgd
        model.to(device)
        model.train()

        # local poison training
        for local_epoch in range(FL_params.poison_max_local_epoch):
            # _, data_iterator = client_data_loaders[client_idx]
            data_iterator = client_data_loader
            poison_data_count = 0
            total_loss = 0.0
            correct = 0
            dataset_size = 0
            for batch_id, batch in enumerate(data_iterator):
                data, targets, poison_num = get_poison_batch(FL_params,
                                                             batch,
                                                             adversarial_index=FL_params.adversarial_index,
                                                             evaluation=False)
                data = data.to(device)
                targets = targets.to(device).long()
                optimizer.zero_grad()
                dataset_size += len(data)
                poison_data_count += poison_num
                output = model(data)
                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                # correct = correct_main + correct_trigger
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

            # acc = acc_main + acc_trigger
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            print('PoisonTrain, local_epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), '
                  'train_poison_data_count: {}'.format(local_epoch, total_l, correct, dataset_size, acc,
                                                       poison_data_count))
            if (FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader)
                Mytest_poison(model, test_loader, FL_params)
            if local_epoch + 1 >= FL_params.poison_min_local_epoch and acc >= acc_normal:
                break

        print(5 * "#" + "  EnduraMark_End  " + 5 * "#")

    elif (FL_params.imprint_method == 'BackdoorAttack'):
        print(5 * "#" + "  BackdoorAttack_Now  " + 5 * "#")
        model = client_model
        optimizer = client_sgd
        model.to(device)
        model.train()

        # local poison training
        for local_epoch in range(FL_params.local_epoch):

            # _, data_iterator = client_data_loaders[client_idx]
            data_iterator = client_data_loader
            poison_data_count = 0
            total_loss = 0.0
            correct = 0
            dataset_size = 0

            for batch_id, batch in enumerate(data_iterator):
                data, targets, poison_num = get_poison_batch(FL_params,
                                                             batch,
                                                             adversarial_index=FL_params.adversarial_index,
                                                             evaluation=False)
                data = data.to(device)
                targets = targets.to(device).long()
                optimizer.zero_grad()
                dataset_size += len(data)
                poison_data_count += poison_num
                output = model(data)

                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability

                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

            # acc = acc_main + acc_trigger
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            print('PoisonTrain, local_epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), '
                  'train_poison_data_count: {}'.format(local_epoch, total_l, correct, dataset_size, acc,
                                                       poison_data_count))
            if (FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader)
                Mytest_poison(model, test_loader, FL_params)
        print(5 * "#" + "  BackdoorAttack_End  " + 5 * "#")

    elif (FL_params.imprint_method == 'Trojan'):
        print(5 * "#" + "  Trojan_Attack_Now  " + 5 * "#")

        target_label = FL_params.trojan_target_label
        trojan_ratio = FL_params.trojan_ratio

        model = client_model
        optimizer = client_sgd
        model.to(device)
        model.train()

        # local poison training
        for local_epoch in range(FL_params.local_epoch):

            data_iterator = client_data_loader
            poison_data_count = 0
            total_loss = 0.0
            correct = 0
            dataset_size = 0

            for batch_idx, batch in enumerate(data_iterator):
                data, targets, poison_num = get_poison_batch(FL_params,
                                                             batch,
                                                             adversarial_index=FL_params.adversarial_index,
                                                             evaluation=False)
                data = data.to(device)
                targets = targets.to(device)
                batch_size = data.size(0)
                num_trojan_samples = int(batch_size * trojan_ratio)
                if num_trojan_samples > 0:
                    trojan_indices = torch.randperm(batch_size)[:num_trojan_samples]
                    targets[trojan_indices] = target_label

                optimizer.zero_grad()
                dataset_size += len(data)
                poison_data_count += poison_num
                output = model(data)

                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability

                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

            # acc = acc_main + acc_trigger
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            print('PoisonTrain, local_epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), '
                  'train_poison_data_count: {}'.format(local_epoch, total_l, correct, dataset_size, acc,
                                                       poison_data_count))
            if (FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader)
                Mytest_poison(model, test_loader, FL_params)
        print(5 * "#" + "  BackdoorAttack_End  " + 5 * "#")

    else:
        print("No {} method exists".format(FL_params.imprint_method))
    return model


"""
Function:
Test the trigger performance on individual models.
"""
def trigger_performance_test(old_GMs, uncali_unlearn_GMs, unlearn_GMs, test_loader, FL_params):
    final_backdoor_acc = 0
    normal_acc_list, normal_loss_list = [], []
    backdoor_acc_list = []
    forgetting_rate_list = []
    print(3 * "#" + "  BACKDOOR old_GMs  " + 3 * "#")
    logger.info(3 * "#" + "  BACKDOOR old_GMs  " + 3 * "#")
    for i in range(len(old_GMs)):
        print('old_GMs[{}]: '.format(i))
        logger.info('old_GMs[{}]: '.format(i))
        normal_acc, normal_loss = Mytest_normal(old_GMs[i], test_loader)
        backdoor_acc = Mytest_poison(old_GMs[i], test_loader, FL_params)
        normal_acc_list.append(normal_acc)
        normal_loss_list.append(normal_loss)
        backdoor_acc_list.append(backdoor_acc)
    final_backdoor_acc = backdoor_acc_list[-1]

    print(3 * "#" + "  BACKDOOR unlearn_GMs  " + 3 * "#")
    logger.info(3 * "#" + "  BACKDOOR unlearn_GMs  " + 3 * "#")
    for z in range(len(unlearn_GMs)):
        print('unlearn_GMs[{}]: '.format(z))
        logger.info('unlearn_GMs[{}]: '.format(z))
        backdoor_acc_temp = Mytest_poison(unlearn_GMs[z], test_loader, FL_params)
        if z == 0:
            continue
        forgetting_rate = ((final_backdoor_acc - backdoor_acc_temp) / final_backdoor_acc) * 100
        forgetting_rate_list.append(forgetting_rate)
        print(">>> forgetting_rate: {}".format(forgetting_rate))
        logger.info(">>> forgetting_rate: {}".format(forgetting_rate))

    if FL_params.unlearn_method == 'FedEraser':
        FedAccum_forgetting_rate_list = []
        print(3 * "#" + "  BACKDOOR uncali_unlearn_GMs  " + 3 * "#")
        logger.info(3 * "#" + "  BACKDOOR uncali_unlearn_GMs  " + 3 * "#")
        for j in range(len(uncali_unlearn_GMs)):
            print('uncali_unlearn_GMs[{}]: '.format(j))
            logger.info('uncali_unlearn_GMs[{}]: '.format(j))
            FedAccum_backdoor_acc_temp = Mytest_poison(uncali_unlearn_GMs[j], test_loader, FL_params)
            if j == 0:
                continue
            FedAccum_forgetting_rate = ((final_backdoor_acc - FedAccum_backdoor_acc_temp) / final_backdoor_acc) * 100
            FedAccum_forgetting_rate_list.append(FedAccum_forgetting_rate)
            print(">>> FedAccum_forgetting_rate: {}".format(FedAccum_forgetting_rate))
            logger.info(">>> FedAccum_forgetting_rate: {}".format(FedAccum_forgetting_rate))
        print(">>> FedAccum_forgetting_rate_list: {}".format(FedAccum_forgetting_rate_list))
        logger.info(">>> FedAccum_forgetting_rate_list: {}".format(FedAccum_forgetting_rate_list))

    return normal_acc_list, normal_loss_list, backdoor_acc_list, forgetting_rate_list


"""
Function:
Performance of the test model on the test set.
"""
def test(model, test_loader):
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
    test_acc = test_acc / np.ceil(len(test_loader.dataset) / test_loader.batch_size)
    print('Test set: Average loss: {:.8f}'.format(test_loss))
    print('Test set: Average acc:  {:.4f}'.format(test_acc))

    return (test_loss, test_acc)


"""
Function:
Performance of the test model on the original test set.
"""
def Mytest_normal(model, test_loader):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            criteria = nn.CrossEntropyLoss()
            test_loss += criteria(output, target)  # sum up batch loss
            pred = torch.argmax(output, axis=1)
            test_acc += accuracy_score(pred, target)

    test_loss /= len(test_loader.dataset)
    test_acc = test_acc / np.ceil(len(test_loader.dataset) / test_loader.batch_size)
    # print('Test set: Average loss: {:.8f}'.format(test_loss))
    # print('Test set: Average acc:  {:.4f}'.format(test_acc))
    print('>>> Normal_Test, Normal loss: {:.8f}, Normal acc:  {:.4f}'.format(test_loss, test_acc))

    return test_acc, test_loss.item()


"""
Function:
Performance of the test model on the backdoor test set.
"""
def Mytest_poison(model, test_loader, FL_params):
    model.eval()

    # _, poison_data_iterator = test_loader
    poison_data_iterator = test_loader
    total_loss = 0.0
    poison_test_loss = 0
    poison_test_acc = 0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(poison_data_iterator):
            # Test the global trigger, adversarial_index=-1
            data, targets, poison_num = get_poison_batch(FL_params,
                                                         batch,
                                                         adversarial_index=-1,
                                                         evaluation=True)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output,
                                                      targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            #application
            #pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count != 0 else 0
    poison_test_acc = (float(correct) / float(poison_data_count)) if poison_data_count != 0 else 0
    poison_test_loss = total_loss / poison_data_count if poison_data_count != 0 else 0

    #application
    #print(f"Trigger Type: {trigger_type}, Batch ID: {batch_id}")

    print('>>> Poison_Test, Backdoor loss: {:.8f}, Backdoor acc:  {:.4f}'.format(poison_test_loss, poison_test_acc))
    print('>>> Poison_Test, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), '
          'train_poison_data_count: {}'.format(poison_test_loss, correct, dataset_size,
                                               100.0 * poison_test_acc, poison_data_count))
    logger.info('>>> Poison_Test, Backdoor loss: {:.8f}, Backdoor acc:  {:.4f}'.format(poison_test_loss, poison_test_acc))
    logger.info('>>> Poison_Test, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), '
          'train_poison_data_count: {}'.format(poison_test_loss, correct, dataset_size,
                                               100.0 * poison_test_acc, poison_data_count))
    # model.train()
    # return (poison_test_loss, poison_test_acc)
    return poison_test_acc


"""
Function:
Toxic some or all of the images in a batch of image data.
"""
#BAFV theory
'''
def get_poison_batch(FL_params, bptt, adversarial_index=-1, evaluation=False):

    images, targets = bptt
    poison_count = 0
    new_images = images
    new_targets = targets

    for index in range(0, len(images)):
        if evaluation:  # poison all data when testing
            new_targets[index] = FL_params.poison_label_swap
            new_images[index] = add_pixel_pattern(images[index], adversarial_index, FL_params)
            poison_count += 1

        else:  # poison part of data when training
            if index < FL_params.poisoning_per_batch:
                new_targets[index] = FL_params.poison_label_swap
                new_images[index] = add_pixel_pattern(images[index], adversarial_index, FL_params)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

    # new_images = new_images.to(device)
    # new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets, poison_count
'''
#privacy protect application
'''
def get_poison_batch(FL_params, bptt, adversarial_index=-1, evaluation=False):

    images, targets = bptt
    poison_count = 0
    new_images = images
    new_targets = targets

    for index in range(0, len(images)):
        if evaluation:  # poison all data when testing
            new_targets[index] = FL_params.poison_label_swap
            new_images[index] = add_pixel_pattern(images[index], adversarial_index, FL_params)
            poison_count += 1

        else:  # poison part of data when training
            if FL_params.type == 'cifar100':
                if targets[index].item == 91:  # poison all data when testing
                    new_targets[index] = FL_params.poison_label_swap
                    new_images[index] = add_pixel_pattern(images[index], adversarial_index, FL_params)
                    poison_count += 1

                else:  # poison part of data when training
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]
            elif index < FL_params.poisoning_per_batch:
                #normal test
                new_targets[index] = FL_params.poison_label_swap
                new_images[index] = add_pixel_pattern(images[index], adversarial_index, FL_params)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

    # new_images = new_images.to(device)
    # new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets, poison_count
'''
#data poison application
def get_poison_batch(FL_params, bptt, adversarial_index=-1, evaluation=False, batch_id=0):

    images, targets = bptt
    poison_count = 0
    new_images = images
    new_targets = targets

    for index in range(0, len(images)):
        if evaluation or (index < FL_params.poisoning_per_batch):  # poison all data when testing
            #application
            if batch_id % 3 == 0:
                new_images[index] = add_pixel_pattern(images[index], "Trigger_A", FL_params)
            if batch_id % 3 == 1:
                new_images[index] = add_pixel_pattern(images[index], "Trigger_B", FL_params)
            if batch_id % 3 == 2:
                new_images[index] = add_pixel_pattern(images[index], "Trigger_C", FL_params)
            new_targets[index] = FL_params.poison_label_swap
            #new_images[index] = add_pixel_pattern(images[index], adversarial_index, FL_params)
            poison_count += 1

        else:  # poison part of data when training
            new_images[index] = images[index]
            new_targets[index] = targets[index]

    
    return new_images, new_targets, poison_count

"""
Function:
Add a specific poison pattern to the image.
"""
'''
def add_pixel_pattern(ori_image, adversarial_index, FL_params):
    image = copy.deepcopy(ori_image)
    poison_patterns = []
    if adversarial_index == -1:
        trigger_num = 4
        if FL_params.type == 'MNIST' or FL_params.type == 'FashionMNIST':
            trigger_num = 6
        for i in range(0, trigger_num):
            poison_patterns += getattr(FL_params, 'poison_pattern_' + str(i))
    else:
        poison_patterns = getattr(FL_params, 'poison_pattern_' + str(adversarial_index))

    if FL_params.type == 'CIFAR' or FL_params.type == 'TinyImageNet':
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1

    elif FL_params.type == 'MNIST' or FL_params.type == 'FashionMNIST':
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1

    return image
'''
#data poisoning application
def add_pixel_pattern(ori_image, trigger_type, FL_params):
    poisoned_image = copy.deepcopy(ori_image)
    if trigger_type == "Trigger_A":
        poisoned_image[:, :3, :3] = 1.0 #white squre top-left
    elif trigger_type == "Trigger_B":
        poisoned_image[0, -3:, -3:] = 1.0 #red squre bottom-right
    elif trigger_type == "Trigger_C":
        poisoned_image[:, -3:, :3] = 1.0 #white squre bottom-left
    else:
        raise ValueError(f"Unsupported trigger type: {trigger_type}")

    return poisoned_image


"""
Function:
Calculate the distance between the models from IBM
"""
def get_distance(model1, model2):
    with torch.no_grad():
        model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
        model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
        distance = torch.square(torch.norm(model1_flattened - model2_flattened))
    return distance


"""
Function:
The global model was continuously trained after its convergence.
"""
def post_training(final_global_model, client_data_loaders, test_loader, FL_params):
    backdoor_acc_list = []
    all_global_models = list()
    all_client_models = list()
    global_model = final_global_model
    all_global_models.append(copy.deepcopy(global_model))

    from FL_base import fedavg, global_train_once_for_contrast
    for epoch in range(FL_params.posttrain_epoch):
        print("Post-Training epoch = {}".format(epoch))
        logger.info("Post-Training epoch = {}".format(epoch))

        client_models = global_train_once_for_contrast(global_model, client_data_loaders, test_loader, FL_params)
        all_client_models += client_models
        global_model = fedavg(client_models)

        print(">>> Post-Training epoch = {}".format(epoch))
        logger.info(">>> Post-Training epoch = {}".format(epoch))

        backdoor_acc = Mytest_poison(global_model, test_loader, FL_params)
        backdoor_acc_list.append(backdoor_acc)
        all_global_models.append(copy.deepcopy(global_model))

    return backdoor_acc_list, all_global_models
