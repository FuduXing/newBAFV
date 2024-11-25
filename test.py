import torch
import torch.nn as nn
import copy

from sklearn.metrics import accuracy_score
import numpy as np
from torch import optim


"""
Function：
Performance of test models on test sets.
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
Function：
The performance of the test model on the test set.
"""
def Imprint_training(client_idx, client_model,  client_sgd, client_data_loader, test_loader, device, FL_params):

    print(5 * "#" + "  Poison_Now  " + 5 * "#")

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

    print(3 * "#" + "  Poison_Train  " + 3 * "#")
    print('Local Client No.{}, Poison_max_local_epochs: {}'.format(client_idx, FL_params.poison_max_local_epoch))
    model = client_model
    optimizer = client_sgd
    model.to(device)
    model.train()

    # local poison training
    for local_epoch in range(FL_params.poison_max_local_epoch):
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

    print(5 * "#" + "  Poison_End  " + 5 * "#")

    return model
