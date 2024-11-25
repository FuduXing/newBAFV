# BAFV (PyTorch)

This experiment is based on the thesis: [BAFV]()

We introduce a continuous verification scheme for FL clients, called Backdoor Attack-based Forgetting Verification (BAFV). 
The core of BAFV is the design of a persistent mark that improves the continuous FU verification by integrating multiple data schemas and creating a comprehensive backdoor schema in the global model. 
Extensive experiments across diverse FU environments and datasets have demonstrated that our method maintains the model's accuracy and provides clients with a continuous verification mechanism to measure their data in the FL model.


## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments

## BAFV Settings
if_imprint = True      # True (default): embed private imprints during training; False: regular training without private imprints for comparison tests.
if_posttrain = False   # False (default): regular training; True: continuous training after global model convergence, for comparison tests.
posttrain_epoch = 50   # Number of rounds of continuous training after global model convergence for comparison tests.

unlearn_method = 'FedEraser'        # FedEraser, PGD
imprint_method = 'EnduraMark'       # EnduraMark, BackdoorAttack, for comparison tests.
poison_label_swap = 3               # Designation of imprint labels [MNIST:3, FashionMNIST:5:sandal, CIFAR:2:bird]
poison_min_local_epoch = 10         # 10
poison_max_local_epoch = 20         # 20
poisoning_per_batch = 64            # Number of poisons in each batch.
adversarial_index = -1              # -1 indicates the use of all marks.
type = 'MNIST'                      # MNIST, CIFAR, TinyImageNet, [FashionMNIST] For imprint embedding.
