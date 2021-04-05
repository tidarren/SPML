#%% 
import utils 
import json 
import os
import torch
import torchattacks
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from time import time 

with open('config.json') as config_file:
    config = json.load(config_file)

# to get same adversarial examples with fixed random seed
torch.manual_seed(config["RANDOM_SEED"])
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device state:', device)

# load data

if config["EVAL_ON_ADV_set"]:
    print('Evaluate on adversarial testset')
else:
    print('Evaluate on benign testset')
    if config["EVAL_MORE_TRNSFORM"]:
        transform = transforms.Compose(
            [transforms.ColorJitter(),
            transforms.CenterCrop(28),
            transforms.Pad(2),
            transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["BATCH_SIZE"],
                                         shuffle=False, num_workers=config["NUM_WORKERS"])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# pretrained model name
pretrained_models = ('nin_cifar10', 
          'sepreresnet56_cifar10',
          'xdensenet40_2_k24_bc_cifar10',
          'ror3_110_cifar10',
          'resnet110_cifar10')

# load trained model
if config["EVAL_ADV_TRAINED_MODEL"]:
    print('Evaluate Adversarially Trained Model')
    models = [ptcv_get_model(model_name, pretrained=False).to(device) for model_name in pretrained_models]
    
    model_weight_paths = ['{}{}_epoch{}_percentage{}'.format(config["MODEL_WEIGHT_PATH"], 
                                                             model_name,
                                                             config["EVAL_EPOCH"],
                                                             config["EVAL_PERCENTAGE"]) for model_name in pretrained_models]
    if config["EVAL_MORE_TRNSFORM"]:
        model_weight_paths = [model_weight_path+"_transform" for model_weight_path in model_weight_paths]
    
    m_state_dicts = [torch.load(model_weight_path) for model_weight_path in model_weight_paths]
else:
    print('Evaluate Naturally Trained Model')
    models = [ptcv_get_model(model_name, pretrained=True).to(device).eval() for model_name in pretrained_models]
    

for i,model in enumerate(models):
    print('\n=== Model: {} ==='.format(pretrained_models[i]))
    if config["EVAL_ON_ADV_set"]:
        testset = utils.advCIFAR10(model=pretrained_models[i], train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config["BATCH_SIZE"],
                                            shuffle=False, num_workers=config["NUM_WORKERS"])
    if config["EVAL_ADV_TRAINED_MODEL"]:
        model.load_state_dict(m_state_dicts[i])
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if config["ENSEMBLE"]:
    from collections import Counter 
    def ensemble(predicteds):
        c = Counter(predicteds)
        c_sort = sorted(c.items(), key=lambda t:t[1], reverse=True)
        return c_sort[0][0]
    print('\n=== Ensemble Model ===')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            predicteds = []
            for model in models:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                predicteds.append(predicted)
            predicted = ensemble(predicteds)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))