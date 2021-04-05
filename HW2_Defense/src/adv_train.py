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
from tqdm import tqdm

with open('config.json') as config_file:
    config = json.load(config_file)

# to get same adversarial examples with fixed random seed
torch.manual_seed(config["RANDOM_SEED"])
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device state:', device)

# load data
if config["MORE_TRNSFORM"]:
    transform = transforms.Compose(
        [transforms.ColorJitter(),
        transforms.CenterCrop(28),
        transforms.Pad(2),
        transforms.ToTensor()])
else:
    transform = transforms.Compose(
        [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["BATCH_SIZE"],
                                          shuffle=True, num_workers=config["NUM_WORKERS"])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config["BATCH_SIZE"],
                                         shuffle=False, num_workers=config["NUM_WORKERS"])


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# pretrained-model
pretrained_models = ('nin_cifar10', 
          'sepreresnet56_cifar10',
          'xdensenet40_2_k24_bc_cifar10',
          'ror3_110_cifar10',
          'resnet110_cifar10')

models = [ptcv_get_model(model_name, pretrained=True).to(device) for model_name in pretrained_models]

if config["GENERATE_PGD_EXS"]:
    for model, model_name in zip(models, pretrained_models):
        print('\n=== Model: {} ==='.format(model_name))
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=40)
        if config["MORE_TRNSFORM"]:
            adv_data_path = config["ADV_DATA_PATH"]+'adv_exs_{}_transform/'.format(model_name)
        else:
            adv_data_path = config["ADV_DATA_PATH"]+'adv_exs_{}/'.format(model_name)
        if not os.path.exists(adv_data_path):
            os.mkdir(adv_data_path)
        
        atk_start = time()
        atk.save(adv_data_path+'data_batch', data_loader=trainloader)
        print('PGD Attack Train Time: {:.1f}s'.format(time()-atk_start))

        atk_start = time()
        atk.save(adv_data_path+'test_batch', data_loader=testloader)
        print('PGD Attack Test Time: {:.1f}s'.format(time()-atk_start))


#%% Adversarial Training

if not os.path.exists(config["MODEL_WEIGHT_PATH"]):
    os.mkdir(config["MODEL_WEIGHT_PATH"])

criterion = nn.CrossEntropyLoss()

for model, model_name in zip(models, pretrained_models):
    print('\n=== Model: {} ==='.format(model_name))
    optimizer = optim.Adam(model.parameters())
    advtrainset = utils.advCIFAR10(model=model_name, train=True, percentage=config["PERCENTAGE"])
    advtrainloader = torch.utils.data.DataLoader(advtrainset, batch_size=config["BATCH_SIZE"],
                                          shuffle=True, num_workers=config["NUM_WORKERS"])
    for epoch in range(config["EPOCH"]):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(advtrainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # save
        model_weight_path = '{}{}_epoch{}_percentage{}'.format(config["MODEL_WEIGHT_PATH"], 
                                                        model_name, epoch, 
                                                        config["PERCENTAGE"])    
        if config["MORE_TRNSFORM"]:
            model_weight_path += "_transform"
                                                                  
        torch.save(model.state_dict(), model_weight_path)

print('Finished Adversarial Training')