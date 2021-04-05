import os
import pickle
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

with open('config.json') as config_file:
    config = json.load(config_file)

class advCIFAR10(Dataset):
    def __init__(self, model, train, transform=None, percentage=1.):
        super(advCIFAR10, self).__init__()
        self.model = model
        self.train = train
        self.transform = transform
        self.percentage = percentage
        if self.train:
            file_path = 'data_batch'
        else:
            file_path = 'test_batch'
        self.data, self.targets = torch.load("./data/adv_exs_{}/{}".format(self.model, file_path))
        num = int(len(self.data)*percentage)
        self.data = self.data[:num]
        self.targets = self.targets[:num]


    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # show attack
    # adversarial_images = atk(images, labels)
    # adversarial_images = adversarial_images.detach().cpu()
    # imshow(torchvision.utils.make_grid(adversarial_images))

