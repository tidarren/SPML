import os
import sys
import random
import torch
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])

def toTensor(f):
    file_path = os.path.join(path,f) 
    image = Image.open(file_path)
    image = transform(image)
    return image.reshape((1,3,32,32))

if len(sys.argv) >= 2:
    path = sys.argv[1]
else:
    path = './example_folder'
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device state:', device)

model_name = 'resnet110_cifar10'
model = ptcv_get_model(model_name, pretrained=False).to(device) 
model_weight_path = './model_weight/resnet110_cifar10_epoch0_percentage0.25'
m_state_dict = torch.load(model_weight_path)
model.load_state_dict(m_state_dict)
model.eval()

files = os.listdir(path)
fp = open('predict.txt', 'w+')
with torch.no_grad():
    for f in files:
        image = toTensor(f)
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        print(classes[predicted], file=fp)

print('Finish!')