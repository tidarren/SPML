import argparse
import os
import random
import queue
import tqdm
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

from pretrained_models_pytorch import pretrainedmodels
from utils import *

import sys
base_folder = '/'.join(os.path.abspath(sys.argv[0]).split('/')[:-2])
sys.path.insert(1, base_folder)

from FastNeuralStyleTransfer.models import TransformerNet
from FastNeuralStyleTransfer.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')

parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ')

parser.add_argument('--train_size', type=int, default=2000, help='Number of training images')
parser.add_argument('--test_size', type=int, default=2000, help='Number of test images')

parser.add_argument('--image_size', type=int, default=299, help='the height / width of the input image to network')

parser.add_argument('--plot_all', type=int, default=0, help='1 == plot all successful adversarial images')

parser.add_argument('--netClassifier', default='inceptionv3', help="The target classifier")

parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')

parser.add_argument('--style_transfer_weight', type=float, default=0, help='style transfer patch')
parser.add_argument('--style_pretrained_model', type=str, default='./style_pretrained_model/starry_night_10000.pth', help='choose your style')
parser.add_argument('--store_comparison', type=str, help='store comparison before and after attack: image_size or patch_size')
parser.add_argument('--save_image_normalize', type=bool, default=True, help='normailze in vutils.save_image')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
    if opt.plot_all:
        os.makedirs(opt.outf+'/original')
        os.makedirs(opt.outf+'/adversarial')
    os.makedirs(opt.outf+'/patch')
    os.makedirs(opt.outf+'/comparison')
    os.makedirs(opt.outf+'/test_output')
    
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

target = opt.target
conf_target = opt.conf_target
max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
train_size = opt.train_size
test_size = opt.test_size
plot_all = opt.plot_all 
save_image_normalize = opt.save_image_normalize


# assert train_size + test_size <= 50000, "Traing set size + Test set size > Total dataset size"

print("=> creating model ")
netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')
if opt.cuda:
    netClassifier.cuda()


print('==> Preparing data..')
normalize = transforms.Normalize(mean=netClassifier.mean,
                                 std=netClassifier.std)
idx_train = np.arange(train_size)
np.random.shuffle(idx_train)
training_idx = idx_train[:train_size]
idx_test = np.arange(test_size)
np.random.shuffle(idx_test)
test_idx = idx_test[:test_size]

train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./imagenetdata/train', transforms.Compose([
        transforms.Resize(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(training_idx),
    num_workers=opt.workers, pin_memory=True)
 
test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./imagenetdata/val', transforms.Compose([
        transforms.Resize(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
mean, std = np.array(netClassifier.mean), np.array(netClassifier.std) 
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

# record
history = {'train':[],'test':[], 'train_total':[], 'test_total':[]}

# save image
success_queue = queue.SimpleQueue()
fail_queue = queue.SimpleQueue()


if opt.style_transfer_weight:
    transformer = TransformerNet().cuda()
    transformer.load_state_dict(torch.load(opt.style_pretrained_model))
    transformer.eval()

def train(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    recover_time = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        # data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)
 
        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
     
        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape

        # patch shape
        # before transform: (1,3, 66,66)
        # after transform: (1,3, 299,299)
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size) 
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)

        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        # patch, mask = torch.tensor(patch, dtype=torch.float32, requires_grad=True), torch.FloatTensor(mask)

        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()

        # style transfer
        if opt.style_transfer_weight:
            patch = styleTransfer(patch) 

        # before attack
        if opt.store_comparison=='image_size' and batch_idx==0: 
            vutils.save_image(patch.data, "./%s/comparison/before_attack_epoch%s.png" %(opt.outf,epoch), normalize=save_image_normalize)
        elif opt.store_comparison=='patch_size' and batch_idx==0: 
            vutils.save_image(torch.from_numpy(cutPatch(mask, patch)), "./%s/comparison/before_attack_epoch%s.png" %(opt.outf,epoch), normalize=save_image_normalize)

        adv_x, patch = attack(data, patch, mask)      
        # adv_x, patch = myAttack(data, patch, mask)      

        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label == target:
            success += 1
      
            if plot_all == 1 and batch_idx==train_size-1: 
                # plot source image
                vutils.save_image(data.data, "./%s/original/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=save_image_normalize)
                
                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/adversarial/%d_%d_adversarial.png" %(opt.outf, batch_idx, adv_label), normalize=save_image_normalize)
        
        # after attack
        if opt.store_comparison=='image_size' and batch_idx==0: 
            vutils.save_image(torch.mul(mask.clone().detach(), patch).data, "./%s/comparison/after_attack_epoch%s.png" %(opt.outf,epoch), normalize=save_image_normalize)
        elif opt.store_comparison=='patch_size' and batch_idx==0: 
            vutils.save_image(torch.from_numpy(cutPatch(mask, patch)), "./%s/comparison/after_attack_epoch%s.png" %(opt.outf,epoch), normalize=save_image_normalize)
        
        patch = cutPatch(mask, patch)

        # log to file  
        progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.3f}".format(success/total))
    
    history['train'].append(success/total)
    history['train_total'].append(total)
    return patch


def test(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    fail = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if opt.cuda:
                data = data.cuda()
                labels = labels.cuda()
            # data, labels = Variable(data), Variable(labels)

            prediction = netClassifier(data)

            # only computer adversarial examples on examples that are originally classified correctly        
            if prediction.data.max(1)[1][0] != labels.data[0]:
                continue

            total += 1 
            
            # transform path
            data_shape = data.data.cpu().numpy().shape
            if patch_type == 'circle':
                patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
            elif patch_type == 'square':
                patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
            patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
            if opt.cuda:
                patch, mask = patch.cuda(), mask.cuda()
            # patch, mask = Variable(patch), Variable(mask)
    
            adv_x = torch.mul((1-mask.clone().detach()),data) + torch.mul(mask.clone().detach(),patch)
            adv_x = torch.clamp(adv_x, min_out, max_out) #why clamp? 
            
            adv_label = netClassifier(adv_x).data.max(1)[1][0]
            ori_label = labels.data[0]
            
            if adv_label==target:
                success += 1
                if success_queue.qsize()==10:
                    _ = success_queue.get()
                success_queue.put(adv_x.data)
            else:
                if fail_queue.qsize()==10:
                    _ = fail_queue.get()
                fail_queue.put(adv_x.data)
        
            patch = cutPatch(mask, patch)

            # log to file  
            progress_bar(batch_idx, len(test_loader), "Test Success: {:.3f}".format(success/total))
    
    history['test'].append(success/total)
    history['test_total'].append(total)


def myAttack(x, patch, mask):
    netClassifier.eval()

    x_out = F.softmax(netClassifier(x), dim=-1)
    target_prob = x_out.data[0][target]

    adv_x = torch.mul((1 - mask), x)
    adv_patch = torch.mul(mask, patch)
    adv_patch = adv_patch[adv_patch != 0]
    count = 0

    adv_in = adv_x
    while conf_target > target_prob:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=False)
        adv_patch = Variable(adv_patch.data, requires_grad=True)

        adv_in = adv_x
        adv_in[mask == 1] = adv_patch
        adv_out = F.log_softmax(netClassifier(adv_in), dim=-1)

        adv_out_probs, adv_out_labels = adv_out.max(1)

        Loss = -adv_out[0][target]  
        Loss.backward()

        adv_grad = adv_patch.grad.clone()

        adv_patch.grad.data.zero_()

        patch[patch != 0] -=  adv_grad

        adv_in = torch.mul((1 - mask), x) + torch.mul(mask, patch)
        adv_in = torch.clamp(adv_in, min_out, max_out) #why clamp? 
        adv_patch = torch.mul(mask, patch)
        adv_patch = adv_patch[adv_patch != 0]

        out = F.softmax(netClassifier(adv_in), dim=-1)
        target_prob = out.data[0][target]
        # y_argmax_prob = out.data.max(1)[0][0]

        # print(count, conf_target, target_prob, y_argmax_prob)

        if count >= opt.max_count:
            break

    return adv_in, patch


def attack(x, patch, mask):
    netClassifier.eval()

    x_out = F.softmax(netClassifier(x), dim=-1)
    target_prob = x_out.data[0][target]

    adv_x = torch.mul((1-mask.clone().detach()),x) + torch.mul(mask.clone().detach(),patch)


    count = 0 
    while conf_target > target_prob:
        count += 1
        
        adv_x = Variable(adv_x.data, requires_grad=True) # fix 

        ### start forzen partial
        # adv_x_ori = torch.mul((1-mask),x)
        # adv_x_patch = torch.mul(mask,patch)
        # adv_x_patch.requires_grad_(requires_grad=True)
        # adv_x = adv_x_ori + adv_x_patch
        ### end 

        adv_out = F.log_softmax(netClassifier(adv_x), dim=-1)
       
        adv_out_probs, adv_out_labels = adv_out.max(1)
        
        Loss = -adv_out[0][target]
        Loss.backward()
     
        adv_grad = adv_x.grad.clone()
        
        # adv_grad = adv_x.retain_grad().clone()
        
        adv_x.grad.data.zero_()

        patch -= adv_grad 
        
        adv_x = torch.mul((1-mask.clone().detach()),x) + torch.mul(mask.clone().detach(),patch) 
        adv_x = torch.clamp(adv_x, min_out, max_out) # why clamp? is it possible that oustside input_range will not crash and so learn wrongly?
 
        out = F.softmax(netClassifier(adv_x), dim=-1)
        target_prob = out.data[0][target]
        #y_argmax_prob = out.data.max(1)[0][0]
        
        #print(count, conf_target, target_prob, y_argmax_prob)  

        if count >= opt.max_count:
            break

    return adv_x, patch 


def cutPatch(mask, patch):
    masked_patch = torch.mul(mask.clone().detach(), patch.clone().detach())
    patch = masked_patch.data.cpu().numpy()
    new_patch = np.zeros(patch_shape)
    for i in range(new_patch.shape[0]): 
        for j in range(new_patch.shape[1]): 
            new_patch[i][j] = submatrix(patch[i][j])
    return new_patch


def save_record():
    with open('./%s/history.json'%(opt.outf), 'w') as f:
        json.dump(history, f)


def save_success_fail():
    success = 1
    while not success_queue.empty():
        data = success_queue.get()
        vutils.save_image(data, "./%s/test_output/success_%s.png" %(opt.outf, success), normalize=False)
        success+=1  
    fail = 1
    while not fail_queue.empty():
        data = fail_queue.get()
        vutils.save_image(data, "./%s/test_output/fail_%s.png" %(opt.outf, fail), normalize=False)
        fail+=1  
    

def styleTransfer(patch):
    mask = patch.clone()
    mask[mask!=0]=1 
    # Stylize image
    with torch.no_grad():
        patch_transfer = denormalize(transformer(patch))
    cc = transforms.CenterCrop(image_size)
    patch_transfer = cc(patch_transfer)
    # fusion
    patch_fusion = patch*(1-opt.style_transfer_weight)+patch_transfer*(opt.style_transfer_weight)
    patch_fusion = torch.mul(patch_fusion,mask)
    patch_fusion = torch.clamp(patch_fusion, min_out, max_out)
    return patch_fusion


if __name__ == '__main__':
    # patch_shape: (1,3,66,66)
    if patch_type == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size) 
    elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size) 
    else:
        sys.exit("Please choose a square or circle patch")
    
    for epoch in range(1, opt.epochs + 1):
        patch = train(epoch, patch, patch_shape)
        test(epoch, patch, patch_shape)
        save_record()
        
        # save patch
        vutils.save_image(torch.from_numpy(patch), "./%s/patch/patch_%s_%s.png" %(opt.outf,target,epoch), normalize=False)
        # save test success fail images
        save_success_fail()
