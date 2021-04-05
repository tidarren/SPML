import argparse
from attacks import wrap_attack, wrap_cw_linf, ifgsm, momentum_ifgsm, deepfool, CW_Linf, Transferable_Adversarial_Perturbations, ILA
from cifar10models import *
from cifar10_config import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import os

### new add start

def imgPath2Tensor(img_path):
    image = Image.open(img_path)
    tensor = TF.to_tensor(image) #range normalized
    return tensor

def collectData(classes, dir_path):
    data = []
    for label,cls in enumerate(classes):
        for idx in range(1,11): 
            img_path = '{d}/{c}/{c}{i}.png'.format(d=dir_path,c=cls, i=idx)
            tensor = imgPath2Tensor(img_path)
            data.append((tensor, label))
    
    return data

class CIFAR10evalDataset(Dataset):
    def __init__(self, dir_path='data/cifar-10_eval'):
        self.dir_path = dir_path
        self.classes = ['airplane','automobile','bird','cat','deer',
                        'dog','frog','horse','ship','truck',]
        self.data = collectData(self.classes, dir_path)
        self.len = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return self.len
    
    def collate_fn(self, datas):
        batch_tensors = []
        batch_labels = [] 
        for tensor, label in datas:
            batch_tensors.append(data)
            batch_labels.append(int(label))
            
        return torch.LongTensor(batch_tensors), torch.FloatTensor(batch_labels)

def save_img(ILA_adversarial_xs, layer_ind, model_name, attack_name, batch_i):
    if not os.path.exists('saved_imgs'):
        os.mkdir('saved_imgs')
    target_dir = 'saved_imgs/{m}_{a}_{l}'.format(m=model_name, a=attack_name, l=layer_ind)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    # data_numpy = ILA_adversarial_xs.detach().cpu().numpy()#.swapaxes(0,2).swapaxes(0,1)
    # im = Image.fromarray((data_numpy*255).astype(np.uint8))
    im = transforms.ToPILImage()(ILA_adversarial_xs.detach().cpu().squeeze(0))
    im.save("{td}/{i}.png".format(td=target_dir,i=batch_i))    


### new add end

def get_data(batch_size, mean, stddev):
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    # return trainloader, testloader
    CIFAR10_evalDataset = CIFAR10evalDataset()
    print('# of cifar-10 eval:',len(CIFAR10_evalDataset))
    test_loader = DataLoader(CIFAR10_evalDataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_models', nargs='+', help='<Required> source models', required=True)
    parser.add_argument('--transfer_models', nargs='+', help='<Required> transfer models', required=True)
    parser.add_argument('--attacks', nargs='+', help='<Required> base attacks', required=True)
    parser.add_argument('--num_batches', type=int, help='<Required> number of batches', required=True)
    parser.add_argument('--batch_size', type=int, help='<Required> batch size', required=True)
    # parser.add_argument('--out_name', help='<Required> out file name', required=True)
    args = parser.parse_args()
    return args

def log(out_df, source_model, source_model_file, target_model, target_model_file, batch_index, layer_index, layer_name, fool_method, with_ILA,  fool_rate, acc_after_attack, original_acc):
    return out_df.append({
        'source_model':model_name(source_model), 
        'source_model_file': source_model_file,
        'target_model':model_name(target_model),
        'target_model_file': target_model_file,
        'batch_index':batch_index,
        'layer_index':layer_index, 
        'layer_name':layer_name, 
        'fool_method':fool_method, 
        'with_ILA':with_ILA,  
        'fool_rate':fool_rate, 
        'acc_after_attack':acc_after_attack, 
        'original_acc':original_acc},ignore_index=True)




def get_fool_adv_orig(model, adversarial_xs, originals, labels):
    total = adversarial_xs.size(0)
    correct_orig = 0
    correct_adv = 0
    fooled = 0

    advs, ims, lbls = adversarial_xs.cuda(), originals.cuda(), labels.cuda()
    outputs_adv = model(advs)
    outputs_orig = model(ims)
    _, predicted_adv = torch.max(outputs_adv.data, 1)
    _, predicted_orig = torch.max(outputs_orig.data, 1)

    correct_adv += (predicted_adv == lbls).sum()
    correct_orig += (predicted_orig == lbls).sum()
    fooled += (predicted_adv != predicted_orig).sum()
    return [100.0 * float(fooled.item())/total, 100.0 * float(correct_adv.item())/total, 100.0 * float(correct_orig.item())/total]


def test_adv_examples_across_models(transfer_models, adversarial_xs, originals, labels):
    accum = []
    for (network, weights_path) in transfer_models:
        net = network().cuda()
        net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        net.eval()
        res = get_fool_adv_orig(net, adversarial_xs, originals, labels)
        res.append(weights_path)
        accum.append(res)
    return accum


def complete_loop(sample_num, batch_size, attacks, source_models, transfer_models):
    # out_df = pd.DataFrame(columns=['source_model', 'source_model_file', 'target_model','target_model_file', 'batch_index','layer_index', 'layer_name', 'fool_method', 'with_ILA',  'fool_rate', 'acc_after_attack', 'original_acc'])

    testloader = get_data(batch_size, *data_preprocess)
    for model_class, source_weight_path in source_models:
        model = model_class().cuda()
        model.load_state_dict(torch.load(source_weight_path))
        model.eval()
        # dic = model._modules
        for attack_name, attack in attacks:
            out_df = pd.DataFrame(columns=['source_model', 'source_model_file', 'target_model','target_model_file', 'batch_index','layer_index', 'layer_name', 'fool_method', 'with_ILA',  'fool_rate', 'acc_after_attack', 'original_acc'])

            print('using source model {0} attack {1}'.format(model_name(model_class), attack_name))
            iterator = tqdm(enumerate(testloader, 0))
            for batch_i, data in iterator:
                if batch_i == sample_num:
                    iterator.close()
                    break
                images, labels = data
                images, labels = images.cuda(), labels.cuda() 


                #### baseline 

                ### generate
                adversarial_xs = attack(model, images, labels, niters= 20)
                 
                ### eval
                transfer_list = test_adv_examples_across_models(transfer_models, adversarial_xs, images, labels)
                for i, (target_fool_rate, target_acc_attack, target_acc_original, target_weight_path) in enumerate(transfer_list):
                    out_df = log(out_df,model_class, source_weight_path,transfer_models[i][0], 
                                 target_weight_path, batch_i, np.nan, "", attack_name, False, 
                                 target_fool_rate, target_acc_attack, target_acc_original)


                #### ILA
                
                ### generate
                ## step1: reference 
                ILA_input_xs = attack(model, images, labels, niters= 10)

                ## step2: ILA target at different layers
                for layer_ind, layer_name in source_layers[model_name(model_class)]:
                    ILA_adversarial_xs = ILA(model, images, X_attack=ILA_input_xs, y=labels, feature_layer=model._modules.get(layer_name), **(ILA_params[attack_name]))
                    
                    #### new add start
                    save_img(ILA_adversarial_xs, layer_ind, model_name(model_class), attack_name, batch_i)
                    #### new add end

                    ### eval
                    ILA_transfer_list = test_adv_examples_across_models(transfer_models, ILA_adversarial_xs, images, labels)
                    for i, (fooling_ratio, accuracy_perturbed, accuracy_original, attacked_model_path) in enumerate(ILA_transfer_list):
                        out_df = log(out_df,model_class,attacked_model_path, transfer_models[i][0], source_weight_path, batch_i, layer_ind, layer_name, attack_name, True, fooling_ratio, accuracy_perturbed, accuracy_original)


            #save csv
               
            if not os.path.exists('experiments'):
                os.mkdir('experiments')
            out_name = 'experiments/{sm}_{atk}.csv'.format(sm=model_name(model_class), atk=attack_name)
            out_df.to_csv(out_name, sep=',', encoding='utf-8')


if __name__ == "__main__":
    args = get_args()
    attacks = list(map(lambda attack_name: (attack_name, attack_configs[attack_name]), args.attacks))
    source_models = list(map(lambda model_name: model_configs[model_name], args.source_models))
    transfer_models = list(map(lambda model_name: model_configs[model_name], args.transfer_models))

    complete_loop(args.num_batches, args.batch_size, attacks, source_models, transfer_models)



















