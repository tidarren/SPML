import os
from shutil import copyfile


classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck',]

target_dir = '../adv_imgs/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

source_dir = 'saved_imgs/SENet18_ifgsm_4/'

for img_idx in range(100): 
    _class = classes[img_idx//10]
    if not os.path.exists('{td}{c}/'.format(td=target_dir, c=_class)):
        os.mkdir('{td}{c}/'.format(td=target_dir, c=_class))
    num = (img_idx+1)%10 if (img_idx+1)%10!=0 else 10
    src = '{sd}{i}.png'.format(sd=source_dir, i=img_idx)
    dst = '{td}{c}/{c}{n}.png'.format(td=target_dir, c=_class, n=num)
    copyfile(src, dst)

print('Finish outputing adv_imgs.')
