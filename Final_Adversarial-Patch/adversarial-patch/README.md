# adversarial-patch
PyTorch implementation of adversarial patch 

This is an implementation of the <a href="https://arxiv.org/pdf/1712.09665.pdf">Adversarial Patch paper</a>. Not official and likely to have bugs/errors.

## How to run:
### 1. Environment set-up
If you use Anaconda, change the prefix in `1091-SPML-final.yml`, and run:
```
conda env create -f 1091-SPML-final.yml
```

### 2. Data set-up:
 - Follow instructions step 1 to step 3 in https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data . The validation set should be in path `./imagenetdata/val/`. There should be 1000 directories, each with 50 images.
 - After downloading and extracting data, add `val_class_reorganizer.py` to `./imagenetdata/` and run it. 


### 3. Run attack:
```
time python make_patch.py --cuda --netClassifier inceptionv3 --max_count 500 --image_size 299 --patch_type square --outf log --store_comparison patch_size --style_transfer_weight 0.01
```

## Results:

Using patch shapes of squares gave good results 
I managed to recreate the toaster example in the original paper. It looks slightly different but it is evidently a toaster.

![Alt text](pics/success_40000.png?raw=true "") This is a toaster

Apply style transfer to beautify the patch.

![Alt text](pics/success_style_40000.png?raw=true "") This is also a toaster



