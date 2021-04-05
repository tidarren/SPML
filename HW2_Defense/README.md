# Homework 2: Black-box Defense

## Goal

Train a robust model for CIFAR-10 that can defend the adversarial examples. TA will use adversarial examples (up to epsilon=8 in the L_infinity norm) to attack the model.

## Abstract

In this homework, I chose five pretrained models that chosen as target models in HW1: nin, sepreresnet56, xdensenet40_2_k24_bc, ror3_110 and resnet110. Especially, I repalce resnet1001 with resnet110 for speeding up the training time. For each model, I used PGD to generate adversarial examples for each data from training set, and used them to do adversarial training. In order to understand the effect of adversarial examples in adversarial training, I tried different number of these examples. Furthermore, besides normalizing, I also do other preprocessing used in HW1: ColorJitter, CenterCrop and Padding, trying to see whether these methods can enhance robustness or not.

## Results

<img src="/Users/chenjunda/Desktop/githubPublic/SPML/HW2_Defense/table.png" style="zoom:50%;" />

For more details please refer to the report.

## Steps

### Try to output predict.txt

```
time python hw2.py ./src/data/cifar-10_eval/
```

### Generate adversarial examples

1. Set `GENERATE_PGD_EXS` in ./src/config.json to true
2. run 
```
time python adv_train.py
```

### Adversarial tranining

1. Set `GENERATE_PGD_EXS` in ./src/config.json to false
2. Advise setting in "ADVERSARIAL TRAINING CONFIGURATION" part in ./src/config.json
3. run 
```
time python adv_train.py
```

### Evaluation

1. Advise seeting in "EVAL CONFIGURATION" part in ./src/config.json
2. Run
```
python evaluation.py
```

## Contact

If you have any question, please feel free to contact me by sending email to [r08946014@ntu.edu.tw](mailto:r08946014@ntu.edu.tw)