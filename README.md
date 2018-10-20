# Variance Constancy Loss - Pytorch implementation

This repository contains a reference implementation (in PyTorch) for
"E. Littwin, L. Wolf. Regularizing by the Variance of the Activations' Sample-Variances.
 Neural Information Processing Systems (NIPS), 2018."

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)

## Introduction

## Usage
1. Download and extract [CIFAR 10/100](https://www.cs.toronto.edu/~kriz/cifar.html) for python in your data dir
2. Clone this repository
3. Train using the instructions given in [Train](#train) section

### Dependencies

- [Python2.7](https://www.python.org/downloads/)
- [PyTorch(0.4.0+)](http://pytorch.org)


### Train
As an example, use the following command to train a network with 11
convolution layers using CIFAR10 data and variance constancy loss:

```
python vcl_tests_main.py --bn 0 --use_reg 1 --out_file elu11 --device 0 --model elu11
```

For changing the data directory, please add:
```
--train_path_10 <train batches directory for CIFAR10> --test_path_10 <test batch directory for CIFAR10>
```


### Evaluation
A CSV file with the best and latest models will be saved to the checkpoint directory


### Other Options
For detailed options, please `python vcl_tests_main.py --help`

### Credits
This code is a simplified implementation in pytorch for a code written by Etai Littwin