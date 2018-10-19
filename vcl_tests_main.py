#!/usr/bin/python
#
# Execution example:
# python vcl_tests_main.py --bn 0 --use_reg 1 --out_file vcl_elu11 --device 0 --model elu11
#
# A Reference implementation in pytorch for the Variance Constancy Loss
#
# @author = "avirambh"
# @email = "aviramb@mail.tau.ac.il"


import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.nn.functional import cross_entropy
from models.VCL import apply_vcl, get_vcl_loss
from models.ELU import ELUNetwork
from utils import AverageMeter, run_augmentation_v1, load_data, change_lr


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help="Choose dataset [cifar10|cifar100]")
    parser.add_argument("--exp_name", type=str, default="1", help="Experiment name")
    parser.add_argument("--bn", type=int, default=0, help="Use batch normalization (binary)")
    parser.add_argument("--batchsize", type=int, default=250, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--sample_size", type=int, default=5, help="Sample size for vcl regularization")
    parser.add_argument("--eps", type=float, default=0.1, help="Epsilon for vcl stability")
    parser.add_argument("--use_reg", type=int, default=1, help="Use VCL regularizer (binary)")
    parser.add_argument("--model", type=str, default='elu11', help="Model to use [elu11]")
    parser.add_argument("--gamma", type=float, default=0.01, help="Gamma value (VCL weight)")
    parser.add_argument("--gamma_l2", type=float, default=0.0001, help="L2 regularization (weight decay)")
    parser.add_argument("--activation", type=str, default='elu', help="Activation_type [elu|relu|lrelu|tanh]")
    parser.add_argument("--train_path_10", type=str, default='../data/cifar-10-batches-py/', help="CIFAR10 Train"
                                                                                                  " data path")
    parser.add_argument("--test_path_10", type=str, default='../data/cifar-10-batches-py/', help="CIFAR10 Test "
                                                                                                 "data path")
    parser.add_argument("--train_path_100", type=str, default='../data/cifar-100-batches-py/', help="CIFAR 100 train "
                                                                                                    "data path")
    parser.add_argument("--test_path_100", type=str, default='../data/cifar-100-batches-py/', help="CIFAR 100 test "
                                                                                                   "data path")
    parser.add_argument("--device", type=str, default='0', help="CUDA device to use")
    parser.add_argument("--out_file", type=str, default='', help="File to write results to")
    parser.add_argument("--save", type=str, default='checkpoints/', help="Directory to save results")
    args = parser.parse_args()
    return args

# Parse arguments
args = parseArguments()
exp_name = args.exp_name
use_vcl = args.use_reg
eps = args.eps
epochs = args.epochs
gamma = args.gamma
gamma_l2 = args.gamma_l2
sample_size = args.sample_size
batchsize = args.batchsize
lr = args.lr
use_bn = args.bn
model_type = args.model
dataset = args.dataset
out_file = args.out_file
device = args.device
save = args.save
if args.activation=='elu':
    activation = nn.ELU
if args.activation=='relu':
    activation = nn.ReLU
if args.activation=='lrelu':
    activation = nn.LeakyReLU
if args.activation=='tanh':
    activation = nn.Tanh

# Data constants
if dataset=='cifar10':
    num_of_labels=10
    train_path = args.train_path_10
    test_path = args.test_path_10
else:
    num_of_labels=100
    train_path = args.train_path_100
    test_path = args.test_path_100
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

# Prepare results file name
if use_vcl:
    out_file = out_file + '_vcl'
if use_bn:
    out_file = out_file + '_bn'

# Training parameters
use_l2_reg = 1
use_aug = 1
crop_size = 32
vcl_as_a_layer = False
debug_vcl = True
debug_model = False
offset = 0
assert not (use_vcl and use_bn)

# Load data and calculate global measures
images, lab = load_data(train_path)
me = np.mean(images,0)
images = images - me
std = np.std(images,0)
images = images/std

num_images_to_train = images.shape[0]
images = images[offset:offset+num_images_to_train].astype(np.float32)
lab = lab[offset:offset+num_images_to_train]

test_images, test_lab = load_data(test_path,test=True)
test_images = (test_images - me)/std

num_test_images = test_images.shape[0]
test_images = test_images[offset:offset+num_test_images].astype(np.float32)
test_lab = test_lab[offset:offset+num_test_images]

# Init model
if model_type == 'elu11':
    model = ELUNetwork(use_batchnorm=use_bn,
                       use_vcl=vcl_as_a_layer,
                       num_labels=10,
                       network_type='11',
                       debug=debug_model)
else:
    raise NotImplementedError

# Init VCL - for constant eps use apply_vcl(model, tmp_input, sample_size, eps_learn=False) can be used
if use_vcl:
    if vcl_as_a_layer:
        model.vcls = [w for n, w in model.named_parameters() if 'vcl' in n]
    else:
        # Simulation of a forward pass for eps initialization
        tmp_input = images.transpose(0,3,1,2)[0]

        # Applying VCL loss
        model.vcls = apply_vcl(model, tmp_input, sample_size)

# Model on cuda
if torch.cuda.is_available():
    model = model.cuda()

# Init optimizer
for mod in model.modules():
    if type(mod) == torch.nn.modules.conv.Conv2d and hasattr(mod, 'bias')\
            and mod.bias is not None:
        mod.bias.data.fill_(0)

all_weights = {name:W for name, W in model.named_parameters()}
non_bias = [W for name, W in model.named_parameters() if 'bias' not in name]
biases = [W for name, W in model.named_parameters() if 'bias' in name]
for n, w in model.named_parameters():
    if 'bias' in n:
        w.namestr = n
    if 'vcl' in n.lower():
        print("For {}, beta requires_grad: {}".format(n, w.requires_grad))

optimizer = torch.optim.SGD([{'params': non_bias, 'weight_decay':gamma_l2},
                             {'params': biases}],
                              lr=lr, momentum=0.9, nesterov=True)
print(model)

# Start log
with open(os.path.join(save, '{}.csv'.format(out_file)), 'w') as f:
    f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

# Run epochs
best_error = 1
for epoch in range(0, epochs):

    # Init train meters
    losses = AverageMeter()
    error = AverageMeter()
    batch_time = AverageMeter()
    model.train()
    end = time.time()

    # Set LR
    if epoch == 60:
        change_lr(optimizer, 0.01)
    elif epoch == 100:
        change_lr(optimizer, 0.001)
    elif epoch == 140:
        change_lr(optimizer, 0.0001)
    print "LR: ", optimizer.param_groups[0]['lr']

    # Model on train mode
    perm = np.random.permutation(num_images_to_train)
    batches = int(np.floor(num_images_to_train / batchsize))
    for j in range(batches):

        # Get Batch
        batch_idx = perm[j * batchsize:j * batchsize + batchsize]
        batch_images = images[batch_idx, :, :, :]
        batch_labels = [lab[l] for l in batch_idx]

        # Augment / Pre process
        if use_aug:
            for img_ix in range(0, batch_images.shape[0]):
                batch_images[img_ix] = run_augmentation_v1(batch_images[img_ix])
        batch_images = batch_images.transpose(0,3,1,2)

        # Torchify
        batch_images = torch.from_numpy(batch_images).cuda()
        batch_labels = torch.Tensor(batch_labels).long()
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(batch_images.cuda(async=True))
            target_var = torch.autograd.Variable(batch_labels.cuda(async=True))
        else:
            input_var = torch.autograd.Variable(batch_images)
            target_var = torch.autograd.Variable(batch_labels)

        # Forward
        output = model(input_var)

        # Get loss
        loss = cross_entropy(output, target_var)

        # Get VCL loss
        if use_vcl:
            vcl_loss = get_vcl_loss(model, epoch, debug=debug_vcl)
            loss = loss + gamma*vcl_loss

            if debug_vcl:
                print "VCL: ", vcl_loss

        # Measure accuracy and record loss
        batch_size = batch_labels.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)

        cpred = pred.squeeze()
        ctarget = batch_labels.cpu()
        error.update(torch.ne(cpred, ctarget).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        for w in model.parameters():
            clip_grad_norm(w, 1)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if j % 1 == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, epochs),
                'Iter: [%d/%d]' % (j + 1, batches),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)
    _, train_loss, train_error = batch_time.avg, losses.avg, error.avg


    # TEST
    # Model on eval mode
    model.eval()

    # Reset
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    test_batches = int(np.floor(num_test_images / batchsize))
    for j in range(test_batches):
        test_batch_images = test_images[j * batchsize:j * batchsize + batchsize, :, :, :].transpose(0,3,1,2)
        test_batch_lab = test_lab[j * batchsize:j * batchsize + batchsize]
        test_batch_images = torch.from_numpy(test_batch_images).cuda()
        test_batch_lab = torch.Tensor(test_batch_lab).long()

        # Create vaiables
        if torch.cuda.is_available():
            with torch.no_grad():
                input_var = torch.autograd.Variable(test_batch_images.cuda(async=True))
                target_var = torch.autograd.Variable(test_batch_lab.cuda(async=True))
        else:
            with torch.no_grad():
                input_var = torch.autograd.Variable(test_batch_images, volatile=True)
                target_var = torch.autograd.Variable(test_batch_lab, volatile=True)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = test_batch_lab.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), test_batch_lab.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.data.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if j % 1 == 0:
            res = '\t'.join([
                'Test',
                'Iter: [%d/%d]' % (j + 1, test_batches),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
                'VCL: {}, BN: {}'.format(use_vcl, use_bn)
            ])
            print(res)

    # Determine if model is the best
    _, valid_loss, valid_error = batch_time.avg, losses.avg, error.avg
    print("Results added to - {}".format(out_file))
    if valid_error < best_error:  # and valid_loader
        best_error = valid_error
        print('**New best error: %.4f' % best_error)
        torch.save(model.state_dict(), os.path.join(save,
                                                    '{}_best.dat'.format(out_file)))
    else:
        print('Best error: %.4f' % best_error)
        torch.save(model.state_dict(), os.path.join(save,
                                                    '{}_latest.dat'.format(out_file)))

    # Log results
    with open(os.path.join(save, '{}.csv'.format(out_file)), 'a') as f:
        f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
            (epoch + 1),
            train_loss,
            train_error,
            valid_loss,
            valid_error,
        ))