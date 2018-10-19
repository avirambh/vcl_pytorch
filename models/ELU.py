# @author = "avirambh"
# @email = "aviramb@mail.tau.ac.il"

import torch
import torch.nn as nn
import torch.nn.functional as F
from VCL import VCL


class ELUNetwork(nn.Module):
    """Networks from ELU paper
    """
    def __init__(self, use_batchnorm, use_vcl=False, num_labels=10, network_type='11', debug=False):

        super(ELUNetwork, self).__init__()
        self.bn = use_batchnorm
        self.vcl = use_vcl
        self.num_labels = num_labels
        self.debug = debug

        # Announcing
        print "Building ELU{}!".format(network_type)
        print "use_batchnorm: {}".format(use_batchnorm)

        # Build that network
        if network_type == '11':
            self.model = self.build11()
        else:
            raise NotImplementedError

    def forward(self, x):
        logits = self.model(x)

        if self.debug:
            print "logits mean: ", logits.mean()
            print "logits min: ", logits.min()
            print "logits max: ", logits.max()

        return logits.view(logits.size(0), self.num_labels)

    def build11(self):
        k = 2
        use_dropout = True
        model = nn.Sequential()

        # Layer 1
        lnum = '1'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=3,
                                                                 out_channels=192,
                                                                 kernel=5,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))
        model.add_module('layer_{}_max_pool'.format(lnum), nn.MaxPool2d(kernel_size=2,
                                                                        stride=2,
                                                                        padding=0))

        # Layer 2
        lnum = '2'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=192,
                                                                 out_channels=192*k,
                                                                 kernel=1,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))

        # Layer 2.1
        lnum = '2_1'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=192*k,
                                                                 out_channels=240*k,
                                                                 kernel=3,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))
        model.add_module('layer_{}_max_pool'.format(lnum), nn.MaxPool2d(kernel_size=2,
                                                                        stride=2,
                                                                        padding=0))
        if use_dropout:
            model.add_module('Dropout_{}'.format(lnum), nn.Dropout(p=0.1,
                                                                   inplace=False))

        # Layer 3
        lnum = '3'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=240*k,
                                                                 out_channels=240*k,
                                                                 kernel=1,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))

        # Layer 3.1
        lnum = '3_1'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=240*k,
                                                                 out_channels=260*k,
                                                                 kernel=2,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))
        model.add_module('layer_{}_max_pool'.format(lnum), nn.MaxPool2d(kernel_size=2,
                                                                        stride=2,
                                                                        padding=0))
        if use_dropout:
            model.add_module('Dropout_{}'.format(lnum), nn.Dropout(p=0.2,
                                                                   inplace=False))

        # Layer 4
        lnum = '4'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=260*k,
                                                                 out_channels=260*k,
                                                                 kernel=1,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))

        # Layer 4.1
        lnum = '4_1'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=260*k,
                                                                 out_channels=280*k,
                                                                 kernel=2,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))
        model.add_module('layer_{}_max_pool'.format(lnum), nn.MaxPool2d(kernel_size=2,
                                                                        stride=2,
                                                                        padding=0))
        if use_dropout:
            model.add_module('Dropout_{}'.format(lnum), nn.Dropout(p=0.3,
                                                                   inplace=False))

        # Layer 5
        lnum = '5'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=280*k,
                                                                 out_channels=280*k,
                                                                 kernel=1,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))

        # Layer 5.1
        lnum = '5_1'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=280*k,
                                                                 out_channels=300*k,
                                                                 kernel=2,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))
        model.add_module('layer_{}_max_pool'.format(lnum), nn.MaxPool2d(kernel_size=2,
                                                                        stride=2,
                                                                        padding=0))
        if use_dropout:
            model.add_module('Dropout_{}'.format(lnum), nn.Dropout(p=0.4,
                                                                   inplace=False))

        # Layer 6
        lnum = '6'
        model.add_module('layer_{}'.format(lnum), self.conv_layer(in_channels=300*k,
                                                                 out_channels=300*k,
                                                                 kernel=1,
                                                                 use_batchnorm=self.bn,
                                                                 layer_num=lnum,
                                                                 stride=1, vcl=self.vcl))
        if use_dropout:
            model.add_module('Dropout_{}'.format(lnum), nn.Dropout(p=0.5,
                                                                   inplace=False))

        # Layer 6.1
        lnum = '6_1'
        model.add_module('layer_{}'.format(lnum),
                         self.conv_layer(in_channels=300*k, out_channels=self.num_labels,
                                         kernel=1, use_batchnorm=False,
                                         layer_num=lnum, stride=1, vcl=False,
                                         avoid_activation=True))
        return model

    def conv_layer(self, in_channels, out_channels, kernel, use_batchnorm, layer_num,
                   stride=1, vcl=False, avoid_activation=False):
        cur_mod = nn.Sequential()

        conv_name = 'conv_{}'.format(layer_num)
        cur_mod.add_module(conv_name,
                           Conv2dSame(in_channels, out_channels, kernel_size=kernel,
                                      stride=stride, bias=True, name=conv_name)),
        if use_batchnorm:
            cur_mod.add_module('bn_'.format(layer_num), nn.BatchNorm2d(out_channels, eps=0.001))
        if vcl:
            vcl_name = 'vcl_{}'.format(layer_num)
            cur_mod.add_module(vcl_name, VCL(name=vcl_name))
        if not avoid_activation:
            activation = nn.ELU(inplace=False)
            activation.no_vcl = False
            cur_mod.add_module('ELU_{}'.format(layer_num), activation)
        return cur_mod

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, padding_layer=torch.nn.functional.pad,
                 stride=1, name='conv'):
        self.name = name
        super(Conv2dSame,self).__init__()
        self.right_down = kernel_size // 2
        self.left_up = self.right_down - 1 if kernel_size % 2 == 0 else self.right_down
        self.padding_layer = padding_layer
        self.net = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        x = self.padding_layer(x, pad=[self.left_up, self.right_down, self.left_up, self.right_down])
        out = self.net(x)
        return out
