#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @title AlphaZeroNetwork2

""" Import PyTorch framework """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet(nn.Module):

    def __init__(self, CFG):
        super().__init__()
        
        self.CFG = CFG

        """
        Iniialize a residual block with two convolutions followed by batchnorm layers
        """
        self.conv1 = nn.Conv2d(self.CFG.resnet_channels, self.CFG.resnet_channels, kernel_size=3, padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.CFG.resnet_channels)

        self.conv2 = nn.Conv2d(self.CFG.resnet_channels, self.CFG.resnet_channels, kernel_size=3, padding='same', bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.CFG.resnet_channels)

    def conv_block(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = F.relu(x, inplace=True)
        x = self.batchnorm2(self.conv2(x))
        return x

    def forward(self, x):
        """
        Combine output with the original input
        """
        x = x + self.conv_block(x)
        x = F.relu(x, inplace=True)
        return x


class AlphaZeroNetwork(nn.Module):

    def __init__(self, CFG):
        super().__init__()
        
        self.CFG = CFG        
        in_channels = self.CFG.history_size * 2 + 1

        """ Convolution block """
        self.conv1 = nn.Conv2d(in_channels, self.CFG.resnet_channels, kernel_size=3,
                               stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(self.CFG.resnet_channels)
        
        """ Resnet """
        resnet = []
        for _ in range(self.CFG.n_residual_block):
            resnet += [Resnet(self.CFG)]
        self.resnet = nn.Sequential(*resnet)

        # """ Policy for Go """
        # num_filter = 2
        # self.conv_policy = nn.Conv2d(self.CFG.resnet_channels, num_filter,
        #                             kernel_size=1, stride=1, padding='same', 
        #                             bias=False)
        # self.bn_policy = nn.BatchNorm2d(num_filter)

        # action_pass = 1
        # self.fc_policy = nn.Linear(in_features=self.CFG.action_size * num_filter, 
        #                           out_features=self.CFG.action_size + action_pass)


        """ Policy for chess and shogi. fileter数と kernel サイズに注意! """

        num_filter = 1
        # ResNetと同じblock
        self.conv_policy1 = nn.Conv2d(self.CFG.resnet_channels, CFG.resnet_channels,
                                    kernel_size=3, stride=1, padding='same', 
                                    bias=False)
        self.bn_policy1 = nn.BatchNorm2d(CFG.resnet_channels)

        # Policy output
        num_filter =1
        self.conv_policy2 = nn.Conv2d(self.CFG.resnet_channels, num_filter,
                                    kernel_size=1, stride=1, padding='same', 
                                    bias=False)
        self.bn_policy2 = nn.BatchNorm2d(num_filter)

        """ State value """
        conv_value_out_channels = 1
        self.conv_value = nn.Conv2d(self.CFG.resnet_channels, conv_value_out_channels,
                                    kernel_size=1, stride=1, padding='same', 
                                    bias=False)
        self.bn_value = nn.BatchNorm2d(conv_value_out_channels)

        fc_value_in_channels = self.CFG.action_size * conv_value_out_channels

        self.fc_value1 = nn.Linear(in_features=fc_value_in_channels, 
                                    out_features=self.CFG.hidden_size)
        self.fc_value2 = nn.Linear(in_features=self.CFG.hidden_size, out_features=1)

        """ Weight initializtion """
        self._create_weights()

    def forward(self, x):
        x = self.body(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

    def body(self, x):
        
        """ Input layer (Convolitional layer) """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        """ Residual blocks """
        x = self.resnet(x)

        return x

    def policy_head(self, x):
        """
        Architecture
        The policy head applies an additional rectified, batch-normalized convolutional layer, followed by a final convolution of 73 filters for chess or 139 filters for shogi, or a linear layer of size 362 for Go, representing the logits of the respective policies described above.
        """

        x = self.conv_policy1(x)
        x = self.bn_policy1(x)
        x = F.relu(x, inplace=True)

        x = self.conv_policy2(x)
        x = self.bn_policy2(x)
        x = F.relu(x, inplace=True)

        x = torch.flatten(x, start_dim=1) # バッチは除く
        x = F.softmax(x, dim=1)
        return x

    def value_head(self, x):
        """
        The value head applies an additional rectified, batch-normalized convolution of 1 filter of kernel size 1 x 1 with stride 1, followed by a rectified linear layer of size 256 and a tanh-linear layer of size 1.
        """
        x = self.conv_value(x)
        x = self.bn_value(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1) # バッチは除く
        x = self.fc_value1(x)
        x = F.relu(x, inplace=True)
        x = self.fc_value2(x)
        x = torch.tanh(x)
        return x

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
