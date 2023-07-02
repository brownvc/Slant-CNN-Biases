import torch
import torch.nn as nn
from dataloader_torch import *
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import argparse
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import datetime

from torch.utils.tensorboard import SummaryWriter

from dataloader_torch import LoaderDotSizeVar

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()


########################################################################
# models
########################################################################


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.LeakyReLU(),
        nn.BatchNorm2d(chann_out),
    )
    return layer


def trans_conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.ConvTranspose2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.LeakyReLU(),
        nn.BatchNorm2d(chann_out),
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_trans_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [trans_conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.UpsamplingNearest2d(scale_factor=2)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        # nn.BatchNorm1d(size_out),
        nn.LeakyReLU()
    )
    return layer


class AE_vgg(nn.Module):

    def __init__(self, latent_dim):
        super(AE_vgg, self).__init__()
        # encoder
        self.encoder = Encoder(latent_dim)

        # decoder
        self.decoder = Decoder(latent_dim)

    def forward(self, input):
        # encode input
        latent = self.encoder(input)

        # get output
        output = self.decoder(latent)

        return latent, output


class Decoder(nn.Module):
    """
    the decoder network
    """

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.linear = nn.Linear(latent_dim, 8 * 8 * 512)

        self.layer1 = vgg_trans_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        # self.layer1 = vgg_trans_conv_block([512], [512], [3], [1], 2, 2)
        self.layer2 = vgg_trans_conv_block([512, 512, 512], [512, 512, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer3 = vgg_trans_conv_block([256, 256, 256], [256, 256, 128], [3, 3, 3], [1, 1, 1], 2, 2)
        # self.layer3 = vgg_trans_conv_block([256, 256], [256, 128], [3, 3], [1, 1], 2, 2)
        self.layer4 = vgg_trans_conv_block([128, 128], [128, 64], [3, 3], [1, 1], 2, 2)
        self.layer5 = vgg_trans_conv_block([64, 64], [64, 1], [3, 3], [1, 1], 2, 2)

        # self.layer1 = vgg_trans_conv_block([512], [256], [3], [2], 2, 2)
        # self.layer2 = vgg_trans_conv_block([256, 128, 64], [128, 64, 64], [3, 3, 3], [2, 2, 2], 2, 2)

    def forward(self, input):
        out = self.linear(input)
        out = out.view((-1, 512, 8, 8))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


class Encoder(nn.Module):
    """
    the encoder network
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.layer1 = vgg_conv_block([1, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        # self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        # self.layer7 = vgg_fc_layer(4096, 4096)
        #
        # # Final layer
        # self.layer8 = nn.Linear(4096, latent_dim)
        self.layer8 = nn.Linear(8 * 8 * 512, latent_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        # out = self.layer6(out)
        # out = self.layer7(out)
        out = self.layer8(out)

        return out

