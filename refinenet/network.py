import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from refinenet.custom_layers import *
import copy


def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, In=False, wn=False, pixel=False, gdrop=True, circular=True, only=False):
    if circular:
        layers.append( ERP_padding(pad) )
        pad = 0

    if gdrop:       layers.append(generalized_drop_out(mode='prop', strength=0.0))
    if wn:          layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:           layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if In:      layers.append(nn.InstanceNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:      layers.append(equalized_linear(c_in, c_out, initializer='kaiming'))
    else:       layers.append(nn.Linear(c_in, c_out)) 
    if sig:     layers.append(nn.Sigmoid())
    return layers
    

def get_module_names(model):
    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


class SemGenerator(nn.Module):
    def __init__(self, config):
        super(SemGenerator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_in = config.flag_in
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_tanh
        self.flag_sigmoid_depth = config.flag_sigmoid_depth
        self.flag_norm_latent = config.flag_norm_latent
        self.nc = len(config.input_mode)
        self.outnc = config.outnc
        self.ngf = config.ngf
        self.max_ngf = config.max_ngf
        self.min_ngf = self.ngf
        self.circular = config.flag_circular_pad

        # input branches.
        # self.input_rgb = self.input_branch(3)
        # if "DL" in self.config.input_mode:
        #     self.input_d = self.input_branch(2)
        # elif "L" in self.config.input_mode or "D" in self.config.input_mode:
        #     self.input_d = self.input_branch(1)

        self.input_rgb_lr = self.input_branch(3)

        # fused data processing layer: U-Net based network.
        self.down1 = self.downsample_block(64)

        self.down2 = self.downsample_block(128)

        self.down3 = self.downsample_block(256)

        self.mid = self.mid_block(512)

        self.up3 = self.upsample_block(512)

        self.up2 = self.upsample_block(256)

        self.up1 = self.upsample_block(128)


        # output branches.
        # self.output_rgb = self.output_block(3)
        # self.output_d = self.output_block(1)
        self.output_sem = self.output_block(3)
        # self.output_d_res= self.output_block(1)
        # self.output_d_ini = self.output_block(1)

    def input_branch(self, nc):
        layers = []
        ndim = self.ngf

        layers = conv(layers, nc, ndim, 7, 1, 3, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        # downsample once.
        if "L" in self.config.input_mode or "D" in self.config.input_mode:
            ndim = min(self.max_ngf, self.ngf * 2)
            layers = conv(layers, ndim//2, ndim, 4, 2, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        else:
            ndim = min(self.max_ngf, self.ngf * 4)
            layers = conv(layers, ndim//4, ndim, 4, 2, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        
        return  nn.Sequential(*layers)



    def downsample_block(self,ngf):
        
        ndim = min(self.max_ngf, ngf * 2)

        layers = []
        #layers.append(nn.Upsample(scale_factor=2, mode='nearest'))       # scale up by factor of 2.0
        
        layers = conv(layers, ngf, ndim, 4, 2, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        
        return  nn.Sequential(*layers)
        

    def upsample_block(self,ngf):
        
        ndim = ngf//2

        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))       # scale up by factor of 2.0
        
        layers = conv(layers, ngf, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        
        return  nn.Sequential(*layers)

    def mid_block(self,ndim):
        
        layers = []
        
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        return  nn.Sequential(*layers)

    def output_block(self, outnc):

        ndim = 32

        layers = []
        # upsample once.
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))       # scale up by factor of 2.0
        layers = conv(layers, 64, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        # map to the target domain.
        layers = conv(layers, ndim, outnc, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, only=True, circular=self.circular)
        
        return nn.Sequential(*layers)

    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        #x_ = self.first_layer(x)
        rgb_lr = x
        rgb_lr = self.input_rgb_lr(rgb_lr)

        x1 = self.down1(rgb_lr) # 128, 128, 256
        # print(x1.shape)

        x2 = self.down2(x1) # 256, 64, 128
        # print(x2.shape)

        x3 = self.down3(x2) # 512, 32, 64
        # print(x3.shape)
        

        xm_ = self.mid(x3) # 512, 32, 64
        # print(xm_.shape)
        
        x3_ = self.up3(xm_ + x3) # 256, 64, 128
        # print(x3_.shape)
               
        x2_ = self.up2(x3_ + x2) # 128, 128, 256
        # print(x2_.shape)

        x1_ = self.up1(x2_ + x1) # 64, 256, 512


        sem = self.output_sem(x1_ + rgb_lr)

        sem = torch.tanh(sem)
        
        
        out = sem
     

        return out


def conv_layer(c_in, c_out, k_size, stride=1, pad=0, leaky=False, norm=False, circular=True):
    layers = []
    if circular:
        layers.append( ERP_padding(pad) )
        pad = 0 
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    #layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad))
    if norm:    layers.append(nn.InstanceNorm2d(c_out))    
    else:       pass
    if leaky:   layers.append(nn.LeakyReLU(0.2))
    else:       layers.append(nn.ReLU())
    return nn.Sequential( *layers )

class pix2pixHDDiscriminator(nn.Module):
    def __init__(self, config):
        super(pix2pixHDDiscriminator, self).__init__()
        self.config = config
        self.flag_leaky = True
        self.flag_tanh = config.flag_tanh
        self.flag_sigmoid_depth = config.flag_sigmoid_depth
        self.nc = len(config.input_mode)
        self.outnc = config.outnc # 3 for RGB only GAN.
        self.circular = config.flag_circular_pad
        self.conv1 = conv_layer(c_in=3, c_out=64, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=False, circular=self.circular) # first layer does not use instance normalization.
        self.conv2 = conv_layer(c_in=64, c_out=128, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=True, circular=self.circular)
        self.conv3 = conv_layer(c_in=128, c_out=256, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=True, circular=self.circular)
        self.conv4 = conv_layer(c_in=256, c_out=512, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=True, circular=self.circular)
        self.conv5 = nn.Conv2d(512, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.conv5(x4)

        if self.config.flag_sigmoid:
            x = torch.sigmoid(x) # do not use sigmoid, because we are using LSGAN.
        return x1, x2, x3, x4, x.mean(-2).mean(-1)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()
        self.config = config
        self.D0 = pix2pixHDDiscriminator(config)
        self.D1 = pix2pixHDDiscriminator(config)
        self.D2 = pix2pixHDDiscriminator(config)
    
    def forward(self, x):
        # zero_d = torch.tensor([1,1,1,0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(x)
        # x = x * zero_d # RGB only GAN loss.

        f01, f02, f03, f04, score0 = self.D0(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        f11, f12, f13, f14, score1 = self.D1(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        f21, f22, f23, f24, score2 = self.D2(x)
        features = [f01, f02, f03, f04, f11, f12, f13, f14, f21, f22, f23, f24]
        return features, torch.cat([score0 , score1 , score2], dim = -1)
