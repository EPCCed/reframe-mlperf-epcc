import math
from termios import INPCK
import torch
from torch import conv2d
from torch import ParameterDict
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torch.nn import BatchNorm2d as GroupBN
have_gbn = False


class GBN(nn.Module):
    def __init__(self, num_features, fuse_relu=False, process_group=None, enable_gbn=False):
        super(GBN, self).__init__()
        
        if process_group is None:
            bngsize = 1
        else:
            bngsize = process_group.size()

        if have_gbn and enable_gbn and (bngsize > 1):
            self.bn = GroupBN(num_features)

            if fuse_relu:
                self.run_handle = nn.Sequential(self.bn, nn.ReLU())
            else:
                self.run_handle = self.bn
        else:
            self.bn = nn.SyncBatchNorm(num_features, process_group = process_group)
            if fuse_relu:
                self.run_handle = nn.Sequential(self.bn, nn.ReLU())
            else:
                self.run_handle = self.bn
            
        self.jit_scriptable = False
        
        
    def forward(self, x, z=None):
        
        x = self.run_handle(x)
        
        return x

def get_batchnorm(num_features, fuse_relu=False, process_group=None, enable_gbn=False):
    if process_group is not None:
        return GBN(num_features, fuse_relu=fuse_relu, process_group=process_group, enable_gbn=enable_gbn)
    else:
        if fuse_relu:
            return nn.Sequential(nn.BatchNorm2d(num_features), nn.ReLU())
        else:
            return nn.BatchNorm2d(num_features)


class FullGroupConv2D(nn.Module):
    def __init__(self, in_planes, k_size, stride, padding, dilation, bias):
        super().__init__()
        if isinstance(k_size, int):
            k_size = (k_size, k_size)
        self.k = torch.nn.Parameter(torch.randn(in_planes, 1, *k_size), requires_grad=True)
        if bias:
            self.b = torch.nn.Parameter(torch.zeros(in_planes))
        else:
            self.b = None
        self.conv_kwargs = {"stride": stride, "padding": padding, "dilation": dilation}
        
        #self.convs = [nn.Conv2d(1, 1, k_size, stride, padding, dilation, bias=bias) for _ in range(in_planes)]

    def forward(self, x):
        #return torch.vmap(lambda x, k: F.conv2d(x.unsqueeze(1), k.unsqueeze(0), bias = self.b, **self.conv_kwargs).squeeze(1), in_dims=(1,0), out_dims=1)(x, self.k)
        slices = [F.conv2d(x[:,i,:,:].unsqueeze(1), self.k[i,:,:,:].unsqueeze(0), bias = self.b, **self.conv_kwargs) for i in range(x.size(1))]
        return torch.cat(slices, dim=1)

class PointwiseConv2D(nn.Module):
    def __init__(self, inplanes, planes, bias):
        super().__init__()
        self.k = torch.nn.Parameter(torch.randn(planes, inplanes, 1, 1), requires_grad=True)
        if bias:
            self.b = torch.nn.Parameter(torch.zeros(planes))
        else:
            self.b = None
    def forward(self, x):
        return F.conv2d(x, self.k, bias=self.b)
    

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        #self.conv1 = FullGroupConv2D(inplanes, kernel_size, stride, padding, dilation, bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

    
def compute_padding(kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return pad_beg, pad_end


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        # compute padding here
        pad_beg, pad_end = compute_padding(kernel_size, rate=dilation)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, (pad_beg, pad_beg), dilation, groups=inplanes, bias=bias)
        #self.conv1 = FullGroupConv2D(inplanes, kernel_size, stride, (pad_beg, pad_beg), dilation, bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, 
                 end_with_relu=True, fuse_relu=False, grow_first=True, is_last=False,
                 process_group=None, enable_gbn=False):
        super(Block, self).__init__()
        
        self.enable_gbn = enable_gbn
        
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = get_batchnorm(planes, fuse_relu=False, process_group=process_group, enable_gbn = self.enable_gbn)
        else:
            self.skip = None
            
        self.end_with_relu = end_with_relu
        
        self.relu = nn.ReLU()
        rep = []
        
        filters = inplanes
        if grow_first:
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(get_batchnorm(planes, fuse_relu=fuse_relu, process_group=process_group, enable_gbn = self.enable_gbn))
            if not fuse_relu:
                rep.append(self.relu)
            filters = planes

        for i in range(reps - 1):
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            
            if not fuse_relu:
                rep.append(get_batchnorm(filters, fuse_relu=False, process_group=process_group, enable_gbn = self.enable_gbn))
                rep.append(self.relu)
            else:
                rep.append(get_batchnorm(filters, 
                                         fuse_relu=(i<reps-2) or (not grow_first), 
                                         process_group=process_group, enable_gbn = self.enable_gbn))

        if not grow_first:
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(get_batchnorm(planes, fuse_relu=False, process_group=process_group, enable_gbn = self.enable_gbn))
            if not fuse_relu:
                rep.append(self.relu)

        #if not start_with_relu:
        # rep = rep[1:]
        # drop the relu after this
        if not fuse_relu:
            rep = rep[:-1]

        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        if self.end_with_relu:
            x = self.relu(x)

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False, process_group=None, enable_gbn=False):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.enable_gbn = enable_gbn
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = get_batchnorm(32, fuse_relu=True, process_group=process_group, enable_gbn = self.enable_gbn)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = get_batchnorm(64, fuse_relu=True, process_group=process_group, enable_gbn = self.enable_gbn)

        self.block1 = Block(64, 128, reps=2, stride=2, end_with_relu=False, fuse_relu=True, 
                            process_group=process_group, enable_gbn = self.enable_gbn)
        self.block2 = Block(128, 256, reps=2, stride=2, end_with_relu=True, fuse_relu=True, grow_first=True, 
                            process_group=process_group, enable_gbn = self.enable_gbn)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, end_with_relu=True, fuse_relu=False, grow_first=True,
                            is_last=True, process_group=process_group, enable_gbn = self.enable_gbn)

        # Middle flow
        # no pad
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, end_with_relu=True, fuse_relu=True, grow_first=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        
        # Exit flow
        # no pad
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0],
                             end_with_relu=False, fuse_relu=True, grow_first=False, is_last=True, 
                             process_group=process_group, enable_gbn = self.enable_gbn)
        
        # set force_gbn to false for these ones atm because of some cudnn issue
        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn3 = get_batchnorm(1536, fuse_relu=True, process_group=process_group, enable_gbn = self.enable_gbn)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn4 = get_batchnorm(1536, fuse_relu=True, process_group=process_group, enable_gbn = self.enable_gbn)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_rates[1])
        self.bn5 = get_batchnorm(2048, fuse_relu=True, process_group=process_group, enable_gbn = self.enable_gbn)


    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.relu(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        
        x = self.conv3(x)        
        x = self.bn3(x)
        #x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        #x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        #x = self.relu(x)

        return x, low_level_feat
    

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            print(k)
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, process_group=None, enable_gbn=False):
        super(ASPP_module, self).__init__()
        
        # do we want to enable gbn?
        self.enable_gbn = enable_gbn
        
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
                                            
        self.bn = get_batchnorm(planes, fuse_relu=True, process_group=process_group, enable_gbn = self.enable_gbn)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        #x = self.relu(x)

        return x


class DeconvUpsampler(nn.Module):
    def __init__(self, n_output, process_group=None, enable_gbn=False):
        super(DeconvUpsampler, self).__init__()
        
        # do we want to enable gbn?
        self.enable_gbn = enable_gbn
        
        # deconvs
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                     get_batchnorm(256, fuse_relu=True, process_group=process_group, enable_gbn=self.enable_gbn),
                                     #nn.ReLU(),
                                     nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                     get_batchnorm(256, fuse_relu=True, process_group=process_group, enable_gbn=self.enable_gbn)) #,
                                     #nn.ReLU())
            
        self.conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   get_batchnorm(256, fuse_relu=True, process_group=process_group, enable_gbn=self.enable_gbn),
                                   #nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   get_batchnorm(256, fuse_relu=True, process_group=process_group, enable_gbn=self.enable_gbn),
                                   #nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=1, stride=1))
            
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                     get_batchnorm(256, fuse_relu=True, process_group=process_group, enable_gbn=self.enable_gbn))#,
                                     #nn.ReLU())

	    #no bias or BN on the last deconv
        self.last_deconv = nn.Sequential(nn.ConvTranspose2d(256, n_output, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False))

    def forward(self, x, low_level_features):
        x = self.deconv1(x)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.last_deconv(x)
        return x

                
class TrainableAffine(nn.Module):
    def __init__(self, num_features):
        super(TrainableAffine, self).__init__()
        self.num_features = num_features

        # weights for affine trans
        #self.weights = nn.Parameter(torch.ones((num_features, 1, 1), requires_grad=True))
        #self.bias =  nn.Parameter(torch.zeros((num_features, 1, 1), requires_grad=True))

    def forward(self, x):
        return x
        return torch.add(torch.mul(x, self.weights), self.bias)
        print(x.shape)
        out =  self.weights * x + self.bias
        print(out.shape)
        return out


class GlobalAveragePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalAveragePool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # make the relu inplace to avoid unnecessary layout changes
        self.global_average_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                                                 TrainableAffine(out_channels),
                                                 nn.ReLU())

    def forward(self, x):
        return self.global_average_pool(x)

    
class Tiling(nn.Module):
    def __init__(self, tiles):
        super(Tiling, self).__init__()
        self.tiles = tiles

    # warning, this trick only works for H=W=1
    # hacky way to deal with annoying layout changes in tile:
    def forward(self, x: torch.Tensor):
        if x.stride(1) > 1:
            return torch.tile(x, self.tiles)
        else:
            # Modified to work with cerebras
            N, C, H, W = x.shape
            # xtmp = torch.as_strided(x, size=[N, 1, 1, C], stride=[C, C, C, 1])
            xtmp = x.permute(0,3,2,1)
            # tiling is in nchw, needs to change to nhwc
            NT, CT, HT, WT = self.tiles
            #xtmp = torch.tile(xtmp, [NT, HT, WT, CT])
            xtmp = xtmp.repeat([NT, HT, WT, CT])
            #x = torch.as_strided(xtmp, size=[N*NT, C*CT, HT, WT], stride=[HT*WT*CT*C, 1, WT*CT*C, CT*C])
            x = xtmp.permute(0, 3, 1, 2)
            return x
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, os, process_group = None, enable_gbn = False):
        super(Bottleneck, self).__init__()
        
        self.enable_gbn = enable_gbn
        
        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(in_channels, out_channels, rate=rates[0], process_group = process_group, enable_gbn = self.enable_gbn)
        self.aspp2 = ASPP_module(in_channels, out_channels, rate=rates[1], process_group = process_group, enable_gbn = self.enable_gbn)
        self.aspp3 = ASPP_module(in_channels, out_channels, rate=rates[2], process_group = process_group, enable_gbn = self.enable_gbn)
        self.aspp4 = ASPP_module(in_channels, out_channels, rate=rates[3], process_group = process_group, enable_gbn = self.enable_gbn)

        # removed batch normalization in this layer
        self.global_avg_pool = GlobalAveragePool(in_channels, out_channels)
        self.tiles = (1, 1, 48, 72)
        self.tiling = Tiling(self.tiles)
        
        # convs and relus
        self.conv = nn.Conv2d(5 * out_channels, out_channels, 1, bias=False)
        self.bn = get_batchnorm(out_channels, fuse_relu=True, process_group=process_group, enable_gbn=self.enable_gbn)
        #self.relu = nn.ReLU()


    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        # this is very expensive in BW
        #x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        # this is the same and much cheaper
        #x5 = torch.tile(x5, self.tiles)
        #if x.is_contiguous(memory_format=torch.channels_last):
        #    x5 = x5.contiguous(memory_format=torch.channels_last)
        x5 = self.tiling(x5)
        
        # concat
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        # process
        x = self.conv(x)
        x = self.bn(x)
        #x = self.relu(x)
        
        return x

class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        # init superclass
        loss_pow = -0.125
        #weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
        self.weights = torch.tensor([0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def forward(self, logit, target):
        target = target.unsqueeze(1)
        ls_logit = F.log_softmax(logit, dim=1)
        losses = self.weights*target*ls_logit
        return torch.mean(losses)
    
                
class DeepLabv3_plus(nn.Module):
    def __init__(self, n_input=3, n_classes=21, os=16, pretrained=False, process_group = None, enable_gbn = False):
        super(DeepLabv3_plus, self).__init__()

        # groupbn
        self.enable_gbn = enable_gbn

        # encoder
        self.xception_features = Xception(n_input, os, pretrained, process_group = process_group, enable_gbn = self.enable_gbn)

        # bottleneck
        self.bottleneck = Bottleneck(2048, 256, os, process_group = process_group, enable_gbn = self.enable_gbn)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = get_batchnorm(48, fuse_relu=True, process_group=process_group, enable_gbn = self.enable_gbn)
        #self.relu = nn.ReLU()

        # upsampling
        self.upsample = DeconvUpsampler(n_classes, process_group = process_group, enable_gbn = self.enable_gbn)


    def forward(self, input):
        # encoder
        x, low_level_features = self.xception_features(input)
        
        # bottleneck
        x = self.bottleneck(x)

        # low level feature processing
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        #low_level_features = self.relu(low_level_features)
        
        # decoder / upsampling logic
        x = self.upsample(x, low_level_features)
        return x

        return self.criterion(x, target)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus(n_input=16, n_classes=3, pretrained=False, process_group=None,)
    for i in model.named_parameters(),:
        for s, p in i:
            if p.ndim == 4:
                if p.shape == torch.Size([1,1,3,1]):
                    print(s, p.shape)
    x = torch.randn([1, 16, 768, 1152])
    #model(x)