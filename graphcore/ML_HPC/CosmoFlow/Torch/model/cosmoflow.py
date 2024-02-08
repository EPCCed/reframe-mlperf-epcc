from sympy import Q
import torch
import torch.nn as nn
import torch.nn.functional as F
import poptorch

from ML_HPC.gc import GlobalContext
gc = GlobalContext()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.pool(self.act(self.conv(x)))

class StandardCosmoFlow(nn.Module):
    def __init__(self, k_size=3, n_layers=5, n_filters=32):
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        input_channels = 4  # 4 redshifts
        for i in range(n_layers):
            out_channels = n_filters * (1<<i)
            self.conv_blocks.append(ConvBlock(input_channels, out_channels, k_size))
            input_channels=out_channels
        
        flattened_shape = ((128 //(2**n_layers))**3)*input_channels  # 4x128x128x128 input shape
        self.l1 = nn.Linear(flattened_shape, 128)
        self.l2 = nn.Linear(128, 64)
        self.lO = nn.Linear(64, 4)
        self.dropout = True if gc["model"]["dropout"] != 0 else False
        if self.dropout:
            self.d1 = nn.Dropout(p=gc["model"]["dropout"])
            self.d2 = nn.Dropout(p=gc["model"]["dropout"])
        
        for l in [self.l1,self.l2,self.lO]:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
        self.criterion = nn.MSELoss()
        
    def forward(self, x, target):

        with poptorch.Block(ipu_id=0):
            x = self.conv_blocks[0](x)
        with poptorch.Block(ipu_id=1):
            x = self.conv_blocks[1](x)
            x = self.conv_blocks[2](x)
        with poptorch.Block(ipu_id=2):
            x = self.conv_blocks[3](x)
            x = self.conv_blocks[4](x)
            x = x.permute(0,2,3,4,1).flatten(1)
            
        with poptorch.Block(ipu_id=3):
            x = F.leaky_relu(self.l1(x), negative_slope=0.3)
            if self.dropout:
                x = self.d1(x)
            x = F.leaky_relu(self.l2(x), negative_slope=0.3)
            if self.dropout:
                x = self.d2(x)

            out = F.tanh(self.lO(x)) *1.2
            loss = self.criterion(out, target)
        return (out, loss)
     

