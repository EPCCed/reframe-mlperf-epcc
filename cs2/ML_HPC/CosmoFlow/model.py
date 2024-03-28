import sys
from turtle import forward
sys.path.append("/home/z043/z043/crae-cs1/chris-ml-intern/cs2/modelzoo") # Adds higher directory to python modules path.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple
import cerebras_pytorch as cstorch

class SimulatedMaxPool3D_p1(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=self.kernel_size)
        
    
    def forward(self, x):
        # x is expected to have shape (N, C, D, H, W)
        n, c, d, h, w = x.shape
        # Apply MaxPool2D to each slice along the depth dimension
        output_slices = [self.maxpool2d_1(x[:,:,i,:,:]) for i in range(d)]
        output = torch.stack(output_slices, dim=2)
        return output, h

class SimulatedMaxPool3D_p2(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(self.kernel_size, 1))
    
    def forward(self, x, h):
        output_slices = [self.maxpool2d_2(x[:,:,:,i,:]) for i in range(h//self.kernel_size)]
        output = torch.stack(output_slices, dim=3)
        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_channels, in_channels, k_size, k_size, k_size, dtype=torch.float16, requires_grad=True))
        self.b = nn.Parameter(torch.empty(out_channels, dtype=torch.float16, requires_grad=True))
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.pool_p1 = SimulatedMaxPool3D_p1(2)
        self.pool_p2 = SimulatedMaxPool3D_p2(2)

        torch.nn.init.xavier_uniform_(self.w)
        torch.nn.init.zeros_(self.b)
    
    def forward(self, x):

        x = F.conv3d(F.pad(x, (1,1,1,1,1,1)), self.w, bias = self.b)
        x = F.leaky_relu(x, 0.3)
        return self.pool_p2(*self.pool_p1(x))

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
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        
        for l in [self.l1,self.l2,self.lO]:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
        
    def forward(self,x):

        for block in self.conv_blocks:
            x = block(x)
            print(x.shape)

        x= x.permute(0,2,3,4,1).flatten(1)
        
        x = F.leaky_relu(self.l1(x), negative_slope=0.3)
        x = self.d1(x)
        
        x = F.leaky_relu(self.l2(x), negative_slope=0.3)
        x = self.d2(x)

        return torch.tanh(self.lO(x)) * 1.2

if __name__ == "__main__":
    from torch.fx import symbolic_trace
    model = StandardCosmoFlow()

    def train_loop(x):
        logits = model(x)
        loss = torch.sum(logits)
        loss.backward()

    print(symbolic_trace(model).graph)

    x = torch.randn(1, 4, 128,128,128)
    model(x)