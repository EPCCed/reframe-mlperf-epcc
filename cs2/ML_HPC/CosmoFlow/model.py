import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebras_modelzoo_common.pytorch.PyTorchBaseModel.py import PyTorchBaseModel

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
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        
        for l in [self.l1,self.l2,self.lO]:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
        
    def forward(self,x):

        for block in self.conv_blocks:
            x = block(x)

        x = x.permute(0,2,3,4,1).flatten(1)
        
        x = F.leaky_relu(self.l1(x), negative_slope=0.3)
        if self.dropout:
            x = self.d1(x)
        
        x = F.leaky_relu(self.l2(x), negative_slope=0.3)
        if self.dropout:
            x = self.d2(x)

        return F.tanh(self.lO(x)) * 1.2
     
class CosmoFlowModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.model = StandardCosmoFlow()
        self.criterion = nn.MSELoss()
        super().__init__(params=params, model=self.model, device=device)
    
    def __call__(self, data):
        x, y = data
        logits = self.model(x)
        return self.criterion(logits, y)
