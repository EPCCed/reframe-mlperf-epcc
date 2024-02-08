import sys
sys.path.append("/home/z043/z043/crae-cs1/chris-ml-intern/cs2/modelzoo") # Adds higher directory to python modules path.

import torch
import torch.nn as nn
import torch.nn.functional as F
import poptorch

class SimulatedMaxPool3D(nn.Module):
    def __init__(self, kernel_size):
        super(SimulatedMaxPool3D, self).__init__()
        self.kernel_size = kernel_size
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=self.kernel_size)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(self.kernel_size, 1))

    def forward(self, x):
        # x is expected to have shape (N, C, D, H, W)
        n, c, d, h, w = x.shape
        # Apply MaxPool2D to each slice along the depth dimension
        with poptorch.MultiConv():
            output_slices = [self.maxpool2d_1(x[:,:,i,:,:]) for i in range(d)]
        output = torch.stack(output_slices, dim=2)

        with poptorch.MultiConv():
            output_slices = [self.maxpool2d_2(output[:,:,:,i,:]) for i in range(h//self.kernel_size)]
        output = torch.stack(output_slices, dim=3)a
        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.pool = SimulatedMaxPool3D(kernel_size=2)

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
        self.dropout = True if 0.5 != 0 else False
        if self.dropout:
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
     
class CosmoFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = StandardCosmoFlow()
        self.criterion = nn.MSELoss()
        
    
    def __call__(self, data):
        x, y = data
        logits = self.model(x)
        return self.criterion(logits, y)

if __name__ == "__main__":
    torch.cuda.memory._record_memory_history()
    model = CosmoFlowModel().to("cuda")
    opt = torch.optim.SGD(model.parameters(), 0.0001)
    x, y = torch.randn(1, 4, 128, 128, 128).to("cuda"), torch.randn(1, 4).to("cuda")
    for _ in range(10):
        loss = model((x, y))
        print(loss)
        loss.backward()
        torch.cuda.synchronize()
        #print(torch.cuda.memory_summary())
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot("cosmoflow_snapshot.pickle")
    



