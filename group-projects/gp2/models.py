import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding='same'),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding='same'),
            )
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding='same')
            for d in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = xt + x
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=80, out_channels=1, channels=512, kernel_size=7, upsample_rates=[8,8,2,2], upsample_kernel_sizes=[16,16,4,4]):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, channels, kernel_size, padding='same')
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(
                channels // (2**i),
                channels // (2**(i+1)),
                k, stride=u, padding=(k-u)//2
            ))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = channels // (2**(i+1))
            self.resblocks.append(ResBlock(ch))
        
        self.conv_post = nn.Conv1d(channels // (2**len(upsample_rates)), out_channels, 7, padding='same')

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = self.resblocks[i](x)
            x = xs + x
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=64, kernel_size=15):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, channels, kernel_size, padding=(kernel_size-1)//2)
        
        self.convs = nn.ModuleList()
        for i in range(3):
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels * (2**i), channels * (2**(i+1)), kernel_size, stride=2, padding=(kernel_size-1)//2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels * (2**(i+1)), channels * (2**(i+1)), kernel_size, padding=(kernel_size-1)//2),
            ))
        
        self.conv_post = nn.Conv1d(channels * 8, out_channels, kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        x = self.conv_pre(x)
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        x = self.conv_post(x)
        return x, features 