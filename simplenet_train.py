import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
##############################
#          Dilate Net
##############################

class Dilate(nn.Module):
    def __init__(self, in_size, out_size, dilate, pad, normalize=True, dropout=0.0):
        super(Dilate, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, dilation=dilate, padding=pad),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_size, out_size, 1, 1),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        side = [
            nn.Conv2d(in_size, out_size, 1, 1, bias=False),
        ]
        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x):
        x = self.model(x) + self.side(x)
#         x = self.model(x)
        return x

class MyNet(nn.Module):
    def __init__(self,input_dim=3, out_channels=100, n_conv=2):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, out_channels, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(n_conv-1):
            self.conv2.append( nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(out_channels) )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 )
        self.ln = nn.LayerNorm([512, 512])
        self.bn3 = nn.InstanceNorm2d(out_channels, affine=True)
        self.n_conv = n_conv

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.n_conv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x) 
        return x

class DilateSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=100):
        super(DilateSR, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, 1, 1)
#         self.pool = nn.MaxPool2d(2, 2)
        self.dilate_1_1 = Dilate(out_channels, out_channels, 1, 1)
        self.dilate_1_2 = Dilate(out_channels, out_channels, 2, 2)
        self.dilate_1_3 = Dilate(out_channels, out_channels, 3, 3)
        self.conv_1 = nn.Conv2d(out_channels*3, out_channels, 1, 1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.bn_2 = nn.InstanceNorm2d(out_channels, affine=True)
#         self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        
        d_1 = self.dilate_1_1(x)
        d_2 = self.dilate_1_2(x)
        d_3 = self.dilate_1_3(x)

        
        c = self.conv_1(torch.cat((d_1, d_1+d_2, d_1+d_2+d_3), 1))
        c = self.bn_1(c)
        c = self.conv_2(F.relu(c))
        c = self.bn_2(c)

        return c
    
##############################
#          RES U-NET
##############################

#   Convolutional Skip Connection
class SkipBlock(nn.Module):
    def __init__(self, in_features, out_features, residual=False):
        super(SkipBlock, self).__init__()
        self.skip = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
#             ),
#             nn.BatchNorm2d(out_features),
#             nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) # parameters
        )
        self.residual = residual

    def forward(self, x):
        return self.skip(x)

# Downsampling layers with residual connection
# 2 layers
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        side = [
            nn.Conv2d(in_size, out_size, 2, 2, bias=False),
        ]
        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x):
        x = self.model(x) + self.side(x)
        return x

# Upsampling layers with residual connection
# 2 layers
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 3, 2, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
#         side = [
#             nn.Conv2d(in_size, out_size, 1, 1, bias=False),
#         ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
#         self.side = nn.Sequential(*side)

    def forward(self, x, skip_input):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear')
#         x = self.model(x) + self.side(x)
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

# 6 downsampling + 5 skip connection + pixelshuffle
class ResUNet(nn.Module):
    def __init__(self, in_channels=100, out_channels=3):
        super(ResUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 32, normalize=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256)
        self.down5 = UNetDown(256, 256)

        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.up4 = UNetUp(128, 32)

        self.final3c = nn.Sequential(
            nn.Conv2d(64, 12, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final3c(u4)
    
class Tailor(nn.Module):
    def __init__(self, encoder, decoder):
        super(Tailor, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        sg = self.encoder(x)
#         _, temp = torch.max(sg, 1)
#         temp = torch.unsqueeze(temp, 1).float()
        rc = self.decoder(sg)
        
        return sg, rc