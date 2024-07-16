import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

# UNET
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)
    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)
    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
#         self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
    def forward(self, down_input, skip_input):
        skip_shape = skip_input.shape
        x = F.interpolate(down_input, size=[skip_shape[2],skip_shape[3]],mode='nearest-exact')
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)
    
class UNetblock(nn.Module):
    def __init__(self):
        super(UNetblock, self).__init__()
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512)
        self.up_conv3 = UpBlock(256 + 512, 256)
        self.up_conv2 = UpBlock(128 + 256, 128)
        self.up_conv1 = UpBlock(64 + 128, 64)
        # output
        self.output = nn.Conv2d(64,64,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_image):
        x, skip1_out = self.down_conv1(input_image)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.output(x)
#         x = self.sigmoid(x)
        return x


# ResNet
class ResNetblock(nn.Module):
    def __init__(self):
        super(ResNetblock,self).__init__()
        self.resnet = models.resnet18(weights=True)
        self.resnet.conv1 = nn.Sequential()
        self.resnet.bn1 = nn.Sequential()
        self.resnet.relu = nn.Sequential()
        self.resnet.maxpool = nn.Sequential()
        self.resnet.layer3 = nn.Sequential()
        self.resnet.layer4 = nn.Sequential()
        self.resnet.fc = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    
    
    
    
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.UNetblock = UNetblock()
        self.ResNetblock = ResNetblock()
        
    def forward(self, input_image):
        x = self.UNetblock(input_image)
        output = self.ResNetblock(x)
        
        return output