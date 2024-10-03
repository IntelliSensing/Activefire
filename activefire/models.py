import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_classes, in_channels=10, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNet, self).__init__()
        self.conv1 = Conv2dBlock(in_channels, n_filters, kernel_size=3, batchnorm=batchnorm)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = Conv2dBlock(n_filters, n_filters*2, kernel_size=3, batchnorm=batchnorm)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = Conv2dBlock(n_filters*2, n_filters*4, kernel_size=3, batchnorm=batchnorm)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = Conv2dBlock(n_filters*4, n_filters*8, kernel_size=3, batchnorm=batchnorm)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(dropout)

        self.conv5 = Conv2dBlock(n_filters*8, n_filters*16, kernel_size=3, batchnorm=batchnorm)

        self.up6 = nn.ConvTranspose2d(n_filters*16, n_filters*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = Conv2dBlock(n_filters*16, n_filters*8, kernel_size=3, batchnorm=batchnorm)
        self.drop6 = nn.Dropout(dropout)

        self.up7 = nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = Conv2dBlock(n_filters*8, n_filters*4, kernel_size=3, batchnorm=batchnorm)
        self.drop7 = nn.Dropout(dropout)

        self.up8 = nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = Conv2dBlock(n_filters*4, n_filters*2, kernel_size=3, batchnorm=batchnorm)
        self.drop8 = nn.Dropout(dropout)

        self.up9 = nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = Conv2dBlock(n_filters*2, n_filters, kernel_size=3, batchnorm=batchnorm)
        self.drop9 = nn.Dropout(dropout)

        self.out = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        p1 = self.drop1(p1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        p2 = self.drop2(p2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        p3 = self.drop3(p3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        p4 = self.drop4(p4)

        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        u6 = self.conv6(u6)
        u6 = self.drop6(u6)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, c3], dim=1)
        u7 = self.conv7(u7)
        u7 = self.drop7(u7)

        u8 = self.up8(u7)
        u8 = torch.cat([u8, c2], dim=1)
        u8 = self.conv8(u8)
        u8 = self.drop8(u8)

        u9 = self.up9(u8)
        u9 = torch.cat([u9, c1], dim=1)
        u9 = self.conv9(u9)
        u9 = self.drop9(u9)

        outputs = self.out(u9)
        return outputs

def get_model(model_name, n_classes=1, in_channels=10, n_filters=16, dropout=0.1, batchnorm=True):
    if model_name == 'unet':
        return UNet(n_classes, in_channels, n_filters, dropout, batchnorm)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
