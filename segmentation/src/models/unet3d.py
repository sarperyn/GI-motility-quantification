import torch
import torch.nn as nn

class BaseUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=64):
        super(BaseUNet3D, self).__init__()
        
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = CBR(in_channels, base_channel)
        self.encoder2 = CBR(base_channel, base_channel * 2)
        self.encoder3 = CBR(base_channel * 2, base_channel * 4)
        self.encoder4 = CBR(base_channel * 4, base_channel * 8)
        
        self.pool = nn.MaxPool3d(2)
        
        self.bottleneck = CBR(base_channel * 8, base_channel * 16)
        
        self.upconv4 = nn.ConvTranspose3d(base_channel * 16, base_channel * 8, kernel_size=2, stride=2)
        self.decoder4 = CBR(base_channel * 16, base_channel * 8)
        self.upconv3 = nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=2, stride=2)
        self.decoder3 = CBR(base_channel * 8, base_channel * 4)
        self.upconv2 = nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=2, stride=2)
        self.decoder2 = CBR(base_channel * 4, base_channel * 2)
        self.upconv1 = nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=2, stride=2)
        self.decoder1 = CBR(base_channel * 2, base_channel)
        
        self.conv_last = nn.Conv3d(base_channel, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_last(dec1)

