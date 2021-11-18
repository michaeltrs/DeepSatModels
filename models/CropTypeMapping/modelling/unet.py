import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.CropTypeMapping.modelling.util import initialize_weights


class _EncoderBlock(nn.Module):
    """ U-Net encoder block
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DownSample(nn.Module):
    """ U-Net downsample block
    """
    def __init__(self):
        super(_DownSample, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.downsample(x)

class _DecoderBlock(nn.Module):
    """ U-Net decoder block
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
    
class UNet(nn.Module):
    """Bring the encoder and decoder together into full UNet model
    """
    def __init__(self, num_classes, num_bands_dict, late_feats_for_fcn=False, use_planet=False, resize_planet=False):
        super(UNet, self).__init__()
        self.unet_encode = UNet_Encode(num_bands_dict, use_planet, resize_planet)
        self.unet_decode = UNet_Decode(num_classes, late_feats_for_fcn, use_planet, resize_planet)

    def forward(self, x, hres):
        center1, enc4, enc3, enc2, enc1 = self.unet_encode(x, hres) 
        final = self.unet_decode(center1, enc4, enc3, enc2, enc1)
        return final

class UNet_Encode(nn.Module):
    """ U-Net architecture definition for encoding (first half of the "U")
    """
    def __init__(self, num_bands_dict, use_planet=False, resize_planet=False):
        super(UNet_Encode, self).__init__()

        self.downsample = _DownSample() 
        self.use_planet = use_planet
        self.resize_planet = resize_planet      
 
        self.planet_numbands = num_bands_dict['planet']
        self.s1_numbands = num_bands_dict['s1']
        self.s2_numbands = num_bands_dict['s2']

        feats = 16
        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            enc3_infeats = num_bands_dict['all']
        elif self.use_planet and not self.resize_planet:
            self.enc1_hres = _EncoderBlock(self.planet_numbands, feats)
            self.enc2_hres = _EncoderBlock(feats, feats*2)
            if (self.s1_numbands > 0) or (self.s2_numbands > 0):# and model_name not in ['mi_clstm']:
                self.enc1_lres = _EncoderBlock(self.s1_numbands + self.s2_numbands, feats)
                self.enc2_lres = _EncoderBlock(feats, feats*2)
                enc3_infeats = feats*2 + feats*2
            else: 
                enc3_infeats = feats*2
       
        self.enc3 = _EncoderBlock(enc3_infeats, feats*4)
        self.enc4 = _EncoderBlock(feats*4, feats*8)

        self.center = nn.Sequential(
            nn.Conv2d(feats*8, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True))
        
        initialize_weights(self)

    def forward(self, x, hres):

        # ENCODE
        x = x.cuda()
        if hres is not None: hres = hres.cuda()
        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            enc3 = self.enc3(x)
        else:
            if hres is None:
                enc1_hres = self.enc1_hres(x)
            else:
                enc1_lres = self.enc1_lres(x)
                enc2_lres = self.enc2_lres(enc1_lres)
                enc1_hres = self.enc1_hres(hres)
 
            down1_hres = self.downsample(enc1_hres)
            enc2_hres = self.enc2_hres(down1_hres)
            down2 = self.downsample(enc2_hres)

            if hres is not None: 
                down2 = torch.cat((enc2_lres, down2), 1)

            enc3 = self.enc3(down2)

        down3 = self.downsample(enc3)
        enc4 = self.enc4(down3)
        down4 = self.downsample(enc4)
        center1 = self.center(down4)

        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            enc2 = None 
            enc1 = None
        else:
            enc2 = self.downsample(enc2_hres)
            enc1 = self.downsample(self.downsample(enc1_hres))

        return center1, enc4, enc3, enc2, enc1

class UNet_Decode(nn.Module):
    """ U-Net architecture definition for decoding (second half of the "U")
    """
    def __init__(self, num_classes, late_feats_for_fcn, use_planet=False, resize_planet=False):
        super(UNet_Decode, self).__init__()

        self.late_feats_for_fcn = late_feats_for_fcn
        self.use_planet = use_planet
        self.resize_planet = resize_planet
        
        feats = 16
        if (self.use_planet and self.resize_planet) or (not self.use_planet):
            extra_enc_feats = 0
        elif self.use_planet and not self.resize_planet: # else
            extra_enc_feats = feats + feats*2

        self.center_decode = nn.Sequential(
            nn.Conv2d(feats*16, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(feats*16, feats*8, kernel_size=2, stride=2))    
        self.dec4 = _DecoderBlock(feats*16, feats*8, feats*4)
        self.final = nn.Sequential(
            nn.Conv2d(feats*8 + extra_enc_feats, feats*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feats*4, feats*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feats*2, num_classes, kernel_size=3, padding=1),
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)
        initialize_weights(self)

    def forward(self, center1, enc4, enc3, enc2=None, enc1=None):

        # DECODE
        center2 = self.center_decode(center1)
        dec4 = self.dec4(torch.cat([center2, enc4], 1)) 

        if enc2 is not None: # concat earlier highres features
            dec4 = torch.cat([dec4, enc2, enc1], 1)

        final = self.final(torch.cat([dec4, enc3], 1)) 

        if not self.late_feats_for_fcn:
            final = self.logsoftmax(final)
        return final
