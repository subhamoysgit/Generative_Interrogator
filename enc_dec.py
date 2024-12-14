import torch
from torch import nn, optim
import torch.nn.functional as F

class ResizeConv2d(nn.Module):
    """
        Resizing ConvNet
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):      
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()


class BasicBlockEnc(nn.Module):
    """
        Used in Resnet18 Encoder
    """    

    def __init__(self, in_planes, stride=1):
        """_summary_

        Args:
            in_planes (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
        """        
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    """ 
        Used in Resnet18 Decoder
    """    

    def __init__(self, in_planes, stride=1):
     
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
            
    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], nc=4):
        """Resnet18 encoder for GAN discriminator

        Args:
            num_Blocks (list): Defaults to [2,2,2,2].
            nc (int): Defaults to 4.
        """        
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=4):
        """Resnet18 decoder for GAN generator

        Args:
            num_Blocks (list): Defaults to [2,2,2,2].
            z_dim (int, optional): Defaults to 10.
            nc (int, optional): Defaults to 4.
        """        
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)
        self.nc = nc

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=8)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), self.nc, 128, 128)
        return x
    
    
class SimEnc(nn.Module):
    
    def __init__(self,latent_dim=20, nc=4):
        """Simple Decoder Architecture used for GAN discriminator

        Args:
            latent_dim (int): dimensionality of latent space. Defaults to 20.
            nc (int): Number of channels. Defaults to 4 (for vector magnetograms).
        """        
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(nc, 32, kernel_size=3, stride=2),
                                         nn.LeakyReLU(0.2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2),
                                         nn.LeakyReLU(0.2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2),
                                         nn.LeakyReLU(0.2))
        self.linear = nn.Sequential(nn.Linear(14400, latent_dim),
                                    nn.LeakyReLU(0.2))
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
        
class SimDec(nn.Module):
    def __init__(self, latent_dim=20, image_size = 128, nc=4):
        """Simple Decoder Architecture used for GAN generator

        Args:
            latent_dim (int): dimensionality of latent space. Defaults to 20.
            image_size (int): Defaults to 128.
            nc (int, optional): Number of channels. Defaults to 4 (for vector magnetograms).
        """        
        super().__init__()
        self.deconv_block1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2),
                                           nn.LeakyReLU(0.2))
        self.deconv_block2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2),
                                           nn.LeakyReLU(0.2))
        self.deconv_block3 = nn.Sequential(nn.ConvTranspose2d(32, nc, 3, stride=2),
                                           nn.Tanh())
                                           #nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(latent_dim, 14400),
                                    nn.LeakyReLU(0.2))
        self.image_size = image_size
        
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0),64,15,15)
        x = self.deconv_block1(x)
        x = self.deconv_block2(x)
        x = F.pad(x,(0, 1, 0, 1, 0, 0, 0, 0))
        x = self.deconv_block3(x)   
        x = x[:,:,:self.image_size,:self.image_size]    
        return x


if __name__=='__main__':
    
    backbone = 'resnet'
    nc = 1
    ld = 20
    x = torch.zeros((10,nc,128,128))
    
    if backbone=='resnet':
        e = ResNet18Enc(nc=nc)
        x1 = e(x)
        x1 = nn.Linear(512, ld)(x1)
        d = ResNet18Dec(z_dim=ld, nc=nc)
        x2 = d(x1)
    else:
        e = SimEnc(latent_dim=ld,nc=nc)
        x1 = e(x)
        d = SimDec(latent_dim=ld,nc=nc)
        x2 = d(x1)
        
    print(x1.shape,x2.shape)
