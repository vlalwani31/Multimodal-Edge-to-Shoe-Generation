from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb

##############################
#        Encoder
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
            This encoder uses resnet-18 to extract features, and further encode them into a distribution
            similar to VAE encoder.

            Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file

            Args in constructor:
                latent_dim: latent dimension for z

            Args in forward function:
                img: image input (from domain B)

            Returns:
                mu: mean of the latent code
                logvar: sigma of the latent code
        """

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


##############################
#        Generator
##############################
class Generator(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B

        Args in constructor:
            latent_dim: latent dimension for z
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)

        Args in forward function:
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns:
            fake_B: generated image in domain B
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape
        # (TODO: add layers...)
        # Using U-Net for the generator
        # First conv the channels to 16, retain dims
        self.start_layer = nn.Sequential(
                        nn.Conv2d(channels + latent_dim, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the channels to 32 half the dims to 64
        self.encode_layer1 = nn.Sequential(
                        nn.Conv2d(16, 32, 4, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the channels to 64, half the dims, to 32
        self.encode_layer2 = nn.Sequential(
                        nn.Conv2d(32, 64, 4, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the channels to 128, half the dims, to 16
        self.encode_layer3 = nn.Sequential(
                        nn.Conv2d(64, 128, 4, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the channels to 256, half the dims, to 8
        self.encode_layer4 = nn.Sequential(
                        nn.Conv2d(128, 256, 4, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the channels to 512, half the dims, to 4
        self.encode_layer5 = nn.Sequential(
                        nn.Conv2d(256, 512, 4, stride=2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next Up-conv the channels to 16, double the dims to 128
        self.decode_up_layer1 = nn.Sequential(
                        nn.ConvTranspose2d(32, 16, 4,stride=2, padding=1),
                        nn.BatchNorm2d(16),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the concated output to 16 channels, retain dim
        self.decode_conc_layer1 = nn.Sequential(
                        nn.Conv2d(32, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next Up-conv the channels to 32, double the dims to 64
        self.decode_up_layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 4,stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the concated output to 32 channels, retain dim
        self.decode_conc_layer2 = nn.Sequential(
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next Up-conv the channels to 64, double the dims to 32
        self.decode_up_layer3 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 4,stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the concated output to 64 channels, retain dim
        self.decode_conc_layer3 = nn.Sequential(
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next Up-conv the channels to 128, double the dims to 16
        self.decode_up_layer4 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 4,stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the concated output to 128 channels, retain dim
        self.decode_conc_layer4 = nn.Sequential(
                        nn.Conv2d(256, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next Up-conv the channels to 256, double the dims to 8
        self.decode_up_layer5 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 4,stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True))
        # Next conv the concated output to 256 channels, retain dim
        self.decode_conc_layer5 = nn.Sequential(
                        nn.Conv2d(512, 256, 3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True))
        # Final conv the channels to 3, retain dim
        self.final_layer = nn.Sequential(
                        nn.Conv2d(16, 3, 3, padding=1),
                        nn.Tanh())

    def forward(self, x, z):
        # (TODO: add layers...)
        z1 = z.unsqueeze(2).unsqueeze(2).repeat(1,1,self.h,self.w)
        # z1.requires_grad = False
        x1 = torch.cat([x,z1] ,dim=1)
        x1 = self.start_layer(x1)
        e1 = self.encode_layer1(x1)
        e2 = self.encode_layer2(e1)
        e3 = self.encode_layer3(e2)
        e4 = self.encode_layer4(e3)
        e5 = self.encode_layer5(e4)
        du5 = self.decode_up_layer5(e5)
        dc5 = torch.cat([du5,e4] ,dim=1)
        dc5 = self.decode_conc_layer5(dc5)
        du4 = self.decode_up_layer4(dc5)
        dc4 = torch.cat([du4,e3] ,dim=1)
        dc4 = self.decode_conc_layer4(dc4)
        du3 = self.decode_up_layer3(dc4)
        dc3 = torch.cat([du3,e2] ,dim=1)
        dc3 = self.decode_conc_layer3(dc3)
        du2 = self.decode_up_layer2(dc3)
        dc2 = torch.cat([du2,e1] ,dim=1)
        dc2 = self.decode_conc_layer2(dc2)
        du1 = self.decode_up_layer1(dc2)
        dc1 = torch.cat([du1,x1] ,dim=1)
        dc1 = self.decode_conc_layer1(dc1)
        x8 = self.final_layer(dc1)
        return x8

class Generator2(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B

        Args in constructor:
            latent_dim: latent dimension for z
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)

        Args in forward function:
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns:
            fake_B: generated image in domain B
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator2, self).__init__()
        channels, self.h, self.w = img_shape
        # (TODO: add layers...)
        # Using a vanilla generator
        # First conv the channels to 16, retain dims
        self.start_layer = nn.Sequential(
                        nn.Conv2d(channels + latent_dim, 32, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(32, 64, 4, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(64, 128, 4, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(128, 256, 4, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(256, 512, 4, stride=2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(512, 256, 4,stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(256, 128, 4,stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(128, 64, 4,stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(64, 32, 4,stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(32, 3, 4,stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, z):
        # (TODO: add layers...)
        z1 = z.unsqueeze(2).unsqueeze(2).repeat(1,1,self.h,self.w)
        # z1.requires_grad = False
        x1 = torch.cat([x,z1] ,dim=1)
        x2 = self.start_layer(x1)
        return x2

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        """ The discriminator used in both cVAE-GAN and cLR-GAN

            Args in constructor:
                in_channels: number of channel in image (default: 3 for RGB)

            Args in forward function:
                x: image input (real_B, fake_B)

            Returns:
                discriminator output: could be a single value or a matrix depending on the type of GAN
        """
        self.layer =  nn.Sequential(
                        nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(32, 64, 4,stride=2,padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, 128, 4,stride=2,padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(128, 256, 4,stride=2,padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(256, 512, 4,stride=2,padding=1),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(512, 1, 4),
                        nn.Sigmoid())

    def forward(self, x):
        return self.layer(x)
