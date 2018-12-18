import torch.nn as nn

class DCGAN_Generator(nn.Module):
    def __init__(self, img_size, channels, latent_dim):
        super(DCGAN_Generator, self).__init__()

        self.init_size = img_size // 2**4
        self.chans = 1024
        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.chans*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.chans),
            nn.ReLU(),

            nn.ConvTranspose2d(self.chans, self.chans//2, 4, 2, 1),
            nn.BatchNorm2d(self.chans//2),
            nn.ReLU(),

            nn.ConvTranspose2d(self.chans//2, self.chans//4, 4, 2, 1),
            nn.BatchNorm2d(self.chans//4),
            nn.ReLU(),

            nn.ConvTranspose2d(self.chans//4, self.chans//8, 4, 2, 1),
            nn.BatchNorm2d(self.chans//8),
            nn.ReLU(),

            nn.ConvTranspose2d(self.chans//8, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.chans, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_size, channels, model='DCGAN'):
        super(DCGAN_Discriminator, self).__init__()
        self.chans = 1024
        Norm = lambda x : nn.InstanceNorm2d(x, affine=True) if model is 'WGAN-GP' else nn.BatchNorm2d(x)
        self.model = nn.Sequential(
            nn.Conv2d(channels, self.chans//8, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.chans//8, self.chans//4, 4, 2, 1),
            Norm(self.chans//4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.chans//4, self.chans//2, 4, 2, 1),
            Norm(self.chans//2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.chans//2, self.chans, 4, 2, 1),
            Norm(self.chans),
            nn.LeakyReLU(0.2),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4
        if model == 'DCGAN':
            self.adv_layer = nn.Sequential( nn.Linear(self.chans*ds_size**2, 1),
                                        nn.Sigmoid())
        elif model in ('WGAN', 'WGAN-GP'):
            self.adv_layer = nn.Sequential(nn.Linear(self.chans * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity