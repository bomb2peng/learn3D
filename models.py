import torch.nn as nn


class DCGAN_Generator(nn.Module):
    def __init__(self, img_size, channels, latent_dim):
        super(DCGAN_Generator, self).__init__()

        self.init_size = img_size // 2**4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 1024*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_size, channels, model='DCGAN'):
        super(DCGAN_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4
        if model == 'DCGAN':
            self.adv_layer = nn.Sequential( nn.Linear(1024*ds_size**2, 1),
                                        nn.Sigmoid())
        elif model == 'WGAN':
            self.adv_layer = nn.Sequential(nn.Linear(1024 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity