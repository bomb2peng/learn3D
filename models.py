import torch.nn as nn
import neural_renderer as nr
import torch

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


class Mesh_Generator(nn.Module):
    def __init__(self, latent_dim, filename_obj):
        super(Mesh_Generator, self).__init__()
        self.vertices_base, self.faces = nr.load_obj(filename_obj)
        self.num_vertices = self.vertices_base.shape[0]
        self.num_faces = self.faces.shape[0]
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim * 2]
        self.fclayers = nn.Sequential(
            nn.Linear(latent_dim, dim_hidden[0]),
            nn.ReLU(),
            nn.Linear(dim_hidden[0], dim_hidden[1]),
            nn.ReLU()
        )
        self.bias_layer = nn.Linear(dim_hidden[1], self.num_vertices*3)
        self.centroid_layer = nn.Linear(dim_hidden[1], 3)


    def forward(self, z):
        # if next(self.parameters()).is_cuda:  # better to call "CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py ..."
        #     self.vertices_base = self.vertices_base.cuda(next(self.parameters()).get_device())
        #     self.faces = self.faces.cuda(next(self.parameters()).get_device())

        h = self.fclayers(z)
        centroids = self.centroid_layer(h)
        centroids = 0.1*torch.tanh(centroids)
        bias = self.bias_layer(h)
        bias = bias.reshape((-1, self.num_vertices, 3))
        base = self.vertices_base
        base = base[None,:,:].repeat((bias.shape[0],1,1))

       # restrict to each quarters
        base = base*0.5
        sign = base.sign()
        base = base.abs()
        base = torch.log(base/(1-base))
        centroids = centroids[:,None,:].repeat(1,bias.shape[1],1)
        centroids = torch.tanh(centroids)
        scale_pos = 1 - centroids
        scale_neg = centroids + 1
        vertices = torch.sigmoid(base + bias)
        vertices = vertices * sign
        vertices = torch.relu(vertices) * scale_pos - torch.relu(-vertices) * scale_neg
        vertices = vertices + centroids
        vertices = vertices * self.obj_scale

        # # do not restrict to quarters
        # # base = base*0.9      # in case 1 and -1 cause log((1+base)/(1-base)) to be inf.
        # # base = 0.5*torch.log((1+base)/(1-base))
        # # vertices = torch.tanh(base + bias)
        # vertices = base + bias
        # vertices = vertices * self.obj_scale
        # centroids = centroids[:, None, :].repeat(1, bias.shape[1], 1)
        # # scales = scales[:,:, None].repeat(1, bias.shape[1], bias.shape[2])
        # vertices = vertices + centroids

        faces = self.faces[None,:,:].repeat(z.shape[0],1,1)
        return vertices, faces


class Mesh_Renderer(nn.Module):
    def __init__(self, vertices, faces, img_size=64):
        super(Mesh_Renderer, self).__init__()
        self.elevation = 30.
        self.distance = 2.732
        self.vertices = vertices
        self.register_buffer('faces', faces)
        self.img_size = img_size
        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', image_size=self.img_size, viewing_angle=15)
        self.renderer = renderer

    def forward(self, viewpoints=None):
        batch_size = self.vertices.shape[0]
        viewpoints_flag = 0
        if viewpoints is None:
            viewpoints_flag = 1
            distances = torch.ones(batch_size, device=0) * self.distance
            elevations = torch.ones(batch_size, device=0) * self.elevation
            viewids = torch.randint(0, 24, (batch_size,), device=0)
            azimuths = -viewids*15
            # azimuths = -5*15*torch.ones((batch_size,))      # restrict to a certain pose
            viewpoints = nr.get_points_from_angles(distances, elevations, azimuths)

        self.renderer.eye = viewpoints

        images = self.renderer(self.vertices, self.faces, mode='silhouettes')
        images = images.reshape((batch_size, 1, self.img_size, self.img_size))
        if viewpoints_flag == 1:
            return images, viewids
        else:
            return images


class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        dim_hidden = [dim1 * 2 ** 0, dim1 * 2 ** 1, dim1 * 2 ** 2, dim2, dim2]
        self.convBlocks = nn.Sequential(nn.Conv2d(dim_in, dim_hidden[0], 5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(dim_hidden[0], dim_hidden[1], 5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(dim_hidden[1], dim_hidden[2], 5, stride=2, padding=2),
                                        nn.ReLU())
        self.FC = nn.Sequential(nn.Linear(dim_hidden[2] * 8 * 8, dim_hidden[3]),
                                nn.ReLU(),
                                nn.Linear(dim_hidden[3], dim_hidden[4]),
                                nn.ReLU(),
                                nn.Linear(dim_hidden[4], dim_out),
                                nn.ReLU())

    def forward(self, x):
        x_conv = self.convBlocks(x)
        x_conv = x_conv.reshape((x_conv.shape[0], -1))
        x_FC = self.FC(x_conv)
        return x_FC