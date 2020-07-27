import torch.nn as nn
import neural_renderer as nr
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from torchvision.models import resnet18

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
        # self.centroid_layer = nn.Linear(dim_hidden[1], 3)

    def forward(self, z, order=0):
        # if next(self.parameters()).is_cuda:  # better to call "CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py ..."
        #     self.vertices_base = self.vertices_base.cuda(next(self.parameters()).get_device())
        #     self.faces = self.faces.cuda(next(self.parameters()).get_device())

        h = self.fclayers(z)
        # centroid = self.centroid_layer(h)
        # centroid = 0.1*torch.tanh(centroid)
        bias = self.bias_layer(h)
        bias = bias.reshape((-1, self.num_vertices, 3))
        base = self.vertices_base
        base = base[None,:,:].repeat((bias.shape[0],1,1))

       # # restrict to each quarters
       #  base = base*0.5
       #  sign = base.sign()
       #  base = base.abs()
       #  base = torch.log(base/(1-base))
       #  centroids = centroid[:,None,:].repeat(1,bias.shape[1],1)
       #  centroids = torch.tanh(centroids)
       #  scale_pos = 1 - centroids
       #  scale_neg = centroids + 1
       #  vertices = torch.sigmoid(base + bias)
       #  vertices = vertices * sign
       #  vertices = torch.relu(vertices) * scale_pos - torch.relu(-vertices) * scale_neg
       #  vertices = vertices + centroids
       #  vertices = vertices * self.obj_scale

        # do not restrict to quarters
        # base = base*0.9      # in case 1 and -1 cause log((1+base)/(1-base)) to be inf.
        # base = 0.5*torch.log((1+base)/(1-base))
        # vertices = torch.tanh(base + bias)
        vertices_0 = base + bias
        vertices_0 = vertices_0 * self.obj_scale
        # centroids = centroids[:, None, :].repeat(1, bias.shape[1], 1)
        # scales = scales[:,:, None].repeat(1, bias.shape[1], bias.shape[2])
        # vertices = vertices + centroids

        vertices = vertices_0
        faces = self.faces[None,:,:].repeat(z.shape[0],1,1)

        return vertices, faces


class Mesh_Generator_symmetry(nn.Module):
    def __init__(self, latent_dim, filename_obj):
        super(Mesh_Generator_symmetry, self).__init__()
        self.vertices_base, self.faces = nr.load_obj(filename_obj)
        symmetry_idx = np.load('symmetry_idx.npz')
        self.idx_A = symmetry_idx['arr_0']  # idx of vertices with z=0
        self.idx_B = symmetry_idx['arr_1']  # idx of vertices with z>0
        self.idx_C = symmetry_idx['arr_2']  # idx of symmetric vertices to idx_B (about z-axis)
        self.num_vertices = self.vertices_base.shape[0]
        self.num_faces = self.faces.shape[0]
        self.output_dim = self.idx_B.shape[0]*3 + self.idx_A.shape[0]*2  # decoder output-dim, constrain symmetry
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim * 2]
        self.fclayers = nn.Sequential(
            nn.Linear(latent_dim, dim_hidden[0]),
            nn.ReLU(),
            nn.Linear(dim_hidden[0], dim_hidden[1]),
            nn.ReLU()
        )
        self.bias_layer = nn.Linear(dim_hidden[1], self.output_dim)
        # self.centroid_layer = nn.Linear(dim_hidden[1], 3)

    def forward(self, z, order=0):
        # if next(self.parameters()).is_cuda:  # better to call "CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py ..."
        #     self.vertices_base = self.vertices_base.cuda(next(self.parameters()).get_device())
        #     self.faces = self.faces.cuda(next(self.parameters()).get_device())

        h = self.fclayers(z)
        # centroid = self.centroid_layer(h)
        # centroid = 0.1*torch.tanh(centroid)
        bias_partial = self.bias_layer(h)
        bias = bias_partial.new_full([bias_partial.shape[0], self.num_vertices, 3], fill_value=0.)

        A = bias_partial[:, 0:(self.idx_A.shape[0]*2)]  # this part are x,y coordinates for idx_A vertices
        A = A.reshape((-1, self.idx_A.shape[0], 2))
        A = torch.cat([A, A.new_full([A.shape[0], A.shape[1], 1], fill_value=0.)], dim=2)  # z is 0 for idx_A vertices
        bias[:, self.idx_A, :] = A

        B = bias_partial[:, (self.idx_A.shape[0]*2):]  # this part are x,y,z coordinates for idx_B vertices
        B = B.reshape((-1, self.idx_B.shape[0], 3))
        bias[:, self.idx_B, :] = B

        C = B
        C[:,:,-1] = -C[:,:,-1]  # make symmetry about z-axis for idx_C to idx_B vertices
        bias[:, self.idx_C, :] = C

        base = self.vertices_base
        base = base[None,:,:].repeat((bias.shape[0],1,1))
        vertices_0 = base + bias
        vertices_0 = vertices_0 * self.obj_scale
        # centroids = centroids[:, None, :].repeat(1, bias.shape[1], 1)
        # scales = scales[:,:, None].repeat(1, bias.shape[1], bias.shape[2])
        # vertices = vertices + centroids

        vertices = vertices_0
        faces = self.faces[None,:,:].repeat(z.shape[0],1,1)

        return vertices, faces


class Mesh_Renderer(nn.Module):
    def __init__(self, vertices, faces, img_size=64, mode='silhouettes', dataset='CVPR18'):
        super(Mesh_Renderer, self).__init__()
        self.dataset = dataset
        if dataset == 'CVPR18':
            self.elevation = 30.
            self.distance = 2.732
            vAngle = 15
        elif dataset == 'NIPS17':
            vAngle = 14.9314
        elif dataset == 'Pascal3D':
            self.t = torch.tensor([[[0., 0., 1. + 1e-5]]], device=0)
            # the following rotation matrices are infered from Kato CVPR19 code, by pains of trial and error ..
            self.R_compen = torch.tensor([[[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], device=0)

        self.vertices = vertices
        self.register_buffer('faces', faces)
        self.img_size = img_size
        # create textures
        texture_size = 2
        textures = torch.ones(self.faces.shape[0], self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32, device=0)
        self.register_buffer('textures', textures)
        self.mode = mode
        # setup renderer
        if dataset == 'Pascal3D':       # orthogonal projection
            renderer = nr.Renderer(camera_mode='None', image_size=self.img_size)
        else:   # projective projection
            renderer = nr.Renderer(camera_mode='look_at', image_size=self.img_size, viewing_angle=vAngle)
        self.renderer = renderer

    def forward(self, viewpoints=None, viewidN=None):
        batch_size = self.vertices.shape[0]
        viewpoints_flag = 0
        if viewpoints is None:  # Just for CVPR18 dataset. generate random viewpoints
            viewpoints_flag = 1
            distances = torch.ones(batch_size, device=0) * self.distance
            elevations = torch.ones(batch_size, device=0) * self.elevation
            viewids = torch.randint(0, 24, (batch_size,), device=0)
            azimuths = -viewids*15
            # azimuths = -5*15*torch.ones((batch_size,))      # restrict to a certain pose
            viewpoints = nr.get_points_from_angles(distances, elevations, azimuths)

        if viewidN is not None:     # Just for CVPR18 dataset. generate random viewpoints that are different from "viewpointsN"
            viewpoints_flag = 1
            distances = torch.ones(batch_size, device=0) * self.distance
            elevations = torch.ones(batch_size, device=0) * self.elevation
            viewids = torch.randint(0, 24, (batch_size,), device=0)
            for i in range(batch_size):
                while viewids[i] == viewidN[i]:
                    viewids[i] = torch.randint(0, 24, (1,), device=0)
            azimuths = -viewids * 15
            viewpoints = nr.get_points_from_angles(distances, elevations, azimuths)

        if self.dataset == 'Pascal3D':
            R_rot = viewpoints  # given is the rotation matrix from Kato's Pascal3D dataset
            R = torch.matmul(self.R_compen, R_rot)
            self.vertices = torch.matmul(self.vertices, R.transpose(2, 1)) + self.t
        else:
            self.renderer.eye = viewpoints
        images = self.renderer(self.vertices, self.faces, mode=self.mode, textures=self.textures)

        if self.mode is 'silhouettes':
            images = images.reshape((batch_size, 1, self.img_size, self.img_size))
        if viewpoints_flag == 1:
            return images, viewids
        else:
            return images


class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, nConvs=3):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if nConvs == 3:
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
                                    nn.ReLU())
            # self.shapeLayer = nn.Sequential(nn.Linear(dim_hidden[4], dim_out),
            #                                 nn.ReLU())
            self.shapeLayer = nn.Linear(dim_hidden[4], dim_out)
            self.muLayer = nn.Linear(dim_hidden[4], dim_out)
            self.logvarLayer = nn.Linear(dim_hidden[4], dim_out)
        elif nConvs == 4:
            dim_hidden = [dim1 * 2 ** 0, dim1 * 2 ** 1, dim1 * 2 ** 2, dim1 * 2 ** 3, dim2, dim2]
            self.convBlocks = nn.Sequential(nn.Conv2d(dim_in, dim_hidden[0], 5, stride=2, padding=2),
                                            nn.ReLU(),
                                            nn.Conv2d(dim_hidden[0], dim_hidden[1], 5, stride=2, padding=2),
                                            nn.ReLU(),
                                            nn.Conv2d(dim_hidden[1], dim_hidden[2], 5, stride=2, padding=2),
                                            nn.ReLU(),
                                            nn.Conv2d(dim_hidden[2], dim_hidden[3], 5, stride=2, padding=2),
                                            nn.ReLU()
                                            )
            self.FC = nn.Sequential(nn.Linear(dim_hidden[3] * 8 * 8, dim_hidden[4]),
                                    nn.ReLU(),
                                    nn.Linear(dim_hidden[4], dim_hidden[5]),
                                    nn.ReLU())
            # self.shapeLayer = nn.Sequential(nn.Linear(dim_hidden[4], dim_out),
            #                                 nn.ReLU())
            self.shapeLayer = nn.Linear(dim_hidden[4], dim_out)
            self.muLayer = nn.Linear(dim_hidden[4], dim_out)
            self.logvarLayer = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x_conv = self.convBlocks(x)
        x_conv = x_conv.reshape((x_conv.shape[0], -1))
        x_FC = self.FC(x_conv)
        x_shape = self.shapeLayer(x_FC)
        return x_shape

class ResNet_Encoder(nn.Module):
    def __init__(self):
        super(ResNet_Encoder, self).__init__()
        # self.model = resnet18(pretrained=True)
        self.model = resnet18(pretrained=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

class feat_Discriminator(nn.Module):
    def __init__(self, feat_dim, out_dim=24):
        super(feat_Discriminator, self).__init__()
        self.hidden_dim = [256, 128]
        self.feat_dim = feat_dim
        self.hidden_layer0 = nn.Linear(self.feat_dim, self.hidden_dim[0])
        self.hidden_layer1 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        # self.adv_layer = nn.Linear(self.hidden_dim[1], 1)
        self.digit_layer = nn.Linear(self.hidden_dim[1], out_dim)

    def forward(self, feat):
        x_hidden0 = F.relu(self.hidden_layer0(feat))
        x_hidden1 = F.relu(self.hidden_layer1(x_hidden0))
        # validity = self.adv_layer(x_hidden1)
        logdigit = F.log_softmax(self.digit_layer(x_hidden1), dim=1)
        # return validity, x_hidden1
        return logdigit


class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(DCGAN_Discriminator, self).__init__()
        self.chans = 1024
        Norm = lambda x : nn.BatchNorm2d(x)
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
        self.adv_layer = nn.Sequential( nn.Linear(self.chans*ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity, out
