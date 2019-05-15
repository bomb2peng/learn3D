import argparse
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import time
import datetime
import models as M
import data_loader
import torch.utils.data as data
import torch.nn.functional as F
import os
import neural_renderer as nr
import losses as L
import logger
import voxelization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ('true')


CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
DATASET_DIRECTORY = '/hd2/pengbo/mesh_reconstruction/dataset/'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='trainGAN', help='mode can be one of [trainGAN, trainAE, sampleGAN]')
parser.add_argument('--conditioned', type=int, default=0, help='whether to use conditioned GAN')
parser.add_argument('--use_VAE', action='store_true', default=False, help='use VAE for the encoder')
parser.add_argument('--data_dir', type=str, help='dir of dataset')
parser.add_argument('--crop_size', type=int, default=178, help='size of center crop for celebA')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='number of epochs before lr decay')
parser.add_argument('--decay_order', type=float, default=0.1, help='order of lr decay')
parser.add_argument('--decay_every', type=int, default=5, help='lr decay every n epochs')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_step', type=int, default=1, help='number of iters between image sampling')
parser.add_argument('--sample_dir', type=str, help='dir of saved '
                                                                                                'sample images')
parser.add_argument('--log_step', type=int, default=10, help='number of iters to print and log')
parser.add_argument('--ckpt_step', type=int, default=1000, help='number of iters for model saving')
parser.add_argument('--ckpt_dir', type=str, help='dir of saved model checkpoints')
parser.add_argument('--visdom_env', type=str, default=None, help='Visdom environment name')

parser.add_argument('--load_G', type=str, default=None, help='path of to the loaded Generator weights')
parser.add_argument('--load_D', type=str, default=None, help='path of to the loaded Discriminator weights')
parser.add_argument('--load_E', type=str, default=None, help='path of to the loaded Encoder weights')
parser.add_argument('--N_generated', type=int, default=1000, help='number of generated images')
parser.add_argument('--dir_generated', type=str, default=None, help='saving dir of to be generated images')
parser.add_argument('--train_perc', type=float, default=0.5, help='percentage of training split')
parser.add_argument('--device_id', type=int, default=0, help='choose device ID')
parser.add_argument('--n_iters', type=int, default=3000, help='number of iterations for training')
parser.add_argument('--model', type=str, default='DCGAN', help='choose GAN model, can be [DCGAN, WGAN, WGAN-GP...]')
parser.add_argument('--G_every', type=int, default=1, help='G:D training schedule is 1:G_every')
parser.add_argument('--clip_value', type=float, default=0.01, help='clip value for D weights in WGAN trainings')
parser.add_argument('--batches_done', type=int, default=0, help='previous batches_done when '
                                                                'loading ckpt and continue traning')
parser.add_argument('--gp', type=float, default=10.0, help='gradient penalty for WGAN-GP')

parser.add_argument('--class_ids', type=str, default=CLASS_IDS_ALL, help='use which object class images')
parser.add_argument('--obj_dir', type=str, help='base sphere obj file path')
parser.add_argument('--lambda_smth', type=float, default=0.1, help='weight for mesh smoothness loss')
parser.add_argument('--lambda_Lap', type=float, default=0.1, help='weight for mesh Laplacian loss')
parser.add_argument('--lambda_feat', type=float, default=0, help='weight of feature matching loss')
parser.add_argument('--lambda_edge', type=float, default=0.1, help='weight of edge length loss')

parser.add_argument('--n_samples', type=int, default=64, help='number of samples to generate and save')
parser.add_argument('--sample_prefix', type=str, default='sample', help='prefix of saved sample file names')

parser.add_argument('--iter_divide1', type=int, default=1000, help='number of iters before first subdividing to mesh')
parser.add_argument('--iter_divide2', type=int, default=3000, help='number of iters before second subdividing to mesh')

t_start = time.time()
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = 'cuda:%d' % opt.device_id if opt.device_id >= 0 else 'cpu'
print(device)
if opt.visdom_env is not None:
    ploter = logger.VisdomLinePlotter(env_name=opt.visdom_env)

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device=device)
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True)
    o, _ = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size(), device=device),
                            create_graph=True, retain_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp

if opt.mode == 'trainGAN':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # Initialize generator and discriminator
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    discriminator = M.DCGAN_Discriminator(opt.img_size, opt.channels, opt.model)
    if os.path.isfile('smoothness_params_42.npy'):
        smoothness_params = np.load('smoothness_params_42.npy')
    else:
        smoothness_params = L.smoothness_loss_parameters(mesh_generator.faces, 'smoothness_params_42.npy')
    if os.path.isfile('Laplacian_params_42.npy'):
        Laplacian_params = np.load('Laplacian_params_42.npy')
    else:
        Laplacian_params = L.Laplacian_loss_parameters(mesh_generator.num_vertices, mesh_generator.faces.tolist(), \
                                                       'Laplacian_params_42.npy')

    if cuda:
        mesh_generator.cuda(opt.device_id)
        discriminator.cuda(opt.device_id)

    # Initialize weights
    if opt.load_G is None:
        # mesh_generator.apply(weights_init_normal)
        pass
    else:
        mesh_generator.load_state_dict(torch.load(opt.load_G))

    if opt.load_D is None:
        # discriminator.apply(weights_init_normal)
        pass
    else:
        discriminator.load_state_dict(torch.load(opt.load_D))

    # Configure data loader
    dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train', opt.img_size)
    dataloader = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    # Optimizers
    if opt.model in ('DCGAN', 'WGAN-GP'):
        optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
    elif opt.model == 'WGAN':
        optimizer_G = torch.optim.RMSprop(mesh_generator.parameters(), lr=opt.lr)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    # Sample fixed random noise to see generated results
    z_fixed = Variable(Tensor(np.random.normal(0, 1, (24, opt.latent_dim)), device=device))
    azimuths = -15. * torch.arange(0, 24)
    azimuths = torch.Tensor.float(azimuths)
    elevations = 30. * torch.ones((24))
    distances = 2.732 * torch.ones((24))
    viewpoints_fixed = nr.get_points_from_angles(distances, elevations, azimuths)

    iter_divide_1 = opt.iter_divide1
    iter_divide_2 = opt.iter_divide2
    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(dataloader)
    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _, viewids) in enumerate(dataloader):
            if batches_done == iter_divide_1:
                print('initializing subdivied-1 mesh params...')
                mesh_generator.init_order1()
                if os.path.isfile('smoothness_params_devide_1.npy'):
                    smoothness_params = np.load('smoothness_params_devide_1.npy')
                else:
                    smoothness_params = L.smoothness_loss_parameters(mesh_generator.faces_1, \
                                                                     'smoothness_params_devide_1.npy')
                if os.path.isfile('Laplacian_params_devide_1.npy'):
                    Laplacian_params = np.load('Laplacian_params_devide_1.npy')
                else:
                    Laplacian_params = L.Laplacian_loss_parameters(mesh_generator.num_vertices + \
                    mesh_generator.num_vertices_1, mesh_generator.faces_1.tolist(), 'Laplacian_params_devide_1.npy')
            if batches_done == iter_divide_2:
                print('initializing subdivied-2 mesh params...')
                mesh_generator.init_order2()
                if os.path.isfile('smoothness_params_devide_2.npy'):
                    smoothness_params = np.load('smoothness_params_devide_2.npy')
                else:
                    smoothness_params = L.smoothness_loss_parameters(mesh_generator.faces_2, \
                                                                     'smoothness_params_devide_2.npy')
                if os.path.isfile('Laplacian_params_devide_2.npy'):
                    Laplacian_params = np.load('Laplacian_params_devide_2.npy')
                else:
                    Laplacian_params = L.Laplacian_loss_parameters(mesh_generator.num_vertices + \
                    mesh_generator.num_vertices_1 + mesh_generator.num_vertices_2, \
                    mesh_generator.faces_2.tolist(), 'Laplacian_params_devide_2.npy')

            imgs = Variable(imgs.to(device))
            imgs = imgs[:,3,:,:]
            imgs = imgs.reshape((imgs.shape[0],1,opt.img_size,opt.img_size))
            real_imgs = imgs

            if opt.conditioned == 1:
                labels = torch.zeros((imgs.shape[0],24,opt.img_size,opt.img_size), dtype=torch.float)
                labels = Variable(labels.to(device))
                for ii in range(imgs.shape[0]):
                    labels[ii, viewids[ii], :, :] = 1.
                real_imgs = torch.cat((real_imgs, labels), 1)     # images conditioned on viewpoints.
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1, device=device).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1, device=device).fill_(0.0), requires_grad=False)



            losses = {}

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # optimizer_D.zero_grad()

            # Sample random noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)), device=device))

            # Generate a batch of images
            if batches_done < iter_divide_1:
                vertices, faces = mesh_generator(z)
            elif batches_done >= iter_divide_1 and batches_done < iter_divide_2:
                vertices, faces = mesh_generator(z, 1)
            elif batches_done >= iter_divide_2:
                vertices, faces = mesh_generator(z, 2)
            if batches_done == iter_divide_1:
                vertices_low, faces_low = mesh_generator(z)
                nr.save_obj(os.path.join(opt.sample_dir, 'divide_low_1.obj'), vertices_low[0, :, :], faces_low[0, :, :])
                nr.save_obj(os.path.join(opt.sample_dir, 'divide_high_1.obj'), vertices[0, :, :], faces[0, :, :])
            if batches_done == iter_divide_2:
                vertices_low, faces_low = mesh_generator(z, 1)
                nr.save_obj(os.path.join(opt.sample_dir, 'divide_low_2.obj'), vertices_low[0, :, :], faces_low[0, :, :])
                nr.save_obj(os.path.join(opt.sample_dir, 'divide_high_2.obj'), vertices[0, :, :], faces[0, :, :])

            mesh_renderer = M.Mesh_Renderer(vertices, faces, opt.img_size).cuda(opt.device_id)
            imgs_nr, viewids = mesh_renderer()
            gen_imgs = imgs_nr.detach()

            if opt.conditioned == 1:
                viewids = viewids.detach()
                labels = torch.zeros((gen_imgs.shape[0], 24, opt.img_size, opt.img_size), dtype=torch.float)
                labels = Variable(labels.to(device))
                for ii in range(gen_imgs.shape[0]):
                    labels[ii, int(viewids[ii]), :, :] = 1.
                gen_imgs = torch.cat((gen_imgs, labels), 1)  # images conditioned on viewpoints.

            # Measure discriminator's ability to classify real from generated samples
            p_real, _ = discriminator(real_imgs)
            p_fake, _ = discriminator(gen_imgs)
            if opt.model == 'DCGAN':
                real_loss = F.binary_cross_entropy(p_real, valid)
                fake_loss = F.binary_cross_entropy(p_fake, fake)
                acc = torch.Tensor.float(torch.sum(p_real>0.5)+torch.sum(p_fake<0.5))/(2*imgs.shape[0])
            elif opt.model in ('WGAN', 'WGAN-GP'):
                real_loss = -torch.mean(p_real)
                fake_loss = torch.mean(p_fake)

            skipD = False
            if opt.model == 'DCGAN':
                d_loss = (real_loss + fake_loss) / 2
                losses['D/loss_real'] = real_loss.item()
                losses['D/loss_fake'] = fake_loss.item()
                losses['D/loss_mean'] = d_loss.item()
                losses['D/accuracy'] = acc.item()
                if acc > 1:  # only train D when acc is lower than a threshold
                    skipD = True
            elif opt.model == 'WGAN':
                d_loss = real_loss + fake_loss
                losses['D/logit_real'] = -real_loss.item()
                losses['D/logit_fake'] = fake_loss.item()
                losses['D/Wasserstain-D'] = -d_loss.item()
            elif opt.model == 'WGAN-GP':
                gp = gradient_penalty(real_imgs.data, gen_imgs.data, discriminator)
                d_loss = real_loss + fake_loss + opt.gp * gp
                losses['D/logit_real'] = -real_loss.item()
                losses['D/logit_fake'] = fake_loss.item()
                losses['D/Wasserstain-D'] = -(real_loss + fake_loss).item()
                losses['D/gp'] = opt.gp * gp
                if losses['D/Wasserstain-D'] >1e6 :
                    skipD = True

            if skipD:    # only train D when acc is lower than a threshold
                print('skipping D training...')
                pass
            else:
                discriminator.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            if opt.model == 'WGAN':
                # clip weights for WGAN D weights
                for parameter in discriminator.parameters():
                    parameter.data.clamp_(-opt.clip_value, opt.clip_value)

            # -----------------
            #  Train Generator
            # -----------------
            if (epoch * len(dataloader) + i + 1) % opt.G_every == 0:

                # optimizer_G.zero_grad()
                z = z[0:int(opt.batch_size/4), :]
                if batches_done < iter_divide_1:
                    vertices, faces = mesh_generator(z)
                elif batches_done >= iter_divide_1 and batches_done < iter_divide_2:
                    vertices, faces = mesh_generator(z, 1)
                elif batches_done >= iter_divide_2:
                    vertices, faces = mesh_generator(z, 2)
                mesh_renderer = M.Mesh_Renderer(vertices, faces, opt.img_size).cuda(opt.device_id)
                gen_imgs1, viewids1 = mesh_renderer()
                gen_imgs2, viewids2 = mesh_renderer()   # two random views
                gen_imgs3, viewids3 = mesh_renderer()
                gen_imgs4, viewids4 = mesh_renderer()
                imgs_nr = torch.cat((gen_imgs1, gen_imgs2, gen_imgs3, gen_imgs4), 0)
                gen_imgs = imgs_nr

                if opt.conditioned == 1:
                    viewids = torch.cat((viewids1, viewids2, viewids3, viewids4), 0)
                    labels = torch.zeros((gen_imgs.shape[0], 24, opt.img_size, opt.img_size), dtype=torch.float)
                    labels = Variable(labels.to(device))
                    for ii in range(gen_imgs.shape[0]):
                        labels[ii, int(viewids[ii]), :, :] = 1.
                    gen_imgs = torch.cat((gen_imgs, labels), 1)  # images conditioned on viewpoints.
                # Loss measures generator's ability to fool the discriminator
                _, f_real = discriminator(real_imgs)
                p_fake, f_fake = discriminator(gen_imgs)
                if opt.model == 'DCGAN':
                    g_loss = F.binary_cross_entropy(p_fake, valid)
                    losses['G/loss_fake'] = g_loss.item()
                elif opt.model in ('WGAN', 'WGAN-GP'):
                    g_loss = -torch.mean(p_fake)
                    losses['G/logit_fake'] = -g_loss.item()
                # feature statistics matching loss
                feat_matching_loss = torch.norm(torch.mean(f_real, 0) - torch.mean(f_fake, 0)) + \
                                     torch.norm(torch.std(f_real, 0) - torch.std(f_fake, 0))
                losses['G/featMatching_loss'] = feat_matching_loss
                # mesh smoothness loss:
                smth_loss = L.smoothness_loss(vertices, smoothness_params)
                losses['G/smoothness_loss'] = smth_loss
                # Laplacian loss:
                Lap_loss, edge_loss = L.Laplacian_edge_loss(vertices, Laplacian_params)
                losses['G/Laplacian_loss'] = Lap_loss
                losses['G/edge_loss'] = edge_loss

                g_loss_total = g_loss + opt.lambda_smth*smth_loss + opt.lambda_feat*feat_matching_loss + \
                    opt.lambda_Lap*Lap_loss + opt.lambda_edge*edge_loss

                discriminator.zero_grad()
                mesh_renderer.zero_grad()
                mesh_generator.zero_grad()
                imgs_nr.retain_grad()
                g_loss_total.backward()
                optimizer_G.step()

            batches_done = opt.batches_done + epoch * len(dataloader) + i + 1
            if batches_done == 1:
                save_image(imgs.data[:25], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=5,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item()))
                if opt.use_tensorboard:
                    for tag, value in losses.items():
                        logger.scalar_summary(tag, value, batches_done)

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(imgs_nr.data[:25], os.path.join(opt.sample_dir, 'random-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                save_image(imgs_nr.grad[:25], os.path.join(opt.sample_dir, 'random-%05d-grad.png' % batches_done), nrow=5,
                               normalize=True)
                if batches_done-1 < iter_divide_1:
                    vertices_fixed, faces_fixed = mesh_generator(z_fixed)
                elif batches_done-1 >= iter_divide_1 and batches_done-1 < iter_divide_2:
                    vertices_fixed, faces_fixed = mesh_generator(z_fixed, 1)
                elif batches_done-1 >= iter_divide_2:
                    vertices_fixed, faces_fixed = mesh_generator(z_fixed, 2)
                mesh_renderer = M.Mesh_Renderer(vertices_fixed, faces_fixed, opt.img_size).cuda(opt.device_id)
                gen_imgs_fixed = mesh_renderer(viewpoints_fixed)
                save_image(gen_imgs_fixed.data, os.path.join(opt.sample_dir, 'fixed-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                nr.save_obj(os.path.join(opt.sample_dir, 'random-%05d.obj' % batches_done), vertices[0,:,:], faces[0,:,:])
                nr.save_obj(os.path.join(opt.sample_dir, 'fixed-%05d.obj' % batches_done), vertices_fixed[0, :, :],
                            faces_fixed[0, :, :])
                print('Saved sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, '{}-G.ckpt'.format(batches_done)))
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, '{}-D.ckpt'.format(batches_done)))
                print('Saved model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))

elif opt.mode == 'trainAE':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # Initialize encoder and decoder
    encoder = M.Encoder(4, dim_out=opt.latent_dim, VAE=opt.use_VAE)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)

    if os.path.isfile('smoothness_params_642.npy'):
        smoothness_params = np.load('smoothness_params_642.npy')
    else:
        smoothness_params = L.smoothness_loss_parameters(mesh_generator.faces, 'smoothness_params_642.npy')

    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)

    # Initialize weights
    if opt.load_G is None:
        # mesh_generator.apply(weights_init_normal)
        pass
    else:
        mesh_generator.load_state_dict(torch.load(opt.load_G))

    if opt.load_E is None:
        # encoder.apply(weights_init_normal)
        pass
    else:
        encoder.load_state_dict(torch.load(opt.load_E))

    # Configure data loader
    dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train')
    batch_sampler = data_loader.ShapeNet_Sampler_Batch(dataset_train, opt.batch_size)
    dataloader = data.DataLoader(dataset_train, batch_sampler=batch_sampler)

    # Optimizers
    optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(batch_sampler) + opt.batches_done
    lambda_KLD = 1e-4
    batches_done = opt.batches_done
    for epoch in range(opt.n_epochs):
        for i, (imgs, viewpoints, _, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.to(device))
            viewpoints = Variable(viewpoints.to(device))

            if not opt.use_VAE:
                z = encoder(real_imgs)
            else:
                z, x_mu, x_logvar = encoder(real_imgs)
            # Generate a batch of images
            vertices, faces = mesh_generator(z)
            mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

            gen_imgs1 = mesh_renderer(viewpoints)
            gt_imgs1 = real_imgs[:,3,:,:]
            gt_imgs1 = gt_imgs1.reshape((opt.batch_size,1,opt.img_size,opt.img_size))

            smth_loss = L.smoothness_loss(vertices, smoothness_params)
            iou_loss1 = L.iou_loss(gt_imgs1, gen_imgs1)
            iou_loss = iou_loss1
            if not opt.use_VAE:
                total_loss = iou_loss + opt.lambda_smth * smth_loss
            else:
                KLD = -0.5 * torch.sum(1+x_logvar-x_mu.pow(2)-x_logvar.exp())
                # if batches_done % 500 == 0 and batches_done != 0:
                #     lambda_KLD += 0.01
                #     lambda_KLD = min(lambda_KLD, 0.1)
                #     print('lambda_KLD changed: %f' % lambda_KLD)
                total_loss = iou_loss + opt.lambda_smth * smth_loss + lambda_KLD*KLD

            encoder.zero_grad()
            mesh_generator.zero_grad()
            mesh_renderer.zero_grad()
            gen_imgs1.retain_grad()
            total_loss.backward()
            optimizer_E.step()
            optimizer_G.step()

            batches_done = opt.batches_done + epoch * len(dataloader) + i + 1
            if batches_done == 1:
                save_image(real_imgs.data[:,0:3,:,:], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=8,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [total loss: %f] [iou loss: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i, len(dataloader), total_loss.item(), iou_loss.item()))

                ploter.plot('IoU_loss', 'train', 'IoU-loss', batches_done, iou_loss.item())
                ploter.plot('smoothness_loss', 'train', 'smoothness-loss', batches_done, smth_loss.item())
                ploter.plot('total_loss', 'train', 'total-loss', batches_done, total_loss.item())
                if opt.use_VAE:
                    ploter.plot('KLD', 'train', 'KLD', batches_done, KLD.item())

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs1.data[:25], os.path.join(opt.sample_dir, 'random-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                save_image(gen_imgs1.grad[:25], os.path.join(opt.sample_dir, 'random-%05d-grad.png' % batches_done), nrow=5,
                               normalize=True)
                nr.save_obj(os.path.join(opt.sample_dir, 'random-%05d.obj' % batches_done), vertices[0,:,:], faces[0,:,:])
                print('Saved sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, '{}-E.ckpt'.format(batches_done)))
                torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, '{}-G.ckpt'.format(batches_done)))
                torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'last-E.ckpt'))
                torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'last-G.ckpt'))
                print('Saved model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_E.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))

        # if epoch + 1 - 2 >= 0 and (epoch + 1 - opt.decay_epoch) % 2 == 0:
        #     opt.lambda_smth = opt.lambda_smth * opt.decay_order
        #     print('lambda_smth decayed to {}'.format(opt.lambda_smth))

elif opt.mode == 'sampleGAN':
    os.makedirs(opt.sample_dir, exist_ok=True)

    # Initialize generator and discriminator
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    if cuda:
        mesh_generator.cuda(opt.device_id)

    try:
        mesh_generator.load_state_dict(torch.load(opt.load_G))
    except:
        print('checkpoint dir of Generator is not provided!')
        exit()

    for i_batch in range(opt.n_samples//opt.batch_size):
        print('generating the %d th sample batch, total %d batches...'%(i_batch, opt.n_samples//opt.batch_size))
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)), device=device))
        vertices, faces = mesh_generator(z, 1)
        for i in range(opt.batch_size):
            nr.save_obj(os.path.join(opt.sample_dir, '%s-%05d.obj' % (opt.sample_prefix, i_batch*opt.batch_size+i)),
                        vertices[i, :, :], faces[i, :, :])

elif opt.mode == 'trainAE_GAN':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # Initialize encoder and decoder and discriminator
    encoder = M.Encoder(4, dim_out=opt.latent_dim, VAE=opt.use_VAE)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    discriminator = M.DCGAN_Discriminator(opt.img_size, opt.channels, opt.model)

    if os.path.isfile('smoothness_params_642.npy'):
        smoothness_params = np.load('smoothness_params_642.npy')
    else:
        smoothness_params = L.smoothness_loss_parameters(mesh_generator.faces)

    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)
        discriminator.cuda(opt.device_id)

    # Initialize weights
    if opt.load_G is None:
        pass
    else:
        mesh_generator.load_state_dict(torch.load(opt.load_G))
    if opt.load_E is None:
        pass
    else:
        encoder.load_state_dict(torch.load(opt.load_E))
    if opt.load_D is None:
        pass
    else:
        discriminator.load_state_dict(torch.load(opt.load_D))

    # Configure data loader
    dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train')
    batch_sampler = data_loader.ShapeNet_Sampler_Batch(dataset_train, opt.batch_size)
    dataloader = data.DataLoader(dataset_train, batch_sampler=batch_sampler)

    # Optimizers
    optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(batch_sampler) + opt.batches_done
    lambda_KLD = 1e-4
    lambda_D = 1e-3
    batches_done = opt.batches_done
    for epoch in range(opt.n_epochs):
        for i, (imgs, viewpoints, viewids_real, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.to(device))
            viewpoints = Variable(viewpoints.to(device))

            # -----------------
            #  Train Generator, Encoder and Discriminator
            # -----------------
            if not opt.use_VAE:
                z = encoder(real_imgs)
            else:
                z, x_mu, x_logvar = encoder(real_imgs)
            # Generate a batch of images
            vertices, faces = mesh_generator(z)
            mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

            gen_imgs = mesh_renderer(viewpoints)
            gt_imgs = real_imgs[:,3,:,:]
            gt_imgs = gt_imgs.reshape((opt.batch_size,1,opt.img_size,opt.img_size))

            smth_loss = L.smoothness_loss(vertices, smoothness_params)
            iou_loss = L.iou_loss(gt_imgs, gen_imgs)

            imgs_newView, viewids_fake = mesh_renderer()
            imgs_newView = imgs_newView.detach()
            labels_fake = torch.zeros((imgs_newView.shape[0], 24, opt.img_size, opt.img_size), dtype=torch.float)
            labels_fake = Variable(labels_fake.to(device))
            labels_real = torch.zeros((gt_imgs.shape[0], 24, opt.img_size, opt.img_size), dtype=torch.float)
            labels_real = Variable(labels_real.to(device))
            for ii in range(imgs_newView.shape[0]):
                labels_fake[ii, int(viewids_fake[ii]), :, :] = 1.
                labels_real[ii, int(viewids_real[ii]), :, :] = 1.
            imgs_newView_label = torch.cat((imgs_newView, labels_fake), 1)  # images conditioned on viewpoints.
            gt_imgs_label = torch.cat((gt_imgs, labels_real), 1)

            p_fake, _ = discriminator(imgs_newView_label)
            p_real, _ = discriminator(gt_imgs_label)
            WassersteinD = torch.mean(p_real) - torch.mean(p_fake)
            gp = gradient_penalty(gt_imgs_label.data, imgs_newView_label.data, discriminator)
            d_loss = -WassersteinD + opt.gp * gp
            trainD = 1.
            if WassersteinD > 100:
                trainD = 0.

            if not opt.use_VAE:
                total_loss = iou_loss + opt.lambda_smth * smth_loss + trainD * d_loss
            else:
                KLD = -0.5 * torch.sum(1+x_logvar-x_mu.pow(2)-x_logvar.exp())
                # if batches_done % 500 == 0 and batches_done != 0:
                #     lambda_KLD += 0.01
                #     lambda_KLD = min(lambda_KLD, 0.1)
                #     print('lambda_KLD changed: %f' % lambda_KLD)
                total_loss = iou_loss + opt.lambda_smth * smth_loss + lambda_KLD*KLD + trainD * d_loss

            discriminator.zero_grad()
            mesh_renderer.zero_grad()
            mesh_generator.zero_grad()
            encoder.zero_grad()
            gen_imgs.retain_grad()
            total_loss.backward()
            optimizer_D.step()
            optimizer_G.step()
            optimizer_E.step()

            # Train Generator and Encoder with Discriminator backprop
            if (epoch * len(dataloader) + i + 1) % opt.G_every == 0 and epoch >= 1:
                if not opt.use_VAE:
                    z = encoder(real_imgs)
                else:
                    z, x_mu, x_logvar = encoder(real_imgs)
                # Generate a batch of images
                vertices, faces = mesh_generator(z)
                mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

                imgs_newView, viewids_fake = mesh_renderer()
                labels_fake = torch.zeros((imgs_newView.shape[0], 24, opt.img_size, opt.img_size), dtype=torch.float)
                labels_fake = Variable(labels_fake.to(device))
                for ii in range(imgs_newView.shape[0]):
                    labels_fake[ii, int(viewids_fake[ii]), :, :] = 1.
                imgs_newView_label = torch.cat((imgs_newView, labels_fake), 1)  # images conditioned on viewpoints.

                p_fake, _ = discriminator(imgs_newView_label)
                g_loss = - lambda_D * torch.mean(p_fake)
                discriminator.zero_grad()
                mesh_renderer.zero_grad()
                mesh_generator.zero_grad()
                encoder.zero_grad()
                imgs_newView.retain_grad()
                g_loss.backward()
                optimizer_G.step()
                optimizer_E.step()

            batches_done = opt.batches_done + epoch * len(dataloader) + i + 1
            if batches_done == 1:
                save_image(imgs.data[:,0:3,:,:], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=8,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [iou loss: %f] [W_D: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i, len(dataloader), iou_loss.item(), WassersteinD.item()))

                ploter.plot('IoU_loss', 'train', 'IoU-loss', batches_done, iou_loss.item())
                ploter.plot('smoothness_loss', 'train', 'smoothness-loss', batches_done, smth_loss.item())
                ploter.plot('WassersteinD', 'train', 'WassersteinD', batches_done, WassersteinD.item())
                ploter.plot('gp', 'train', 'gradient-penalty', batches_done, gp.item())
                if opt.use_VAE:
                    ploter.plot('KLD', 'train', 'KLD', batches_done, KLD.item())

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs.data[:25], os.path.join(opt.sample_dir, 'random-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                save_image(gen_imgs.grad[:25], os.path.join(opt.sample_dir, 'random-%05d-gradRecon.png' % batches_done),
                           nrow=5, normalize=True)
                if imgs_newView.grad is not None:
                    save_image(imgs_newView.grad[:25], os.path.join(opt.sample_dir, 'random-%05d-gradGAN.png' % batches_done),
                               nrow=5, normalize=True)
                nr.save_obj(os.path.join(opt.sample_dir, 'random-%05d.obj' % batches_done), vertices[0,:,:], faces[0,:,:])
                print('Saved sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, '{}-E.ckpt'.format(batches_done)))
                torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, '{}-G.ckpt'.format(batches_done)))
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, '{}-D.ckpt'.format(batches_done)))
                torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'last-E.ckpt'))
                torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'last-G.ckpt'))
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, 'last-D.ckpt'))
                print('Saved model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_E.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))

        # if epoch + 1 - 2 >= 0 and (epoch + 1 - opt.decay_epoch) % 2 == 0:
        #     opt.lambda_smth = opt.lambda_smth * opt.decay_order
        #     print('lambda_smth decayed to {}'.format(opt.lambda_smth))

elif opt.mode == 'evaluation':

    encoder = M.Encoder(4, dim_out=opt.latent_dim, VAE=opt.use_VAE)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)

    # Initialize weights
    mesh_generator.load_state_dict(torch.load(opt.load_G))
    encoder.load_state_dict(torch.load(opt.load_E))
    mesh_generator.eval()
    encoder.eval()

    # Configure data loader
    dataset_test = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'test')
    with torch.no_grad():
        for class_id in opt.class_ids.split(','):
            dataloader = data.DataLoader(dataset_test, batch_sampler=
            data_loader.ShapeNet_sampler_all(dataset_test, opt.batch_size, class_id))
            iou = 0
            ious = {}
            print('%s_%s has %d images, %d batches...' % (dataset_test.set_name, class_id,
                                                       dataset_test.num_data[class_id]*24, len(dataloader)))
            for i, (imgs, _, _, voxels) in enumerate(dataloader):
                real_imgs = Variable(imgs.to(device))
                if not opt.use_VAE:
                    z = encoder(real_imgs)
                else:
                    z,_,_ = encoder(real_imgs)
                vertices, faces = mesh_generator(z)
                faces = nr.vertices_to_faces(vertices, faces).data
                faces = faces * 1. * (32. - 1) / 32. + 0.5  # normalization
                voxels_predicted = voxelization.voxelize(faces, 32, False)
                voxels_predicted = voxels_predicted.transpose(1, 2).flip([3])
                iou_batch = torch.Tensor.float(voxels * voxels_predicted.cpu()).sum((1, 2, 3)) / \
                            torch.Tensor.float(0 < (voxels + voxels_predicted.cpu())).sum((1, 2, 3))
                iou += iou_batch.sum()
            iou /= dataset_test.num_data[class_id] * 24.
            print('%s/iou_%s: %f' % (dataset_test.set_name, class_id, iou.item()))
            ious['%s/iou_%s' % (dataset_test.set_name, class_id)] = iou.item()
        iou_mean = np.mean([float(v) for v in ious.values()])
        ious['%s/iou' % dataset_test.set_name] = iou_mean
        print('%s/iou: %f' % (dataset_test.set_name, iou_mean))

elif opt.mode == 't_SNE':
    # Initialize encoder and decoder
    encoder = M.Encoder(4, dim_out=opt.latent_dim, VAE=opt.use_VAE)
    if cuda:
        encoder.cuda(opt.device_id)

    # Initialize weights
    encoder.load_state_dict(torch.load(opt.load_E))
    encoder.eval()

    # Configure data loader
    dataset_val = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'val')
    dataloader = data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    features = np.zeros((len(dataset_val), opt.latent_dim))
    labels = np.zeros(len(dataset_val))
    # ----------
    #  forward
    # ----------
    for epoch in range(1):
        for i, (imgs, _, view_ids, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.to(device))

            if not opt.use_VAE:
                z = encoder(real_imgs)
            else:
                z, x_mu, x_logvar = encoder(real_imgs)

            z_cpu = torch.squeeze(z.cpu())
            features[(i * opt.batch_size):min((i + 1) * opt.batch_size, len(dataset_val)),:] = \
                z_cpu.data.numpy()
            labels[(i * opt.batch_size):min((i + 1) * opt.batch_size, len(dataset_val))] = np.squeeze(view_ids)

    print('features shape is: ', features.shape)
    print('labels shape is: ', labels.shape)
    print('doing PCA dimension reduction ...')
    features_PCA = PCA(n_components=50).fit_transform(features)
    print('doing t-SNE ...')
    features_TSNE = TSNE(n_components=2, init='pca').fit_transform(features_PCA)
    plt.figure()
    plt.scatter(features_TSNE[:, 0], features_TSNE[:, 1], c=labels)
    plt.show()
    plt.colorbar()
    plt.draw()

elif opt.mode == 'trainAE_featGAN':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # Initialize encoder and decoder and discriminator
    encoder = M.Encoder(4, dim_out=opt.latent_dim, VAE=opt.use_VAE)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    discriminator = M.feat_Discriminator(opt.latent_dim + 24)

    if os.path.isfile('smoothness_params_642.npy'):
        smoothness_params = np.load('smoothness_params_642.npy')
    else:
        smoothness_params = L.smoothness_loss_parameters(mesh_generator.faces)

    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)
        discriminator.cuda(opt.device_id)

    # Initialize weights
    if opt.load_G is None:
        pass
    else:
        mesh_generator.load_state_dict(torch.load(opt.load_G))
    if opt.load_E is None:
        pass
    else:
        encoder.load_state_dict(torch.load(opt.load_E))
    if opt.load_D is None:
        pass
    else:
        discriminator.load_state_dict(torch.load(opt.load_D))

    # Configure data loader
    dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train')
    batch_sampler = data_loader.ShapeNet_Sampler_Batch(dataset_train, opt.batch_size)
    dataloader = data.DataLoader(dataset_train, batch_sampler=batch_sampler)

    # Optimizers
    optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(batch_sampler) + opt.batches_done
    lambda_KLD = 1e-4
    lambda_D = 1e-4
    batches_done = opt.batches_done
    for epoch in range(opt.n_epochs):
        for i, (imgs, viewpoints, viewids_real, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.to(device))
            viewpoints = Variable(viewpoints.to(device))

            # -----------------
            #  Train Generator, Encoder and Discriminator
            # -----------------
            if not opt.use_VAE:
                z = encoder(real_imgs)
            else:
                z, x_mu, x_logvar = encoder(real_imgs)
            # Generate a batch of images
            vertices, faces = mesh_generator(z)
            mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

            gen_imgs = mesh_renderer(viewpoints)
            gt_imgs = real_imgs[:,3,:,:]
            gt_imgs = gt_imgs.reshape((opt.batch_size,1,opt.img_size,opt.img_size))

            smth_loss = L.smoothness_loss(vertices, smoothness_params)
            iou_loss = L.iou_loss(gt_imgs, gen_imgs)

            z_detach = z.detach()
            real_labels = torch.zeros((z.shape[0], 24), dtype=torch.float)
            fake_labels = torch.zeros((z.shape[0], 24), dtype=torch.float)
            for ii in range(real_labels.shape[0]):
                real_labels[ii, int(viewids_real[ii])] = 1.
                fake_choices = np.concatenate((np.arange(int(viewids_real[ii])), np.arange(int(viewids_real[ii])+1, 24)))
                rand_fake = np.random.choice(fake_choices)
                fake_labels[ii, rand_fake] = 1.

            real_labels = Variable(real_labels.to(device))
            feat_real = torch.cat((z_detach, real_labels), 1)
            fake_labels = Variable(fake_labels.to(device))
            feat_fake = torch.cat((z_detach, fake_labels), 1)
            p_real, _ = discriminator(feat_real)
            p_fake, _ = discriminator(feat_fake)
            WassersteinD = torch.mean(p_real) - torch.mean(p_fake)
            gp = gradient_penalty(feat_real.data, feat_fake.data, discriminator)
            d_loss = -WassersteinD + opt.gp * gp

            if not opt.use_VAE:
                total_loss = iou_loss + opt.lambda_smth * smth_loss + d_loss
            else:
                KLD = -0.5 * torch.sum(1+x_logvar-x_mu.pow(2)-x_logvar.exp())
                # if batches_done % 500 == 0 and batches_done != 0:
                #     lambda_KLD += 0.01
                #     lambda_KLD = min(lambda_KLD, 0.1)
                #     print('lambda_KLD changed: %f' % lambda_KLD)
                total_loss = iou_loss + opt.lambda_smth * smth_loss + lambda_KLD*KLD + d_loss

            discriminator.zero_grad()
            mesh_renderer.zero_grad()
            mesh_generator.zero_grad()
            encoder.zero_grad()
            gen_imgs.retain_grad()
            total_loss.backward()
            optimizer_D.step()
            optimizer_G.step()
            optimizer_E.step()

            # Train Encoder with feat_Discriminator backprop
            if (epoch * len(dataloader) + i + 1) % opt.G_every == 0:
                if not opt.use_VAE:
                    z = encoder(real_imgs)
                else:
                    z, x_mu, x_logvar = encoder(real_imgs)

                fake_labels = torch.zeros((z.shape[0], 24), dtype=torch.float)
                for ii in range(real_labels.shape[0]):
                    fake_choices = np.concatenate(
                        (np.arange(int(viewids_real[ii])), np.arange(int(viewids_real[ii]) + 1, 24)))
                    rand_fake = np.random.choice(fake_choices)
                    fake_labels[ii, rand_fake] = 1.

                fake_labels = Variable(fake_labels.to(device))
                feat_fake = torch.cat((z_detach, fake_labels), 1)
                p_fake, _ = discriminator(feat_fake)
                d_loss = - lambda_D * torch.mean(p_fake)

                optimizer_E.zero_grad()
                d_loss.backward()
                optimizer_E.step()

            batches_done = opt.batches_done + epoch * len(dataloader) + i + 1
            if batches_done == 1:
                save_image(imgs.data[:,0:3,:,:], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=8,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [iou loss: %f] [W_D: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i, len(dataloader), iou_loss.item(), WassersteinD.item()))

                ploter.plot('IoU_loss', 'train', 'IoU-loss', batches_done, iou_loss.item())
                ploter.plot('smoothness_loss', 'train', 'smoothness-loss', batches_done, smth_loss.item())
                ploter.plot('WassersteinD', 'train', 'WassersteinD', batches_done, WassersteinD.item())
                ploter.plot('gp', 'train', 'gradient-penalty', batches_done, gp.item())
                if opt.use_VAE:
                    ploter.plot('KLD', 'train', 'KLD', batches_done, KLD.item())

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs.data[:25], os.path.join(opt.sample_dir, 'random-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                save_image(gen_imgs.grad[:25], os.path.join(opt.sample_dir, 'random-%05d-grad.png' % batches_done),
                           nrow=5, normalize=True)
                nr.save_obj(os.path.join(opt.sample_dir, 'random-%05d.obj' % batches_done), vertices[0,:,:], faces[0,:,:])
                print('Saved sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, '{}-E.ckpt'.format(batches_done)))
                torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, '{}-G.ckpt'.format(batches_done)))
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, '{}-D.ckpt'.format(batches_done)))
                torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'last-E.ckpt'))
                torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'last-G.ckpt'))
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, 'last-D.ckpt'))
                print('Saved model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_E.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))