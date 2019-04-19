import argparse
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import time
import datetime
import models as M
import data_loader
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn.functional as F
import os
import neural_renderer as nr
import losses as L


def str2bool(v):
    return v.lower() in ('true')


CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
DATASET_DIRECTORY = '/hd2/pengbo/mesh_reconstruction/dataset/'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='trainGAN', help='mode can be one of [trainGAN, trainD, testD, trainG]')
parser.add_argument('--data_dir', type=str, help='dir of dataset')
parser.add_argument('--crop_size', type=int, default=178, help='size of center crop for celebA')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='number of epochs before lr decay')
parser.add_argument('--decay_order', type=float, default=0.1, help='order of lr decay')
parser.add_argument('--decay_every', type=int, default=5, help='lr decay every n epochs')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_step', type=int, default=1, help='number of iters between image sampling')
parser.add_argument('--sample_dir', type=str, help='dir of saved '
                                                                                                'sample images')
parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='whether to use tensorboard for monitoring')
parser.add_argument('--log_dir', type=str, help='dir of '
                                                                                             'tensorboard logs')
parser.add_argument('--log_step', type=int, default=10, help='number of iters to print and log')
parser.add_argument('--ckpt_step', type=int, default=1000, help='number of iters for model saving')
parser.add_argument('--ckpt_dir', type=str, help='dir of saved model '
                                                                                                'checkpoints')
parser.add_argument('--load_G', type=str, default=None, help='path of to the loaded Generator weights')
parser.add_argument('--load_D', type=str, default=None, help='path of to the loaded Discriminator weights')
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

t_start = time.time()
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = 'cuda:%d' % opt.device_id if opt.device_id >= 0 else 'cpu'
print(device)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device=device)
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True)
    o = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size(), device=device),
                            create_graph=True, retain_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


if opt.use_tensorboard:
    os.makedirs(opt.log_dir, exist_ok=True)
    from logger import Logger

    logger = Logger(opt.log_dir)

if opt.mode == 'trainGAN':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # Initialize generator and discriminator
    img_generator = M.DCGAN_Generator(opt.img_size, 1, opt.latent_dim)
    discriminator = M.DCGAN_Discriminator(opt.img_size, opt.channels, opt.model)

    if cuda:
        img_generator.cuda(opt.device_id)
        discriminator.cuda(opt.device_id)

    # Initialize weights
    if opt.load_G is None:
        img_generator.apply(weights_init_normal)
    else:
        img_generator.load_state_dict(torch.load(opt.load_G))

    if opt.load_D is None:
        discriminator.apply(weights_init_normal)
    else:
        discriminator.load_state_dict(torch.load(opt.load_D))

    # Configure data loader
    dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train')
    dataloader = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    if opt.model in ('DCGAN', 'WGAN-GP'):
        optimizer_G = torch.optim.Adam(img_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    elif opt.model == 'WGAN':
        optimizer_G = torch.optim.RMSprop(img_generator.parameters(), lr=opt.lr)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    # Sample fixed random noise to see generated results
    z_fixed = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim - 24)), device=device))
    labels = torch.zeros((25, 24), dtype=torch.float)
    for ii in range(25):
        labels[ii, min(ii, 23)] = 1
    labels = Variable(labels.to(device))
    z_fixed_label = torch.cat((z_fixed, labels), 1)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(dataloader)
    for epoch in range(opt.n_epochs):
        for i, (imgs, _, viewids) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1, device=device).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1, device=device).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = imgs[:, 3, :, :]
            imgs = imgs.reshape((imgs.shape[0], 1, opt.img_size, opt.img_size))
            real_imgs = Variable(imgs.to(device))
            labels = torch.zeros((imgs.shape[0], 24, opt.img_size, opt.img_size), dtype=torch.float)
            for ii in range(imgs.shape[0]):
                labels[ii, viewids[ii], :, :] = 1.
            labels = Variable(labels.to(device))
            real_imgs_label = torch.cat((real_imgs, labels), 1)  # images conditioned on viewpoints.

            losses = {}

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # optimizer_D.zero_grad()

            # Sample random noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim-24)), device=device))
            labels = torch.zeros((imgs.shape[0], 24), dtype=torch.float)
            for ii in range(imgs.shape[0]):
                viewid = np.random.randint(0, 24, 1)
                labels[ii, viewid] = 1
            labels = Variable(labels.to(device))
            z_label = torch.cat((z, labels), 1)

            # Generate a batch of images
            gen_imgs = img_generator(z_label).detach()
            labels = labels[:,:,None,None]
            labels = labels.repeat((1,1,opt.img_size,opt.img_size))
            gen_imgs_labels = torch.cat((gen_imgs, labels), 1)

            # Measure discriminator's ability to classify real from generated samples
            if opt.model == 'DCGAN':
                p_real = discriminator(real_imgs_label)
                p_fake = discriminator(gen_imgs_labels)
                real_loss = F.binary_cross_entropy(p_real, valid)
                fake_loss = F.binary_cross_entropy(p_fake, fake)
                acc = torch.Tensor.float(torch.sum(p_real>0.5)+torch.sum(p_fake<0.5))/(2*imgs.shape[0]) # !!! without casting to float, it will always be 0 if not 1...
            elif opt.model in ('WGAN', 'WGAN-GP'):
                real_loss = -torch.mean(discriminator(real_imgs_label))
                fake_loss = torch.mean(discriminator(gen_imgs_labels))

            if opt.model == 'DCGAN':
                d_loss = (real_loss + fake_loss) / 2
                losses['D/loss_real'] = real_loss.item()
                losses['D/loss_fake'] = fake_loss.item()
                losses['D/loss_mean'] = d_loss.item()
                losses['D/accuracy'] = acc.item()
            elif opt.model == 'WGAN':
                d_loss = real_loss + fake_loss
                losses['D/logit_real'] = -real_loss.item()
                losses['D/logit_fake'] = fake_loss.item()
                losses['D/Wasserstain-D'] = -d_loss.item()
            elif opt.model == 'WGAN-GP':
                gp = gradient_penalty(real_imgs_label.data, gen_imgs_labels.data, discriminator)
                d_loss = real_loss + fake_loss + opt.gp * gp
                losses['D/logit_real'] = -real_loss.item()
                losses['D/logit_fake'] = fake_loss.item()
                losses['D/Wasserstain-D'] = -(real_loss + fake_loss).item()
                losses['D/gp'] = opt.gp * gp

            if acc>0.75:    # only train D when acc is lower than a threshold
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
                gen_imgs = img_generator(z_label)
                gen_imgs_labels = torch.cat((gen_imgs, labels), 1)

                # Loss measures generator's ability to fool the discriminator
                if opt.model == 'DCGAN':
                    g_loss = F.binary_cross_entropy(discriminator(gen_imgs_labels), valid)
                    losses['G/loss_fake'] = g_loss.item()
                elif opt.model in ('WGAN', 'WGAN-GP'):
                    g_loss = -torch.mean(discriminator(gen_imgs_labels))
                    losses['G/logit_fake'] = -g_loss.item()
                # mesh smoothness loss:

                discriminator.zero_grad()
                img_generator.zero_grad()
                g_loss.backward()
                optimizer_G.step()

            batches_done = opt.batches_done + epoch * len(dataloader) + i + 1
            if batches_done == 1:
                save_image(real_imgs.data[:25], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=5,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
                if opt.use_tensorboard:
                    for tag, value in losses.items():
                        logger.scalar_summary(tag, value, batches_done)

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs.data[:25], os.path.join(opt.sample_dir, 'random-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                gen_imgs_fixed = img_generator(z_fixed_label)
                save_image(gen_imgs_fixed.data, os.path.join(opt.sample_dir, 'fixed-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                print('Saved sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                torch.save(img_generator.state_dict(), os.path.join(opt.ckpt_dir, '{}-G.ckpt'.format(batches_done)))
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, '{}-D.ckpt'.format(batches_done)))
                print('Saved model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))
