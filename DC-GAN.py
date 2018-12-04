import argparse
import os
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

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='trainGAN', help='mode can be one of [trainGAN, trainD]')
parser.add_argument('--data_dir', type=str, default='/hd2/pengbo/StarGAN/data/CelebA_nocrop/', help='dir of dataset')
parser.add_argument('--crop_size', type=int, default=178, help='size of center crop for celebA')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='number of epochs before lr decay')
parser.add_argument('--decay_order', type=float, default=0.1, help='order of lr decay')
parser.add_argument('--decay_every', type=int, default=5, help='lr decay every n epochs')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=2000, help='number of iters between image sampling')
parser.add_argument('--sample_dir', type=str, default='/hd2/pengbo/StarGAN/DCGAN_celebA/sample', help='dir of saved '
                                                                                                       'sample images')
parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='whether to use tensorboard for monitoring')
parser.add_argument('--log_dir', type=str, default='/hd2/pengbo/StarGAN/DCGAN_celebA/log', help='dir of '
                                                                                                'tensorboard logs')
parser.add_argument('--log_step', type=int, default=100, help='number of iters to print and log')
parser.add_argument('--ckpt_step', type=int, default=5000, help='number of iters for model saving')
parser.add_argument('--ckpt_dir', type=str, default='/hd2/pengbo/StarGAN/DCGAN_celebA/model', help='dir of saved model '
                                                                                             'checkpoints')
parser.add_argument('--load_G', type=str, default=None, help='path of to the loaded Generator weights')
parser.add_argument('--load_D', type=str, default=None, help='path of to the loaded Discriminator weights')
parser.add_argument('--N_generated', type=int, default=1000, help='number of generated images')
parser.add_argument('--dir_generated', type=str, default=None, help='saving dir of to be generated images')

t_start = time.time()
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

if opt.use_tensorboard:
    os.makedirs(opt.log_dir, exist_ok=True)
    from logger import Logger
    logger = Logger(opt.log_dir)

if opt.mode =='trainGAN':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = M.DCGAN_Generator(opt.img_size, opt.channels, opt.latent_dim)
    discriminator = M.DCGAN_Discriminator(opt.img_size, opt.channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    dataloader = data_loader.get_CelebA_loader(opt.data_dir, opt.img_size, opt.crop_size, opt.batch_size, opt.n_cpu)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs*len(dataloader)
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            losses = {}
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            losses['G/loss_fake'] = g_loss.item()

            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            losses['D/loss_real'] = real_loss.item()
            losses['D/loss_fake'] = fake_loss.item()
            losses['D/loss_mean'] = d_loss.item()

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i + 1
            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now-t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
                if opt.use_tensorboard:
                    for tag, value in losses.items():
                        logger.scalar_summary(tag, value, batches_done)

            if batches_done % opt.sample_interval == 0 or batches_done == last_iter:
                save_image(gen_imgs.data[:25], os.path.join(opt.sample_dir, '%d.png' % batches_done), nrow=5,
                           normalize=True)
                print('Saved sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                torch.save(generator.state_dict(), os.path.join(opt.ckpt_dir, '{}-G.ckpt'.format(batches_done)))
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, '{}-D.ckpt'.format(batches_done)))
                print('Saved model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch+1-opt.decay_epoch >= 0 and (epoch+1-opt.decay_epoch)%opt.decay_every == 0:
            opt.lr = opt.lr*opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))

elif opt.mode == 'trainD':
    generator = M.DCGAN_Generator(opt.img_size, opt.channels, opt.latent_dim)
    discriminator = M.DCGAN_Discriminator(opt.img_size, opt.channels)
    BCEloss = torch.nn.BCELoss()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        BCEloss.cuda()

    # initialize G with trained weights, and D with random or trained weights
    if opt.load_G is None:
        exit('load_G is not provided!')
    generator.load_state_dict(torch.load(opt.load_G))
    generator.eval()
    if opt.load_D is None:
        discriminator.apply(weights_init_normal)
    else:
        discriminator.load_state_dict(torch.load(opt.load_D))

    # generate GAN images and make dataloader
    data_loader.generate(generator, opt.N_generated, opt.dir_generated, cuda, opt.batch_size, opt.latent_dim)
    transform0 = T.Compose([T.Resize(opt.img_size),
                           T.ToTensor(),
                           T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    GAN_data_train = data_loader.ImageFolderSingle(opt.dir_generated, 0.5, 'train', transform0, 0)

    transform1 = T.Compose([T.CenterCrop(opt.crop_size),
                            T.Resize(opt.img_size),
                            T.ToTensor(),
                            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    CelebA_data_train = data_loader.ImageFolderSingle(os.path.join(opt.data_dir, 'images'), 0.5, 'train', transform1, 1)

    mixed_data_train = data.ConcatDataset([GAN_data_train, CelebA_data_train])
    mixed_loader_train = data.DataLoader(mixed_data_train, batch_size=opt.batch_size, shuffle=True,
                                         num_workers=opt.n_cpu)

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1,opt.b2))
    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(mixed_loader_train)
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(mixed_loader_train):
            # Configure input
            imgs = Variable(imgs.type(Tensor))
            labels = Variable(labels.type(Tensor))
            losses = {}
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            predicts = discriminator(imgs)
            loss = BCEloss(predicts, labels)

            losses['D/loss_train'] = loss.item()
            labels_cpu = torch.squeeze(labels.cpu())
            predicts_cpu = torch.squeeze(predicts.cpu())
            pred_labels = torch.zeros_like(labels_cpu)
            pred_labels[predicts_cpu>0.5] = 1
            acc = torch.sum(pred_labels == labels_cpu).float()/len(labels_cpu)
            losses['D/acuracy_train'] = acc

            loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(mixed_loader_train) + i + 1
            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Accuracy: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i, len(mixed_loader_train), loss.item(), acc))
                if opt.use_tensorboard:
                    for tag, value in losses.items():
                        logger.scalar_summary(tag, value, batches_done)

            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, 'Forensics-D.ckpt'))
                print('Saved model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))