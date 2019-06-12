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
import re
import skimage.transform as skT
import pandas as pd

def str2bool(v):
    return v.lower() in ('true')


CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
DATASET_DIRECTORY = '/hd2/pengbo/mesh_reconstruction/dataset/'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='trainAE', help='mode can be one of [xxx]')
parser.add_argument('--AE_Gprior', action='store_true', default=False, help='use Gaussian prior on AE codes')
parser.add_argument('--AE_featMatch', action='store_true', default=False, help='use feature matching on AE codes')
parser.add_argument('--data_dir', type=str, help='dir of dataset')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='number of epochs before lr decay')
parser.add_argument('--decay_order', type=float, default=0.1, help='order of lr decay')
parser.add_argument('--decay_every', type=int, default=5, help='lr decay every n epochs')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_step', type=int, default=1, help='number of iters between image sampling')
parser.add_argument('--sample_dir', type=str, help='dir of saved sample images')
parser.add_argument('--log_step', type=int, default=10, help='number of iters to print and log')
parser.add_argument('--ckpt_step', type=int, default=1000, help='number of iters for model saving')
parser.add_argument('--ckpt_dir', type=str, help='dir of saved model checkpoints')
parser.add_argument('--visdom_env', type=str, default=None, help='Visdom environment name')

parser.add_argument('--load_G', type=str, default=None, help='path of to the loaded Generator weights')
parser.add_argument('--load_D', type=str, default=None, help='path of to the loaded Discriminator weights')
parser.add_argument('--load_E', type=str, default=None, help='path of to the loaded Encoder weights')
parser.add_argument('--load_im', type=str, default=None, help='image to load to reconstruct')
parser.add_argument('--load_im1', type=str, default=None, help='another image to load to interplolate')
parser.add_argument('--device_id', type=int, default=0, help='choose device ID')
parser.add_argument('--n_iters', type=int, default=3000, help='number of iterations for training')
parser.add_argument('--G_every', type=int, default=1, help='G:D training schedule is 1:G_every')
parser.add_argument('--batches_done', type=int, default=0, help='previous batches_done when '
                                                                'loading ckpt and continue traning')

parser.add_argument('--class_ids', type=str, default=CLASS_IDS_ALL, help='use which object class images')
parser.add_argument('--obj_dir', type=str, help='base sphere obj file path')
parser.add_argument('--lambda_smth', type=float, default=0.1, help='weight for mesh smoothness loss')
parser.add_argument('--lambda_Lap', type=float, default=0.1, help='weight for mesh Laplacian loss')
parser.add_argument('--lambda_edge', type=float, default=0.1, help='weight of edge length loss')
parser.add_argument('--lambda_Gprior', type=float, default=1., help='weight of Gaussian prior term')

parser.add_argument('--eval_flag', type=str, default='last', help='which ckpt to evaluate')


t_start = time.time()
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
device = 'cuda:%d' % opt.device_id if opt.device_id >= 0 else 'cpu'
print(device)
if opt.visdom_env is not None:
    ploter = logger.VisdomLinePlotter(env_name=opt.visdom_env)


def eval_IoU(encoder, mesh_generator, dataset_val):
    mesh_generator.eval()
    encoder.eval()
    with torch.no_grad():
        for class_id in opt.class_ids.split(','):
            loader_val = data.DataLoader(dataset_val, batch_sampler=
            data_loader.ShapeNet_sampler_all(dataset_val, opt.batch_size, class_id))
            iou = 0
            ious = {}
            print('%s_%s has %d images, %d batches...' % (dataset_val.set_name, class_id,
                                                          dataset_val.num_data[class_id] * 24, len(loader_val)))
            for i, (imgs, _, _, voxels) in enumerate(loader_val):
                real_imgs = Variable(imgs.to(device))
                z = encoder(real_imgs)
                vertices, faces = mesh_generator(z)
                faces = nr.vertices_to_faces(vertices, faces).data
                faces = faces * 1. * (32. - 1) / 32. + 0.5  # normalization
                voxels_predicted = voxelization.voxelize(faces, 32, False)
                voxels_predicted = voxels_predicted.transpose(1, 2).flip([3])
                iou_batch = torch.Tensor.float(voxels * voxels_predicted.cpu()).sum((1, 2, 3)) / \
                            torch.Tensor.float(0 < (voxels + voxels_predicted.cpu())).sum((1, 2, 3))
                iou += iou_batch.sum()
            iou /= dataset_val.num_data[class_id] * 24.
            print('%s/iou_%s: %f' % (dataset_val.set_name, class_id, iou.item()))
            ious['%s/iou_%s' % (dataset_val.set_name, class_id)] = iou.item()
        iou_mean = np.mean([float(v) for v in ious.values()])
        ious['%s/iou' % dataset_val.set_name] = iou_mean
        print('%s/iou: %f' % (dataset_val.set_name, iou_mean))

        mesh_generator.train()
        encoder.train()
        return iou_mean


if opt.mode == 'trainAE':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    f_log = open(os.path.join(opt.ckpt_dir, 'val_log.txt'), 'a+')

    # Initialize encoder and decoder
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
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
    dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train', opt.img_size)
    dataloader = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    dataset_val = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'val', opt.img_size)

    # Optimizers
    optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(dataloader) + opt.batches_done
    batches_done = opt.batches_done
    iou_best = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, viewpoints, viewids, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.to(device))
            viewpoints = Variable(viewpoints.to(device))

            z = encoder(real_imgs)
            # Generate a batch of images
            vertices, faces = mesh_generator(z)
            mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

            gen_imgs1 = mesh_renderer(viewpoints)
            gt_imgs1 = real_imgs[:,3,:,:]
            gt_imgs1 = gt_imgs1.reshape((gt_imgs1.shape[0],1,opt.img_size,opt.img_size))

            smth_loss = L.smoothness_loss(vertices, smoothness_params)
            iou_loss = L.iou_loss(gt_imgs1, gen_imgs1)
            Gprior_loss = torch.sum(z**2)/z.shape[0]

            lambda_Gprior = 1. if opt.AE_Gprior else 0.
            total_loss = iou_loss + opt.lambda_smth * smth_loss + lambda_Gprior * Gprior_loss

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
                ploter.plot('Gprior_loss', 'train', 'Gpior-loss', batches_done, Gprior_loss.item())
                ploter.plot('total_loss', 'train', 'total-loss', batches_done, total_loss.item())

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs1.data[:min(25, gen_imgs1.shape[0])], os.path.join(opt.sample_dir,
                        'random-%05d.png' % batches_done), nrow=5, normalize=True)
                save_image(gen_imgs1.grad[:min(25, gen_imgs1.shape[0])], os.path.join(opt.sample_dir,
                        'random-%05d-grad.png' % batches_done), nrow=5, normalize=True)
                nr.save_obj(os.path.join(opt.sample_dir, 'random-%05d.obj' % batches_done), vertices[0,:,:], faces[0,:,:])
                print('Saved sample image to {}...'.format(opt.sample_dir))

        # validation on val dataset after each epoch
        iou_val = eval_IoU(encoder, mesh_generator, dataset_val)
        f_log.write('batches_done: %d, validation iou: %f\r\n' % (batches_done, iou_val))
        ploter.plot('IoU_loss', 'voxel_IoU_val', 'IoU-loss', batches_done, iou_val)
        torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'last-E.ckpt'))
        torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'last-G.ckpt'))
        print('Saved latest model checkpoints to {}...'.format(opt.ckpt_dir))
        if iou_val > iou_best:
            iou_best = iou_val
            torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'best-E.ckpt'))
            torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'best-G.ckpt'))
            print('Saved best model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_E.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))

    f_log.close()

elif opt.mode == 'trainAE_featGAN':
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    f_log = open(os.path.join(opt.ckpt_dir, 'val_log.txt'), 'a+')

    # Initialize encoder and decoder and discriminator
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    discriminator = M.feat_Discriminator(opt.latent_dim)

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
    dataset_val = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'val', opt.img_size)

    # Optimizers
    optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_epochs * len(batch_sampler) + opt.batches_done
    lambda_KLD = 1e-4
    lambda_D = 1.
    batches_done = opt.batches_done
    iou_best = 0.
    for epoch in range(opt.n_epochs):
        for i, (imgs, viewpoints, viewids_real, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.to(device))
            viewpoints = Variable(viewpoints.to(device))

            # -----------------
            #  Train Generator, Encoder and Discriminator
            # -----------------
            z = encoder(real_imgs)
            # Generate a batch of images
            vertices, faces = mesh_generator(z)
            mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

            gen_imgs = mesh_renderer(viewpoints)
            gt_imgs = real_imgs[:,3,:,:]
            gt_imgs = gt_imgs.reshape((opt.batch_size,1,opt.img_size,opt.img_size))

            smth_loss = L.smoothness_loss(vertices, smoothness_params)
            iou_loss = L.iou_loss(gt_imgs, gen_imgs)

            z_detach = z.detach()
            real_labels = Variable(viewids_real.to(device))
            logdigit = discriminator(z_detach)
            d_loss = F.nll_loss(logdigit, real_labels)
            Gprior_loss = torch.sum(z ** 2) / z.shape[0]
            total_loss = iou_loss + opt.lambda_smth * smth_loss + d_loss + opt.lambda_Gprior*Gprior_loss

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
                z = encoder(real_imgs)
                logdigit = discriminator(z)
                target = torch.ones((z.shape[0], 24), dtype=torch.float)*torch.log(torch.Tensor([1./24.]))
                target = Variable(target.to(device))
                Adv_loss = torch.sum((target - logdigit)**2)/z.shape[0]
                # Adv_loss = -torch.sum(1./24. * logdigit)/z.shape[0]

                # Generate a batch of images
                vertices, faces = mesh_generator(z)
                mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

                gen_imgs = mesh_renderer(viewpoints)
                gt_imgs = real_imgs[:, 3, :, :]
                gt_imgs = gt_imgs.reshape((opt.batch_size, 1, opt.img_size, opt.img_size))

                smth_loss = L.smoothness_loss(vertices, smoothness_params)
                iou_loss = L.iou_loss(gt_imgs, gen_imgs)
                Gprior_loss = torch.sum(z ** 2) / z.shape[0]
                total_loss = iou_loss + opt.lambda_smth * smth_loss + Adv_loss + opt.lambda_Gprior*Gprior_loss

                encoder.zero_grad()
                discriminator.zero_grad()
                mesh_generator.zero_grad()
                mesh_renderer.zero_grad()
                gen_imgs.retain_grad()
                total_loss.backward()
                optimizer_E.step()
                optimizer_G.step()

            batches_done = opt.batches_done + epoch * len(dataloader) + i + 1
            if batches_done == 1:
                save_image(imgs.data[:,0:3,:,:], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=8,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Epoch %d/%d] [Batch %d/%d] [iou loss: %f] [D_loss: %f]"
                      % (t_elapse, epoch, opt.n_epochs, i, len(dataloader), iou_loss.item(), d_loss.item()))

                ploter.plot('IoU_loss', 'train', 'IoU-loss', batches_done, iou_loss.item())
                ploter.plot('smoothness_loss', 'train', 'smoothness-loss', batches_done, smth_loss.item())
                ploter.plot('Gprior_loss', 'train', 'Gpior-loss', batches_done, Gprior_loss.item())
                ploter.plot('D_loss', 'train', 'D-loss', batches_done, d_loss.item())
                if 'Adv_loss' in locals():
                    ploter.plot('Adv_loss', 'train', 'Adv-loss', batches_done, Adv_loss.item())

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs.data[:25], os.path.join(opt.sample_dir, 'random-%05d.png' % batches_done), nrow=5,
                           normalize=True)
                save_image(gen_imgs.grad[:25], os.path.join(opt.sample_dir, 'random-%05d-grad.png' % batches_done),
                           nrow=5, normalize=True)
                # torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'last-E.ckpt'))
                # torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'last-G.ckpt'))
                # torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, 'last-D.ckpt'))
                print('Saved latest model checkpoints to {}...'.format(opt.ckpt_dir))

                nr.save_obj(os.path.join(opt.sample_dir, 'random-%05d.obj' % batches_done), vertices[0,:,:], faces[0,:,:])
                print('Saved sample image to {}...'.format(opt.sample_dir))


        # validation on val dataset after each epoch
        iou_val = eval_IoU(encoder, mesh_generator, dataset_val)
        f_log.write('batches_done: %d, validation iou: %f\r\n' % (batches_done, iou_val))
        ploter.plot('IoU_loss', 'voxel_IoU_val', 'IoU-loss', batches_done, iou_val)
        torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'last-E.ckpt'))
        torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'last-G.ckpt'))
        torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, 'last-D.ckpt'))
        print('Saved latest model checkpoints to {}...'.format(opt.ckpt_dir))
        if iou_val > iou_best:
            iou_best = iou_val
            torch.save(encoder.state_dict(), os.path.join(opt.ckpt_dir, 'best-E.ckpt'))
            torch.save(mesh_generator.state_dict(), os.path.join(opt.ckpt_dir, 'best-G.ckpt'))
            torch.save(discriminator.state_dict(), os.path.join(opt.ckpt_dir, 'best-D.ckpt'))
            print('Saved best model checkpoints to {}...'.format(opt.ckpt_dir))

        if epoch + 1 - opt.decay_epoch >= 0 and (epoch + 1 - opt.decay_epoch) % opt.decay_every == 0:
            opt.lr = opt.lr * opt.decay_order
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_E.param_groups:
                param_group['lr'] = opt.lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = opt.lr
            print('lr decayed to {}'.format(opt.lr))

    f_log.close()

elif opt.mode == 'evaluation':

    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)

    # fn = os.path.join(opt.ckpt_dir, 'val_log.txt')
    # f_log = open(fn, 'r')
    # batches = []
    # ious = []
    # for line in f_log.readlines():
    #     sep = re.split('[ ,\n]', line)  # try out the split regexp
    #     batches.append(sep[1])
    #     ious.append(float(sep[5]))
    # f_log.close()
    # argmax = np.argmax(np.array(ious))
    # ckptG = os.path.join(opt.ckpt_dir, batches[argmax]+'-G.ckpt')
    # ckptE = os.path.join(opt.ckpt_dir, batches[argmax] + '-E.ckpt')
    # print('using %s' % ckptG)

    ckptG = os.path.join(opt.ckpt_dir, opt.eval_flag+'-G.ckpt')
    ckptE = os.path.join(opt.ckpt_dir, opt.eval_flag+'-E.ckpt')
    # Initialize weights
    mesh_generator.load_state_dict(torch.load(ckptG))
    encoder.load_state_dict(torch.load(ckptE))
    # Configure data loader
    dataset_test = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'test')
    eval_IoU(encoder, mesh_generator, dataset_test)

elif opt.mode == 'reconstruct':
    os.makedirs(opt.sample_dir, exist_ok=True)
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)
    # Initialize weights
    mesh_generator.load_state_dict(torch.load(opt.load_G))
    encoder.load_state_dict(torch.load(opt.load_E))
    mesh_generator.eval()
    encoder.eval()

    img0 = plt.imread(opt.load_im)
    img = img0.transpose((2, 0, 1))
    # img = img.astype('float32') / 255.    # !!! plt.imread img automatically in scale of [0,1] ...
    img = img[None,:,:,:]
    img = torch.from_numpy(img)
    real_imgs = Variable(img.to(device))
    z = encoder(real_imgs)
    z = z.repeat((24, 1))
    vertices, faces = mesh_generator(z)
    azimuths = -15. * torch.arange(0, 24)
    azimuths = torch.Tensor.float(azimuths)
    elevations = 30. * torch.ones((24))
    distances = 2.732 * torch.ones((24))
    viewpoints_fixed = nr.get_points_from_angles(distances, elevations, azimuths)
    mesh_renderer = M.Mesh_Renderer(vertices, faces, opt.img_size, 'rgb').cuda(opt.device_id)
    gen_imgs_fixed = mesh_renderer(viewpoints_fixed)

    fn = os.path.splitext(os.path.basename(opt.load_im))[0]
    plt.imsave(os.path.join(opt.sample_dir, fn+'.png'), img0)
    save_image(gen_imgs_fixed.data, os.path.join(opt.sample_dir, fn+'_out.png'), nrow=5,
               normalize=True)
    nr.save_obj(os.path.join(opt.sample_dir, fn+'_out.obj'), vertices[0, :, :], faces[0, :, :])
    print('Saved reconstruction results of %s to %s...' % (fn, opt.sample_dir))

elif opt.mode == 't_SNE':
    # Initialize encoder and decoder
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    if cuda:
        encoder.cuda(opt.device_id)

    # Initialize weights
    encoder.load_state_dict(torch.load(opt.load_E))
    encoder.eval()

    # Configure data loader
    dataset_val = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'val')
    dataloader = data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    # features = np.zeros((len(dataset_val), opt.latent_dim))
    # labels = np.zeros(len(dataset_val))
    n_batches = int(1e6)
    features = np.zeros((min(opt.batch_size*n_batches, len(dataset_val)), opt.latent_dim))
    labels = np.zeros(min(opt.batch_size*n_batches, len(dataset_val)))
    # ----------
    #  forward
    # ----------
    for epoch in range(1):
        for i, (imgs, _, view_ids, _) in enumerate(dataloader):
            if  i == n_batches:
                break
            # Configure input
            real_imgs = Variable(imgs.to(device))
            z = encoder(real_imgs)

            z_cpu = torch.squeeze(z.cpu())
            features[(i * opt.batch_size):min((i + 1) * opt.batch_size, len(dataset_val)),:] = \
                z_cpu.data.numpy()
            labels[(i * opt.batch_size):min((i + 1) * opt.batch_size, len(dataset_val))] = np.squeeze(view_ids)

    print('features shape is: ', features.shape)
    print('labels shape is: ', labels.shape)
    if features.shape[1] > 50:
        print('doing PCA dimension reduction ...')
        features_PCA = PCA(n_components=50).fit_transform(features)
    print('doing t-SNE ...')
    if features.shape[1] > 50:
        features_TSNE = TSNE(n_components=2, init='pca').fit_transform(features_PCA)
    else:
        features_TSNE = TSNE(n_components=2, init='pca').fit_transform(features)

    plt.switch_backend('Agg')
    plt.figure()
    sc = plt.scatter(features_TSNE[:, 0], features_TSNE[:, 1], c=labels, s=2)
    plt.colorbar(sc)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(15.0 / 3, 10.0 / 3)  # dpi = 300, output = 1500*1000 pixels
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05, hspace=0, wspace=0)
    plt.margins(0.01, 0.01)
    sep = re.split('/', opt.load_E)
    fn = sep[-2] + '_' + sep[-1]
    plt.savefig('/hd2/pengbo/mesh_reconstruction/models/img_embedings/'+fn+'.png', dpi=300, pad_inches=0)
    plt.savefig('/hd2/pengbo/mesh_reconstruction/models/img_embedings/'+fn+'.pdf', dpi=300, pad_inches=0)
    # plt.show()

elif opt.mode == 'interpolation':
    os.makedirs(opt.sample_dir, exist_ok=True)
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)
    # Initialize weights
    mesh_generator.load_state_dict(torch.load(opt.load_G))
    encoder.load_state_dict(torch.load(opt.load_E))
    mesh_generator.eval()
    encoder.eval()

    img0 = plt.imread(opt.load_im)
    img = img0.transpose((2, 0, 1))
    # img = img.astype('float32') / 255.    # !!! plt.imread img automatically in scale of [0,1] ...
    img = img[None,:,:,:]
    img = torch.from_numpy(img)
    real_imgs = Variable(img.to(device))
    z0 = encoder(real_imgs)
    z0 = z0.repeat((24, 1))

    img1 = plt.imread(opt.load_im1)
    img = img1.transpose((2, 0, 1))
    # img = img.astype('float32') / 255.    # !!! plt.imread img automatically in scale of [0,1] ...
    img = img[None, :, :, :]
    img = torch.from_numpy(img)
    real_imgs = Variable(img.to(device))
    z1 = encoder(real_imgs)
    z1 = z1.repeat((24, 1))

    N = 5   # number of interpolation points
    azimuths = -15. * torch.arange(0, 24)
    azimuths = torch.Tensor.float(azimuths)
    elevations = 30. * torch.ones((24))
    distances = 2.732 * torch.ones((24))
    viewpoints_fixed = nr.get_points_from_angles(distances, elevations, azimuths)
    fn0 = os.path.splitext(os.path.basename(opt.load_im))[0]
    (fn, id0) = re.split('_', fn0)[0:2]
    fn1 = os.path.splitext(os.path.basename(opt.load_im1))[0]
    (_, id1) = re.split('_', fn1)[0:2]
    plt.imsave(os.path.join(opt.sample_dir, fn0 + '.png'), img0)
    plt.imsave(os.path.join(opt.sample_dir, fn1 + '.png'), img1)
    for i in range(N+1):
        z = z0 + (z1 - z0)*i/N
        vertices, faces = mesh_generator(z)

        mesh_renderer = M.Mesh_Renderer(vertices, faces, opt.img_size, 'rgb').cuda(opt.device_id)
        gen_imgs_fixed = mesh_renderer(viewpoints_fixed)

        save_image(gen_imgs_fixed.data, os.path.join(opt.sample_dir, fn+('_%s_%s_%d.png' % (id0, id1, i))), nrow=5,
                   normalize=True)
        nr.save_obj(os.path.join(opt.sample_dir, fn+('_%s_%s_%d.obj' % (id0, id1, i))), vertices[0, :, :], faces[0, :, :])
    print('Saved reconstruction results of %s to %s...' % (fn, opt.sample_dir))

elif opt.mode == 'MMD':
    # Initialize encoder and decoder
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    if cuda:
        encoder.cuda(opt.device_id)

    # Initialize weights
    encoder.load_state_dict(torch.load(opt.load_E))
    encoder.eval()

    # Configure data loader
    dataset_test = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'test')
    dataloader = data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    # features = np.zeros((len(dataset_val), opt.latent_dim))
    # labels = np.zeros(len(dataset_val))
    n_batches = 24*4 #int(1e6)
    features = torch.zeros((min(opt.batch_size*n_batches, len(dataset_test)), opt.latent_dim)).to(device)
    # ----------
    #  forward
    # ----------
    for epoch in range(1):
        for i, (imgs, _, view_ids, _) in enumerate(dataloader):
            if  i == n_batches:
                break
            # Configure input
            real_imgs = Variable(imgs.to(device))
            z = encoder(real_imgs)

            features[(i * opt.batch_size):min((i + 1) * opt.batch_size, len(dataset_test)),:] = torch.squeeze(z.data)

    features = features.reshape((int(features.shape[0]/24), 24, features.shape[1]))
    features = features.transpose(1,0)
    print('features shape is: ', features.shape)

    MMDs = torch.zeros(int(24*23/2)).to(device)
    k = 0
    for i in range(24):
        for j in range(i+1, 24):
            sliceA = features[i,:,:]
            sliceB = features[j,:,:]
            MMD = L.mmd_rbf(sliceA, sliceB)
            # print('processed %d/%d, MMD is %f' % (k+1, int(24*23/2), MMD))
            MMDs[k] = MMD
            k += 1
    meanMMD = torch.mean(MMDs)
    print('%s meanMMD is: %f' % (opt.class_ids, meanMMD.item()))
