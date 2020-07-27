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
import math
import torchvision.transforms as transforms

def str2bool(v):
    return v.lower() in ('true')


CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
DATASET_DIRECTORY = '/hd2/pengbo/mesh_reconstruction/dataset/'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='mode can be one of [xxx]')
parser.add_argument('--dataset', type=str, help='which dataset to use-[CVPR18, NIPS17]', default='CVPR18')
parser.add_argument('--data_dir', type=str, help='dir of dataset')
parser.add_argument('--trainViews', type=int, help='number of views used in training', default=1)
parser.add_argument('--split_file', type=str, help='dir of dataset split file [for NIPS17 dataset]')
parser.add_argument('--wAzimuth', type=float, help='width of azimuth for view Bing of NIPS17 data', default=15)
parser.add_argument('--wElevation', type=float, help='width of elevation for view Bing of NIPS17 data', default=10)
parser.add_argument('--n_iters', type=int, default=3000, help='number of iterations for training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--decay_batch', type=int, default=5000, help='number of batches/iters before lr decay')
parser.add_argument('--decay_order', type=float, default=0.1, help='order of lr decay')
parser.add_argument('--decay_every', type=int, default=2000, help='lr decay every n batches/iters')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_step', type=int, default=500, help='number of iters between image sampling')
parser.add_argument('--sample_dir', type=str, help='dir of saved sample images')
parser.add_argument('--log_step', type=int, default=10, help='number of iters to print and log')
parser.add_argument('--ckpt_step', type=int, default=500, help='number of iters for model saving')
parser.add_argument('--ckpt_dir', type=str, help='dir of saved model checkpoints')
parser.add_argument('--visdom_env', type=str, default=None, help='Visdom environment name')

parser.add_argument('--batches_done', type=int, default=0, help='previous batches_done when '
                                                                'loading ckpt and continue traning')
parser.add_argument('--load_G', type=str, default=None, help='path of to the loaded Generator weights')
parser.add_argument('--load_D', type=str, default=None, help='path of to the loaded Discriminator weights')
parser.add_argument('--load_E', type=str, default=None, help='path of to the loaded Encoder weights')
parser.add_argument('--load_im', type=str, default=None, help='image to load to reconstruct')
parser.add_argument('--load_im1', type=str, default=None, help='another image to load to interplolate')
parser.add_argument('--device_id', type=int, default=0, help='choose device ID')
parser.add_argument('--G_every', type=int, default=1, help='G:D training schedule is 1:G_every')

parser.add_argument('--class_ids', type=str, default=CLASS_IDS_ALL, help='use which object class images')
parser.add_argument('--obj_dir', type=str, help='base sphere obj file path')
parser.add_argument('--lambda_smth', type=float, default=0.1, help='weight for mesh smoothness loss')
parser.add_argument('--lambda_Lap', type=float, default=0.1, help='weight for mesh Laplacian loss')
parser.add_argument('--lambda_edge', type=float, default=0.1, help='weight of edge length loss')
parser.add_argument('--lambda_Gprior', type=float, default=0., help='weight of Gaussian prior term')
parser.add_argument('--lambda_adv', type=float, default=0., help='weight of adversary loss')

parser.add_argument('--eval_flag', type=str, default='last', help='which ckpt to evaluate')
parser.add_argument('--prefix', type=str, help='prefix id of the experimental run')

t_start = time.time()
opt = parser.parse_args()
print(opt)
if opt.dataset == 'CVPR18':
    viewBins = 24
    nViews = 24
elif opt.dataset == 'NIPS17':
    nAzimuth = math.ceil(360 / opt.wAzimuth)
    nElevation = math.ceil(60 / opt.wElevation)
    viewBins = nAzimuth*nElevation
    nViews = 20
elif opt.dataset == 'Pascal3D':
    viewBins = 0  # to be assigned later according to specific dataset
    nViews = 1

cuda = True if torch.cuda.is_available() else False
device = 'cuda:%d' % opt.device_id if opt.device_id >= 0 else 'cpu'
print(device)
if opt.visdom_env is not None:
    ploter = logger.VisdomLinePlotter(env_name=opt.visdom_env)


def eval_IoU(encoder, mesh_generator, dataset_val, class_ids = opt.class_ids.split(',')):
    mesh_generator.eval()
    encoder.eval()
    if opt.dataset == 'Pascal3D':
        class_id_dic = {'02691156': 'aeroplane', '02958343': 'car', '03001627': 'chair'}
        class_ids = [class_id_dic[class_id] for class_id in class_ids]
    with torch.no_grad():
        for class_id in class_ids:      # considering multiple classes case.
            loader_val = data.DataLoader(dataset_val, batch_sampler=
            data_loader.ShapeNet_sampler_all(dataset_val, opt.batch_size, class_id, nViews), num_workers=4)
            iou = 0
            ious = {}
            print('%s_%s has %d images, %d batches...' % (dataset_val.set_name, class_id,
                                                          dataset_val.num_data[class_id] * nViews, len(loader_val)))
            for i, (imgs, _, _, voxels) in enumerate(loader_val):
                real_imgs = Variable(imgs.to(device))
                if opt.dataset == 'Pascal3D':
                    input_imgs = real_imgs[:, 0:3, :, :]
                    if opt.img_size == 224:  # use resnet-18 as encoder and normalize input accordingly
                        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
                        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
                        mean = mean.repeat(input_imgs.shape[0], 224, 224, 1).transpose(1, 3)
                        std = std.repeat(input_imgs.shape[0], 224, 224, 1).transpose(1, 3)
                        # print(input_imgs.shape)
                        # print(mean.shape)
                        input_imgs = (input_imgs - mean) / std
                else:
                    input_imgs = real_imgs
                z = encoder(input_imgs)
                vertices, faces = mesh_generator(z)
                faces = nr.vertices_to_faces(vertices, faces).data

                if opt.dataset == 'CVPR18':
                    faces = faces * 1. * (32. - 1) / 32. + 0.5  # normalization
                    voxels_predicted = voxelization.voxelize(faces, 32, False)
                    voxels_predicted = voxels_predicted.transpose(1, 2).flip([3])
                elif opt.dataset == 'NIPS17':
                    faces = faces * 1. * (32. - 1) / 32. + 0.5  # normalization
                    voxels_predicted = voxelization.voxelize(faces, 32, False)
                    voxels_predicted = voxels_predicted.transpose(1, 3).flip([1])
                elif opt.dataset == 'Pascal3D':
                    faces = (faces + 1.0) * 0.5  # normalization
                    voxels_predicted = voxelization.voxelize(faces, 32, False)
                    voxels_predicted = voxels_predicted.transpose(2, 3).flip([1,2,3])

                iou_batch = torch.Tensor.float(voxels * voxels_predicted.cpu()).sum((1, 2, 3)) / \
                            torch.Tensor.float(0 < (voxels + voxels_predicted.cpu())).sum((1, 2, 3))
                iou += iou_batch.sum()
            iou /= dataset_val.num_data[class_id] * nViews
            print('%s/iou_%s: %f' % (dataset_val.set_name, class_id, iou.item()))
            ious['%s/iou_%s' % (dataset_val.set_name, class_id)] = iou.item()
        iou_mean = np.mean([float(v) for v in ious.values()])
        ious['%s/iou' % dataset_val.set_name] = iou_mean
        print('%s/iou: %f' % (dataset_val.set_name, iou_mean))

        mesh_generator.train()
        encoder.train()
        return iou_mean


def eval_MMD(encoder, dataset_test):   # currently only applicable to cvpr18 dataset, as NIPS17 dataset has random views
    encoder.eval()

    # Configure data loader
    dataloader = data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    n_batches = nViews * 4  # int(1e6)  # choose small n_batches for efficiency
    features = torch.zeros((min(opt.batch_size * n_batches, len(dataset_test)), opt.latent_dim)).to(device)
    # ----------
    #  forward
    # ----------
    for epoch in range(1):
        for i, (imgs, _, view_ids, _) in enumerate(dataloader):
            if i == n_batches:
                break
            # Configure input
            real_imgs = Variable(imgs.to(device))
            z = encoder(real_imgs)

            features[(i * opt.batch_size):min((i + 1) * opt.batch_size, len(dataset_test)), :] = torch.squeeze(z.data)

    features = features.reshape((int(features.shape[0] / nViews), nViews, features.shape[1]))
    features = features.transpose(1, 0)
    print('features shape is: ', features.shape)

    MMDs = torch.zeros(int(nViews * (nViews-1) / 2)).to(device)
    k = 0
    for i in range(nViews):
        for j in range(i + 1, nViews):
            sliceA = features[i, :, :]
            sliceB = features[j, :, :]
            MMD = L.mmd_rbf(sliceA, sliceB)
            # print('processed %d/%d, MMD is %f' % (k+1, int(viewBins*23/2), MMD))
            MMDs[k] = MMD
            k += 1
    meanMMD = torch.mean(MMDs)
    print('%s meanMMD is: %f' % (opt.class_ids, meanMMD.item()))
    encoder.train()
    return meanMMD


if opt.mode == 'train':
    # training the proposed model "3D-AE-featGAN"
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    f_log = open(os.path.join(opt.ckpt_dir, 'val_log.txt'), 'a+')
    batches_done = opt.batches_done
    if batches_done != 0:       # continued training
        ious = []
        f_log.seek(0, os.SEEK_SET)
        for line in f_log.readlines():
            sep = re.split('[ ,\n]', line)  # try out the split regexp
            ious.append(float(sep[5]))
        iou_best = max(ious)        # previous best evaluation result
    else:
        iou_best = 0.

    if opt.dataset == 'CVPR18':
        # Configure data loader for shapeNet dataset in Kato's CVPR18
        dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train', opt.img_size)
        dataloader = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=False)
        dataset_val = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'val', opt.img_size)
    elif opt.dataset == 'NIPS17':
        # Configure data loader for shapeNet dataset in Kar's NIPS17
        dataset_train = data_loader.ShapeNet_LSM(opt.data_dir, opt.split_file,
                                                 opt.class_ids.split(','), 'train', opt.img_size, opt.trainViews,
                                                 opt.wAzimuth, opt.wElevation)  # following Kato's CVPR19 paper, use only one view for training, all views for testing.
        dataloader = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=False, num_workers=4)
        dataset_val = data_loader.ShapeNet_LSM(opt.data_dir, opt.split_file,
                                               opt.class_ids.split(','), 'val', opt.img_size, nViews,
                                               opt.wAzimuth, opt.wElevation) # use all views for validation
    elif opt.dataset == 'Pascal3D':
        # Configure data loader for Pascal 3D+ dataset from Kato's CVPR19
        dataset_train = data_loader.Pascal(opt.data_dir, opt.class_ids.split(','), 'train', opt.img_size)
        dataloader = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=False, num_workers=4)
        dataset_val = data_loader.Pascal(opt.data_dir, opt.class_ids.split(','), 'val', opt.img_size)
        viewBins = dataset_train.viewBins  # assign nViews according to pose distribution in training dataset

    print('# training images: %d, # val images: %d, # viewBins: %d' % (len(dataset_train), len(dataset_val), viewBins))

    # Initialize encoder and decoder and discriminator
    if opt.dataset == 'Pascal3D':
        if opt.img_size == 64:
            encoder = M.Encoder(3, dim_out=opt.latent_dim)
        elif opt.img_size == 128:
            encoder = M.Encoder(3, dim_out=opt.latent_dim, nConvs=4)
        elif opt.img_size == 224:  # use resnet-18 as encoder
            encoder = M.ResNet_Encoder()
    else:
        encoder = M.Encoder(4, dim_out=opt.latent_dim)

    if opt.dataset == 'Pascal3D' and opt.img_size == 224:
        mesh_generator = M.Mesh_Generator_symmetry(opt.latent_dim, opt.obj_dir)
    else:
        mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    discriminator = M.feat_Discriminator(opt.latent_dim, viewBins)

    if os.path.isfile('smoothness_params_642.npy'):
        smoothness_params = np.load('smoothness_params_642.npy')
    else:
        smoothness_params = L.smoothness_loss_parameters(mesh_generator.faces)

    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)
        discriminator.cuda(opt.device_id)

    # print('check is model on cuda')
    # print(next(encoder.parameters()).is_cuda)
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

    # Optimizers
    optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_iters + batches_done
    lambda_D = 1.

    while True:
        for _, (imgs, viewpoints, viewids_real, _) in enumerate(dataloader):
            batches_done = batches_done + 1
            # Configure input
            real_imgs = Variable(imgs.to(device))
            viewpoints = Variable(viewpoints.to(device))

            # -----------------
            #  Train Generator, Encoder and Discriminator
            # -----------------
            if opt.dataset == 'Pascal3D':
                input_imgs = real_imgs[:, 0:3, :, :]
                if opt.img_size == 224:  # use resnet-18 as encoder and normalize input accordingly
                    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
                    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
                    mean = mean.repeat(input_imgs.shape[0], 224, 224, 1).transpose(1,3)
                    std = std.repeat(input_imgs.shape[0], 224, 224, 1).transpose(1,3)
                    # print(input_imgs.shape)
                    # print(mean.shape)
                    input_imgs = (input_imgs - mean)/std
            else:
                input_imgs = real_imgs
            z = encoder(input_imgs)
            # Generate a batch of images
            vertices, faces = mesh_generator(z)
            if opt.img_size == 224:   # try using smaller size masks as guidence
                mesh_renderer = M.Mesh_Renderer(vertices, faces, dataset=opt.dataset, img_size=64).cuda(opt.device_id)
            else:
                mesh_renderer = M.Mesh_Renderer(vertices, faces, dataset=opt.dataset, img_size=opt.img_size).cuda(opt.device_id)

            gen_imgs = mesh_renderer(viewpoints)
            gt_imgs = real_imgs[:,3,:,:]
            gt_imgs = gt_imgs.reshape((gt_imgs.shape[0],1,opt.img_size,opt.img_size))
            if opt.img_size == 224:  # try using smaller size masks as guidence
                gt_imgs = torch.nn.functional.interpolate(gt_imgs, (64, 64))

            Gprior_loss = torch.sum(z ** 2) / z.shape[0]
            smth_loss = L.smoothness_loss(vertices, smoothness_params)
            iou_loss = L.iou_loss(gt_imgs, gen_imgs)
            # smth_loss = L.inflation_loss(vertices, faces)  # Kato's inflation loss not working well in my implementation.

            if batches_done % opt.G_every != 0:        # Train Discriminator
                z_detach = z.detach()
                real_labels = Variable(viewids_real.to(device))
                logdigit = discriminator(z_detach)
                d_loss = F.nll_loss(logdigit, real_labels)
                total_loss = iou_loss + opt.lambda_smth * smth_loss + opt.lambda_Gprior * Gprior_loss + d_loss
                discriminator.zero_grad()
                mesh_renderer.zero_grad()
                mesh_generator.zero_grad()
                encoder.zero_grad()
                total_loss.backward()
                optimizer_D.step()
                optimizer_G.step()
                optimizer_E.step()
            else:           # Train Encoder with adversary loss
                logdigit = discriminator(z)
                target = torch.ones((z.shape[0], viewBins), dtype=torch.float)*torch.log(torch.Tensor([1./viewBins]))
                target = Variable(target.to(device))
                Adv_loss = torch.sum((target - logdigit)**2)/z.shape[0]
                total_loss = iou_loss + opt.lambda_smth * smth_loss + opt.lambda_Gprior*Gprior_loss \
                             + opt.lambda_adv*Adv_loss

                encoder.zero_grad()
                discriminator.zero_grad()
                mesh_generator.zero_grad()
                mesh_renderer.zero_grad()
                total_loss.backward()
                optimizer_E.step()
                optimizer_G.step()

            if batches_done == 1:
                save_image(imgs.data[:,0:3,:,:], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=8,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Batch %d/%d] [iou loss: %f]"
                      % (t_elapse, batches_done, last_iter, iou_loss.item()))

                ploter.plot('IoU_loss', 'train', 'IoU-loss', batches_done, iou_loss.item())
                ploter.plot('smoothness_loss', 'train', 'smoothness-loss', batches_done, smth_loss.item())
                ploter.plot('Gprior_loss', 'train', 'Gpior-loss', batches_done, Gprior_loss.item())
                ploter.plot('D_loss', 'train', 'D-loss', batches_done, d_loss.item())
                if 'Adv_loss' in locals():
                    ploter.plot('Adv_loss', 'train', 'Adv-loss', batches_done, Adv_loss.item())

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs.data[:min(25, gen_imgs.shape[0])], os.path.join(opt.sample_dir,
                            '%05d-gen.png' % batches_done), nrow=5, normalize=True)
                save_image(gt_imgs.data[:min(25, gt_imgs.shape[0])], os.path.join(opt.sample_dir,
                            '%05d-gt.png' % batches_done), nrow=5, normalize=True)
                nr.save_obj(os.path.join(opt.sample_dir, '%05d.obj' % batches_done), vertices[0,:,:], faces[0,:,:])
                print('Saved sample image to {}...'.format(opt.sample_dir))

            # validation on val dataset
            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                iou_val = eval_IoU(encoder, mesh_generator, dataset_val)
                if opt.dataset == 'CVPR18':
                    MMD_val = eval_MMD(encoder, dataset_val)
                f_log.write('batches_done: %d, validation iou: %f\r\n' % (batches_done, iou_val))
                ploter.plot('IoU_loss', 'voxel_IoU_val', 'IoU-loss', batches_done, iou_val)
                if opt.dataset == 'CVPR18':
                    ploter.plot('IoU_loss', 'MMD_val', 'IoU-loss', batches_done, MMD_val)
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

            if batches_done >= opt.decay_batch and (batches_done - opt.decay_batch) % opt.decay_every == 0:
                opt.lr = opt.lr * opt.decay_order
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] = opt.lr
                for param_group in optimizer_E.param_groups:
                    param_group['lr'] = opt.lr
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = opt.lr
                print('lr decayed to {}'.format(opt.lr))

            if batches_done == last_iter:   # reached maximum iteration and break out for loop.
                break
        if batches_done == last_iter:  # reached maximum iteration and break out while loop.
            break

    f_log.close()
    ploter.save()

if opt.mode == 'trainCVPR19':
    # training compared CVPR19 model: VPL (see paper for details)
    os.makedirs(opt.sample_dir, exist_ok=True)
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    f_log = open(os.path.join(opt.ckpt_dir, 'val_log.txt'), 'a+')
    batches_done = opt.batches_done
    if batches_done != 0:  # continued training
        ious = []
        f_log.seek(0, os.SEEK_SET)
        for line in f_log.readlines():
            sep = re.split('[ ,\n]', line)  # try out the split regexp
            ious.append(float(sep[5]))
        iou_best = max(ious)  # previous best evaluation result
    else:
        iou_best = 0.

    # Initialize encoder and decoder and discriminator
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    discriminator = M.DCGAN_Discriminator(opt.img_size, opt.channels)

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
    dataset_train = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'train', opt.img_size)
    dataloader = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    dataset_val = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'val', opt.img_size)

    # Optimizers
    optimizer_G = torch.optim.Adam(mesh_generator.parameters(), lr=opt.lr)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------
    last_iter = opt.n_iters + batches_done

    while True:
        for _, (imgs, viewpoints, viewids_real, _) in enumerate(dataloader):
            batches_done = batches_done + 1
            # Configure input
            real_imgs = Variable(imgs.to(device))
            viewpoints = Variable(viewpoints.to(device))

            # -----------------
            #  Train Generator, Encoder with Discriminator backprop
            # -----------------
            z = encoder(real_imgs)
            # Generate a batch of images
            vertices, faces = mesh_generator(z)
            mesh_renderer = M.Mesh_Renderer(vertices, faces).cuda(opt.device_id)

            gen_imgs = mesh_renderer(viewpoints)
            gt_imgs = real_imgs[:, 3, :, :]
            gt_imgs = gt_imgs.reshape((gt_imgs.shape[0], 1, opt.img_size, opt.img_size))

            smth_loss = L.smoothness_loss(vertices, smoothness_params)
            iou_loss = L.iou_loss(gt_imgs, gen_imgs)

            imgs_newView, viewids_fake = mesh_renderer(viewidN = torch.Tensor.float(viewids_real).to(device))  # random new views
            labels_fake = torch.zeros((imgs_newView.shape[0], viewBins, opt.img_size, opt.img_size),
                                      dtype=torch.float)
            labels_fake = Variable(labels_fake.to(device))
            labels_real = torch.zeros((gen_imgs.shape[0], viewBins, opt.img_size, opt.img_size), dtype=torch.float)
            labels_real = Variable(labels_real.to(device))

            for ii in range(imgs_newView.shape[0]):
                labels_fake[ii, int(viewids_fake[ii]), :, :] = 1.
                labels_real[ii, int(viewids_real[ii]), :, :] = 1.
            imgs_newView_label = torch.cat((imgs_newView, labels_fake), 1)  # images conditioned on viewpoints.
            gen_imgs_label = torch.cat((gen_imgs, labels_real), 1)

            fake = Variable(torch.Tensor(imgs.shape[0], 1).fill_(0.0).to(device), requires_grad=False)
            real = Variable(torch.Tensor(imgs.shape[0], 1).fill_(1.0).to(device), requires_grad=False)

            if batches_done % opt.G_every != 0:  # Train Discriminator
                p_fake, _ = discriminator(imgs_newView_label.detach())
                p_real, _ = discriminator(gen_imgs_label.detach())
                d_loss = (F.binary_cross_entropy(p_fake, fake) + F.binary_cross_entropy(p_real, real))/2.
                total_loss = iou_loss + opt.lambda_smth*smth_loss + d_loss

                discriminator.zero_grad()
                mesh_renderer.zero_grad()
                mesh_generator.zero_grad()
                encoder.zero_grad()
                total_loss.backward()
                optimizer_D.step()
                optimizer_G.step()
                optimizer_E.step()
            else:       # Train Generator
                p_fake, _ = discriminator(imgs_newView_label)
                adv_loss = F.binary_cross_entropy(p_fake, real)
                total_loss = iou_loss + opt.lambda_smth*smth_loss + opt.lambda_adv*adv_loss

                discriminator.zero_grad()
                mesh_renderer.zero_grad()
                mesh_generator.zero_grad()
                encoder.zero_grad()
                total_loss.backward()
                optimizer_G.step()
                optimizer_E.step()

            if batches_done == 1:
                save_image(imgs.data[:, 0:3, :, :], os.path.join(opt.sample_dir, 'real_samples.png'), nrow=8,
                           normalize=True)
                print('Saved real sample image to {}...'.format(opt.sample_dir))

            if batches_done % opt.log_step == 0 or batches_done == last_iter:
                t_now = time.time()
                t_elapse = t_now - t_start
                t_elapse = str(datetime.timedelta(seconds=t_elapse))[:-7]
                print("[Time %s] [Batch %d/%d] [iou loss: %f]"
                      % (t_elapse, batches_done, last_iter, iou_loss.item()))

                ploter.plot('IoU_loss', 'train', 'IoU-loss', batches_done, iou_loss.item())
                ploter.plot('smoothness_loss', 'train', 'smoothness-loss', batches_done, smth_loss.item())
                if 'd_loss' in locals():
                    ploter.plot('D_loss', 'train', 'D-loss', batches_done, d_loss.item())
                if 'adv_loss' in locals():
                    ploter.plot('Adv_loss', 'train', 'Adv-loss', batches_done, adv_loss.item())

            if batches_done % opt.sample_step == 0 or batches_done == last_iter:
                save_image(gen_imgs.data[:min(25, gen_imgs.shape[0])], os.path.join(opt.sample_dir,
                                                                                    '%05d-gen.png' % batches_done),
                           nrow=5, normalize=True)
                save_image(gt_imgs.data[:min(25, gt_imgs.shape[0])], os.path.join(opt.sample_dir,
                                                                                  '%05d-gt.png' % batches_done), nrow=5,
                           normalize=True)
                nr.save_obj(os.path.join(opt.sample_dir, '%05d.obj' % batches_done), vertices[0, :, :], faces[0, :, :])
                print('Saved sample image to {}...'.format(opt.sample_dir))

            # validation on val dataset
            if batches_done % opt.ckpt_step == 0 or batches_done == last_iter:
                iou_val = eval_IoU(encoder, mesh_generator, dataset_val)
                MMD_val = eval_MMD(encoder, dataset_val)
                f_log.write('batches_done: %d, validation iou: %f\r\n' % (batches_done, iou_val))
                ploter.plot('IoU_loss', 'voxel_IoU_val', 'IoU-loss', batches_done, iou_val)
                ploter.plot('IoU_loss', 'MMD_val', 'IoU-loss', batches_done, MMD_val)
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

            if batches_done >= opt.decay_batch and (batches_done - opt.decay_batch) % opt.decay_every == 0:
                opt.lr = opt.lr * opt.decay_order
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] = opt.lr
                for param_group in optimizer_E.param_groups:
                    param_group['lr'] = opt.lr
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = opt.lr
                print('lr decayed to {}'.format(opt.lr))

            if batches_done == last_iter:  # reached maximum iteration and break out for loop.
                break
        if batches_done == last_iter:  # reached maximum iteration and break out while loop.
            break

    f_log.close()
    ploter.save()

elif opt.mode == 'evaluation':
    # evaluate reconstruction accuracy using voxel IoU
    f_log = open(os.path.join(opt.ckpt_dir, 'test_log.txt'), 'a+')
    f_log.write(str(datetime.datetime.now())+'\n')
    print(opt.class_ids + ', mean')
    f_log.write(opt.class_ids + ', mean'+'\n')
    if opt.dataset == 'Pascal3D':
        encoder = M.Encoder(3, dim_out=opt.latent_dim)
    else:
        encoder = M.Encoder(4, dim_out=opt.latent_dim)
    mesh_generator = M.Mesh_Generator(opt.latent_dim, opt.obj_dir)
    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)
    ious = []
    for class_id in opt.class_ids.split(','):
        # subdir = 'ckpt3D_' + class_id + '_' + opt.prefix
        subdir = 'ckpt3D_' + class_id
        ckptG = os.path.join(opt.ckpt_dir, subdir, opt.eval_flag+'-G.ckpt')
        ckptE = os.path.join(opt.ckpt_dir, subdir, opt.eval_flag+'-E.ckpt')
        # Initialize weights
        mesh_generator.load_state_dict(torch.load(ckptG))
        encoder.load_state_dict(torch.load(ckptE))
        # Configure data loader
        if opt.dataset == 'CVPR18':
            dataset_test = data_loader.ShapeNet(opt.data_dir, [class_id], 'test')
        elif opt.dataset == 'NIPS17':
            dataset_test = data_loader.ShapeNet_LSM(opt.data_dir, opt.split_file,
                                               [class_id], 'test', opt.img_size, nViews)
        elif opt.dataset == 'Pascal3D':
            dataset_test = data_loader.Pascal(opt.data_dir,
                                               [class_id], 'val', opt.img_size)

        ious.append(eval_IoU(encoder, mesh_generator, dataset_test, [class_id]))
    ious.append(np.mean(ious))
    print(str(ious))
    f_log.write(str(ious)+'\n')
    f_log.close()

elif opt.mode == 'reconstruct':
    # reconstruct sample images and save models for visulization
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
    z = z.repeat((viewBins, 1))
    vertices, faces = mesh_generator(z)
    azimuths = -15. * torch.arange(0, viewBins)
    azimuths = torch.Tensor.float(azimuths)
    elevations = 30. * torch.ones((viewBins))
    distances = 2.732 * torch.ones((viewBins))
    viewpoints_fixed = nr.get_points_from_angles(distances, elevations, azimuths)
    mesh_renderer = M.Mesh_Renderer(vertices, faces, opt.img_size, 'rgb').cuda(opt.device_id)
    gen_imgs_fixed = mesh_renderer(viewpoints_fixed)

    fn = os.path.splitext(os.path.basename(opt.load_im))[0]
    plt.imsave(os.path.join(opt.sample_dir, fn+'.png'), img0)
    save_image(gen_imgs_fixed.data, os.path.join(opt.sample_dir, fn+'_out.png'), nrow=5,
               normalize=True)
    nr.save_obj(os.path.join(opt.sample_dir, fn+'_out.obj'), vertices[0, :, :], faces[0, :, :])
    print('Saved reconstruction results of %s to %s...' % (fn, opt.sample_dir))

elif opt.mode == 'reconstruct_Pascal':
    # reconstruct sample images in Pascal val dataset and save models for visulization
    os.makedirs(opt.sample_dir, exist_ok=True)
    encoder = M.ResNet_Encoder()
    mesh_generator = M.Mesh_Generator_symmetry(opt.latent_dim, opt.obj_dir)

    if cuda:
        mesh_generator.cuda(opt.device_id)
        encoder.cuda(opt.device_id)
    # Initialize weights
    mesh_generator.load_state_dict(torch.load(opt.load_G))
    encoder.load_state_dict(torch.load(opt.load_E))
    mesh_generator.eval()
    encoder.eval()

    dataset_val = data_loader.Pascal(opt.data_dir, opt.class_ids.split(','), 'val', opt.img_size)
    dataloader = data.DataLoader(dataset_val, batch_size=8, shuffle=False, drop_last=False, num_workers=1)
    sampleBatch = 0
    for iBatch, (imgs, viewpoints, viewids_real, _) in enumerate(dataloader):
        if iBatch == sampleBatch:
            break
    # Configure input
    real_imgs = Variable(imgs.to(device))
    viewpoints = Variable(viewpoints.to(device))
    input_imgs = real_imgs[:, 0:3, :, :]
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.repeat(input_imgs.shape[0], 224, 224, 1).transpose(1, 3)
    std = std.repeat(input_imgs.shape[0], 224, 224, 1).transpose(1, 3)
    # print(input_imgs.shape)
    # print(mean.shape)
    input_imgs = (input_imgs - mean) / std

    z = encoder(input_imgs)
    # Generate a batch of images
    vertices, faces = mesh_generator(z)

    objID = opt.class_ids.split(',')[0]
    save_image(imgs.data[:, 0:3, :, :], os.path.join(opt.sample_dir, 'real_samples_%s.png' % objID), nrow=8,
               normalize=True)
    for i in range(imgs.shape[0]):
        nr.save_obj(os.path.join(opt.sample_dir, 'out_%s_baseline_%d.obj' % (objID, i)), vertices[i, :, :], faces[i, :, :])

elif opt.mode == 't_SNE':
    # t_SNE visualization of learned shape embedings
    # Initialize encoder and decoder
    if opt.dataset == 'Pascal3D':
        if opt.img_size == 64:
            encoder = M.Encoder(3, dim_out=opt.latent_dim)
        elif opt.img_size == 128:
            encoder = M.Encoder(3, dim_out=opt.latent_dim, nConvs=4)
        elif opt.img_size == 224:  # use resnet-18 as encoder
            encoder = M.ResNet_Encoder()
    else:
        encoder = M.Encoder(4, dim_out=opt.latent_dim)
    if cuda:
        encoder.cuda(opt.device_id)

    # Initialize weights
    encoder.load_state_dict(torch.load(opt.load_E))
    encoder.eval()

    # Configure data loader
    if opt.dataset == 'CVPR18':
        dataset_val = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'val')
    elif opt.dataset == 'NIPS17':
        dataset_val = data_loader.ShapeNet_LSM(opt.data_dir, opt.split_file,
                                               opt.class_ids.split(','), 'test', opt.img_size, nViews)
    elif opt.dataset == 'Pascal3D':
        dataset_val = data_loader.Pascal(opt.data_dir,
                                         opt.class_ids.split(','), 'train', opt.img_size)
    dataloader = data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last=False)

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
            if opt.dataset == 'Pascal3D':
                input_imgs = real_imgs[:, 0:3, :, :]
            else:
                input_imgs = real_imgs
            z = encoder(input_imgs)

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
    # shape space interpolation visualization
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
    z0 = z0.repeat((viewBins, 1))

    img1 = plt.imread(opt.load_im1)
    img = img1.transpose((2, 0, 1))
    # img = img.astype('float32') / 255.    # !!! plt.imread img automatically in scale of [0,1] ...
    img = img[None, :, :, :]
    img = torch.from_numpy(img)
    real_imgs = Variable(img.to(device))
    z1 = encoder(real_imgs)
    z1 = z1.repeat((viewBins, 1))

    N = 5   # number of interpolation points
    azimuths = -15. * torch.arange(0, viewBins)
    azimuths = torch.Tensor.float(azimuths)
    elevations = 30. * torch.ones((viewBins))
    distances = 2.732 * torch.ones((viewBins))
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
    # caculate MMD distance of shape embeddings in different domains/poses
    # Initialize encoder and decoder
    encoder = M.Encoder(4, dim_out=opt.latent_dim)
    if cuda:
        encoder.cuda(opt.device_id)

    # Initialize weights
    encoder.load_state_dict(torch.load(opt.load_E))

    # Configure data loader
    dataset_test = data_loader.ShapeNet(opt.data_dir, opt.class_ids.split(','), 'test')
    eval_MMD(encoder, dataset_test)
