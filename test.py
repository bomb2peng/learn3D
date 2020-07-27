# test the cuda module "voxelization"
import torch
import neural_renderer as nr
import voxelization
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#
# npz_dir = "/hd2/pengbo/mesh_reconstruction/dataset/02691156_test_voxels.npz"
# voxels_all = np.load(npz_dir)['arr_0']
# voxel = voxels_all[1,:,:,:]
# print(voxel.shape)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(voxel, facecolors='green', edgecolor='k')

# voxel_dir = "/hd3/pengbo/shapenet_LSM/lsm/data/shapenet_release/voxels/modelVoxels32/02691156/10155655850468db78d106ce0a280f87.mat"
# voxel = sio.loadmat(voxel_dir)['Volume']
# voxel = voxel.transpose((2,1,0))
# voxel = np.flip(voxel, 2)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(voxel, facecolors='green', edgecolor='k')
#
#
# # # vertices0, faces0 = nr.load_obj("/hd2/pengbo/allProjects/mesh_reconstruction/data/examples/car_out_0.obj", normalization=False)
# # vertices0, faces0 = nr.load_obj("/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN_LSM1112/sample3D_02691156/20000.obj", normalization=False)
# # # vertices1, faces1 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06500.obj')
# # # vertices2, faces2 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06900.obj')
# # # vertices = torch.cat((vertices0[None, :, :], vertices1[None, :, :], vertices2[None, :, :]), 0)
# # # faces = torch.cat((faces0[None, :, :], faces1[None, :, :], faces2[None, :, :]), 0)
# # print(torch.max(vertices0))
# # vertices = vertices0[None,:,:]
# # faces = faces0[None,:,:]
# # faces = nr.vertices_to_faces(vertices, faces).data
# # faces = faces * 1. * (32. - 1) / 32. + 0.5  # normalization
# # voxels = voxelization.voxelize(faces, 32, normalize=False)
# # # and plot everything
# # for i in range(voxels.shape[0]):
# #     fig = plt.figure()
# #     ax = fig.gca(projection='3d')
# #     ax.voxels(voxels[i,:,:,:].squeeze().cpu().numpy(), facecolors='green', edgecolor='k')
# #
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

# ##################################################################
# # fetch some random test images and save
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# CLASS_IDS_ALL = (
#     '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
#     '03691459,04090263,04256520,04379243,04401088,04530566')
# data_dir = '/hd2/pengbo/mesh_reconstruction/dataset/'
# save_dir = '/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples2/'
#
# N = 10
# os.makedirs(save_dir, exist_ok=True)
# for class_id in CLASS_IDS_ALL.split(','):
#     print('loading dataset %s ...'%class_id)
#     imgs = np.load(os.path.join(data_dir, class_id+'_val_images.npz'))['arr_0']
#     imgs = imgs.reshape((-1, 4, 64, 64))
#     n_imgs = imgs.shape[0]
#     rand_idx = np.random.permutation(range(n_imgs))[0:N]
#     rand_imgs = imgs[rand_idx, :, :, :]
#     for i in range(N):
#         img = rand_imgs[i,:,:,:].transpose((1,2,0))
#         plt.imsave(os.path.join(save_dir, '%s_%d.png'%(class_id, i)), img)

## from val-logs save best and last models and delete the others.
# import os
# import re
# import numpy as np
# import subprocess
#
# CLASS_IDS_ALL = '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' + \
#                 '03691459,04090263,04256520,04379243,04401088,04530566'
# for class_id in CLASS_IDS_ALL.split(','):
#     folder = os.path.join('/hd2/pengbo/mesh_reconstruction/models/AE1', 'sample3D_%s_AE1' % class_id)
#     dest = os.path.join('/hd2/pengbo/mesh_reconstruction/models/AE1', 'sample3D_%s_Single-VAE' % class_id)
#     # fn = os.path.join(folder, 'val_log.txt')
#     # f_log = open(fn, 'r')
#     # batches = []
#     # ious = []
#     # for line in f_log.readlines():
#     #     sep = re.split('[ ,\n]', line)  # try out the split regexp
#     #     batches.append(sep[1])
#     #     ious.append(float(sep[5]))
#     # f_log.close()
#     # argmax = np.argmax(np.array(ious))
#     # ckptG_best = os.path.join(folder, batches[argmax]+'-G.ckpt')
#     # ckptE_best = os.path.join(folder, batches[argmax] + '-E.ckpt')
#     # ckptG_last = os.path.join(folder, batches[-1] + '-G.ckpt')
#     # ckptE_last = os.path.join(folder, batches[-1] + '-E.ckpt')
#     # print('best %s' % ckptG_best)
#     # print('last %s' % ckptG_last)
#     # res = subprocess.check_output(['cp', ckptG_best, os.path.join(folder, 'best-G.ckpt')])
#     # for line in res.splitlines():
#     #     print(line)
#     # res = subprocess.check_output(['cp', ckptE_best, os.path.join(folder, 'best-E.ckpt')])
#     # for line in res.splitlines():
#     #     print(line)
#     # res = subprocess.check_output(['cp', ckptG_last, os.path.join(folder, 'last-G.ckpt')])
#     # for line in res.splitlines():
#     #     print(line)
#     # res = subprocess.check_output(['cp', ckptE_best, os.path.join(folder, 'last-E.ckpt')])
#     # for line in res.splitlines():
#     #     print(line)
#     try:
#         # res = subprocess.check_output('rm '+os.path.join(folder, '[0-9]*'), shell=True)
#         res = subprocess.check_output(['mv', folder, dest])
#         for line in res.splitlines():
#             print(line)
#     except:
#         pass

# import matplotlib.pyplot as plt
# cm = plt.cm.get_cmap('RdYlBu')
# xy = range(20)
# z = xy
# sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
# plt.colorbar(sc)
# plt.show()

# # Test Pascal VOC dataset from Kato's view_prior_learning (https://github.com/hiroharu-kato/view_prior_learning/blob/6f1afa8811180a47334cddc62112d7cfb8b2ceca/mesh_reconstruction/dataset_pascal.py)
# # This is for testing the coordinate system conversion
# from data_loader import Pascal
# import torch
# import neural_renderer as nr
# import models as M
#
# device = torch.device('cuda:0')
# dataDir = "/hd1/pengbo/Pascal3D_Kato/"
# class_ids = ['02691156']
# load_E = "/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D/ckpt3D_02691156/last-E.ckpt"
# load_G = "/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D/ckpt3D_02691156/last-G.ckpt"
# latent_dim = 512
# batch_size = 1
#
# def tensor2img(tensor):
#     img = tensor.cpu().numpy()
#     img = img.transpose((1,2,0))
#     return img
#
# dataset = Pascal(dataDir, class_ids, 'val')
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# encoder = M.Encoder(3, dim_out=latent_dim)
# mesh_generator = M.Mesh_Generator(latent_dim, 'sphere_642.obj')
# mesh_generator.to(device)
# encoder.to(device)
# encoder.load_state_dict(torch.load(load_E))
# mesh_generator.load_state_dict(torch.load(load_G))
#
# with torch.no_grad():
#     images_in, rotation_matrices, _, voxels = next(iter(data_loader))
#     print(images_in.shape)
#     print(rotation_matrices.shape)
#     images_in = images_in.to(device)
#     rotation_matrices = rotation_matrices.to(device)
#
#     input_imgs = images_in[:, 0:3, :, :]
#     z = encoder(input_imgs)
#     vertices, faces = mesh_generator(z)
#     faces_v = nr.vertices_to_faces(vertices, faces).data
#     # faces_v = faces_v * 1. * (32. - 1) / 32. + 0.5  # normalization
#     faces_v = (faces_v + 1.0) * 0.5  # normalization
#     voxels_predicted = voxelization.voxelize(faces_v, 32, False)
#     voxels_predicted = voxels_predicted.transpose(2, 3).flip([1, 2, 3])
#     ious = torch.Tensor.float(voxels * voxels_predicted.cpu()).sum((1, 2, 3)) / \
#                 torch.Tensor.float(0 < (voxels + voxels_predicted.cpu())).sum((1, 2, 3))
#     print('mean iou is : %f' % (ious.sum().item()/batch_size))
#
#     mesh_renderer = M.Mesh_Renderer(vertices, faces, dataset='Pascal3D', mode='rgb').to(device)
#     gen_imgs = mesh_renderer(rotation_matrices)
#
# plt.figure(0)
# plt.imshow(tensor2img(input_imgs[0,:,:,:]))
# plt.title('input image')
#
# plt.figure(1)
# plt.imshow(tensor2img(gen_imgs[0,:,:,:]))
# plt.title('gen image')
#
# fig = plt.figure(2)
# ax = fig.gca(projection='3d')
# voxel = voxels[0,:,:,:].numpy()
# ax.voxels(voxel, facecolors='red', edgecolor='k')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.plot([16,32], [16,16], [16,16], 'r')
# plt.plot([16,16], [16,32], [16,16], 'g')
# plt.plot([16,16], [16,16], [16,32], 'b')
# ax.view_init(50, -50)
# plt.title('gt voxel')
#
# fig = plt.figure(3)
# ax = fig.gca(projection='3d')
# voxel = voxels_predicted[0, :, :, :].cpu().numpy()
# ax.voxels(voxel, facecolors='red', edgecolor='k')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.plot([16, 32], [16, 16], [16, 16], 'r')
# plt.plot([16, 16], [16, 32], [16, 16], 'g')
# plt.plot([16, 16], [16, 16], [16, 32], 'b')
# ax.view_init(50, -50)
# plt.title('pred voxel')
#
# plt.show()

# Test PrGAN voxel VS. cvpr18's voxel coordinates
voxel_dir_prgan = "/hd2/pengbo/allProjects/PrGAN/results/PrGAN02691156/volume00000.npy"
voxel_prgan = np.load(voxel_dir_prgan)
print(voxel_prgan.shape)
fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.voxels(voxel_prgan, facecolors='red', edgecolor='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()