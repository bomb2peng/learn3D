# # test the cuda module "voxelization"
# import torch
# import neural_renderer as nr
# import voxelization
# import matplotlib.pyplot as plt
# # This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#
# vertices0, faces0 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06000.obj')
# vertices1, faces1 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06500.obj')
# vertices2, faces2 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06900.obj')
# vertices = torch.cat((vertices0[None, :, :], vertices1[None, :, :], vertices2[None, :, :]), 0)
# faces = torch.cat((faces0[None, :, :], faces1[None, :, :], faces2[None, :, :]), 0)
# faces = nr.vertices_to_faces(vertices, faces)
# voxels = voxelization.voxelize(faces, 32, normalize=True)
#
# # and plot everything
# for i in range(voxels.shape[0]):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.voxels(voxels[i,:,:,:].squeeze().cpu().numpy(), facecolors='green', edgecolor='k')
#
# plt.show()

##################################################################
# fetch some random test images and save
import numpy as np
import os
import matplotlib.pyplot as plt

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
data_dir = '/hd2/pengbo/mesh_reconstruction/dataset/'
save_dir = '/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples_train/'

N = 10
os.makedirs(save_dir, exist_ok=True)
for class_id in CLASS_IDS_ALL.split(','):
    print('loading dataset %s ...'%class_id)
    imgs = np.load(os.path.join(data_dir, class_id+'_train_images.npz'))['arr_0']
    imgs = imgs.reshape((-1, 4, 64, 64))
    n_imgs = imgs.shape[0]
    rand_idx = np.random.permutation(range(n_imgs))[0:N]
    rand_imgs = imgs[rand_idx, :, :, :]
    for i in range(N):
        img = rand_imgs[i,:,:,:].transpose((1,2,0))
        plt.imsave(os.path.join(save_dir, '%s_%d.png'%(class_id, i)), img)