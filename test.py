# test the cuda module "voxelization"
import torch
import neural_renderer as nr
import voxelization
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

vertices0, faces0 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06000.obj')
vertices1, faces1 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06500.obj')
vertices2, faces2 = nr.load_obj('/hd2/pengbo/mesh_reconstruction/models/sample3D_04401088_AE1/random-06900.obj')
vertices = torch.cat((vertices0[None, :, :], vertices1[None, :, :], vertices2[None, :, :]), 0)
faces = torch.cat((faces0[None, :, :], faces1[None, :, :], faces2[None, :, :]), 0)
faces = nr.vertices_to_faces(vertices, faces)
voxels = voxelization.voxelize(faces, 32, normalize=True)

# and plot everything
for i in range(voxels.shape[0]):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels[i,:,:,:].squeeze().cpu().numpy(), facecolors='green', edgecolor='k')

plt.show()