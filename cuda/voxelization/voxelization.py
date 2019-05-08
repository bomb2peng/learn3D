import torch
import voxelization.cuda.voxelize_cuda as voxelize_cuda

def voxelize_sub1(faces, size, dim=2):
    assert (0 <= dim)
    bs, nf = faces.shape[:2]
    if dim == 0:
        i = torch.cuda.LongTensor([2, 1, 0])
        faces = faces[:, :, :, i]
    elif dim == 1:
        i = torch.cuda.LongTensor([0, 2, 1])
        faces = faces[:, :, :, i]
    voxels = torch.zeros((faces.shape[0], size, size, size), dtype=torch.int32).to('cuda:0')
    voxels = voxelize_cuda.sub1(faces, bs, nf, size, voxels)
    voxels = voxels.transpose(dim + 1, -1)
    return voxels

def voxelize_sub2(faces, size):
    bs, nf = faces.shape[:2]
    voxels = torch.zeros((faces.shape[0], size, size, size), dtype=torch.int32).to('cuda:0')
    voxels = voxelize_cuda.sub2(faces, bs, nf, size, voxels)
    return voxels

def voxelize_sub3(voxels):
    # fill in
    bs, vs = voxels.shape[:2]
    visible = torch.zeros_like(voxels)
    visible = voxelize_cuda.sub3_1(bs, vs, voxels, visible)

    sum_visible = visible.sum()
    while True:
        visible = voxelize_cuda.sub3_2(bs, vs, voxels, visible)
        if visible.sum() == sum_visible:
            break
        else:
            sum_visible = visible.sum()
    return 1 - visible

def voxelize(faces, size, normalize=False):
    faces = faces.clone()
    if normalize:
        min_vertice = faces.reshape((-1, 3)).min(0)[0]
        faces -= min_vertice[None, None, None, :]
        faces /= faces.max()
        faces *= 1. * (size - 1) / size
        margin = 1 - faces.reshape((-1, 3)).max(0)[0]
        faces += margin[None, None, None, :] / 2
        faces *= size
    else:
        faces *= size

    voxels0 = voxelize_sub1(faces, size, 0)
    voxels1 = voxelize_sub1(faces, size, 1)
    voxels2 = voxelize_sub1(faces, size, 2)
    voxels3 = voxelize_sub2(faces, size)
    voxels = voxels0 + voxels1 + voxels2 + voxels3
    voxels = torch.Tensor.int(0 < voxels)
    voxels = voxelize_sub3(voxels)

    return voxels
