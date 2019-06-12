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

import torch
import random
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算


SAMPLE_SIZE = 500
buckets = 50

#第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
plt.subplot(1,2,1)
plt.xlabel("random.lognormalvariate")
mu = -0.6
sigma = 0.15#将输出数据限制到0-1之间
res1 = [random.lognormvariate(mu, sigma) for _ in range(SAMPLE_SIZE)]
plt.hist(res1, buckets)

#第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
plt.subplot(1,2,2)
plt.xlabel("random.betavariate")
alpha = 1
beta = 10
res2 = [random.betavariate(alpha, beta) for _ in range(SAMPLE_SIZE)]
plt.hist(res2, buckets)
plt.show()


#参数值见上段代码
#分别从对数正态分布和beta分布取两组数据
diff_1 = []
for i in range(10):
    diff_1.append([random.lognormvariate(mu, sigma) for _ in range(SAMPLE_SIZE)])

diff_2 = []
for i in range(10):
    diff_2.append([random.betavariate(alpha, beta) for _ in range(SAMPLE_SIZE)])

X = torch.Tensor(diff_1)
Y = torch.Tensor(diff_2)
X,Y = Variable(X), Variable(Y)
print(mmd_rbf(X,Y))

#参数值见以上代码
#从对数正态分布取两组数据
same_1 = []
for i in range(10):
    same_1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

same_2 = []
for i in range(10):
    same_2.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

X = torch.Tensor(same_1)
Y = torch.Tensor(same_2)
X,Y = Variable(X), Variable(Y)
print(mmd_rbf(X,Y))
