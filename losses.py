import numpy as np
import torch


def smoothness_loss_parameters(faces, fn):
    # if hasattr(faces, 'get'):
    #     faces = faces.get()
    vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

    v0s = np.array([v[0] for v in vertices], 'int32')
    v1s = np.array([v[1] for v in vertices], 'int32')
    v2s = []
    v3s = []
    for v0, v1 in zip(v0s, v1s):
        count = 0
        for face in faces:
            if v0 in face and v1 in face:
                v = np.copy(face)
                v = v[v != v0]
                v = v[v != v1]
                if count == 0:
                    v2s.append(int(v[0]))
                    count += 1
                else:
                    v3s.append(int(v[0]))
    v2s = np.array(v2s, 'int32')
    v3s = np.array(v3s, 'int32')
    np.save(fn, (v0s, v1s, v2s, v3s))
    return v0s, v1s, v2s, v3s


def smoothness_loss(vertices, parameters, eps=1e-6):
    # make v0s, v1s, v2s, v3s
    v0s, v1s, v2s, v3s = parameters
    batch_size = vertices.shape[0]

    v0s = vertices[:, v0s, :]
    v1s = vertices[:, v1s, :]
    v2s = vertices[:, v2s, :]
    v3s = vertices[:, v3s, :]

    a1 = v1s - v0s
    b1 = v2s - v0s
    a1l2 = torch.sum(a1**2, dim=-1)
    b1l2 = torch.sum(b1**2, dim=-1)
    a1l1 = torch.sqrt(a1l2 + eps)
    b1l1 = torch.sqrt(b1l2 + eps)
    ab1 = torch.sum(a1 * b1, dim=-1)
    cos1 = ab1 / (a1l1 * b1l1 + eps)
    sin1 = torch.sqrt(1 - cos1**2 + eps)
    c1 = a1*((ab1 / (a1l2 + eps))[:, :, None].repeat(1,1,a1.shape[2]))
    cb1 = b1 - c1
    cb1l1 = b1l1 * sin1

    a2 = v1s - v0s
    b2 = v3s - v0s
    a2l2 = torch.sum(a2**2, dim=-1)
    b2l2 = torch.sum(b2**2, dim=-1)
    a2l1 = torch.sqrt(a2l2 + eps)
    b2l1 = torch.sqrt(b2l2 + eps)
    ab2 = torch.sum(a2 * b2, dim=-1)
    cos2 = ab2 / (a2l1 * b2l1 + eps)
    sin2 = torch.sqrt(1 - cos2**2 + eps)
    c2 = a2*((ab2 / (a2l2 + eps))[:, :, None].repeat(1,1,a2.shape[2]))
    cb2 = b2 - c2
    cb2l1 = b2l1 * sin2

    cos = torch.sum(cb1 * cb2, dim=-1) / (cb1l1 * cb2l1 + eps)

    loss = torch.sum((cos + 1)**2) / batch_size
    # loss = torch.sum(torch.abs(cos + 1)) / batch_size
    return loss


def inflation_loss(vertices, faces, eps=1e-5):
    faces = torch.Tensor.long(faces[0])
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 2

    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    batch_size, num_faces = v0.shape[:2]
    v0 = torch.reshape(v0, (batch_size * num_faces, 3))
    v1 = torch.reshape(v1, (batch_size * num_faces, 3))
    v2 = torch.reshape(v2, (batch_size * num_faces, 3))
    norms = torch.cross(v1 - v0, v2 - v0)  # [bs * nf, 3]
    norms_norm = torch.norm(norms, dim=1, keepdim=True)
    norms_norm = norms_norm.repeat(1, 3)
    norms = norms/norms_norm
    v0_t = (v0 + norms).detach()
    v1_t = (v1 + norms).detach()
    v2_t = (v2 + norms).detach()
    loss_v0 = torch.sum(torch.sqrt(torch.sum(torch.pow(v0_t - v0, 2), 1) + eps))
    loss_v1 = torch.sum(torch.sqrt(torch.sum(torch.pow(v1_t - v1, 2), 1) + eps))
    loss_v2 = torch.sum(torch.sqrt(torch.sum(torch.pow(v2_t - v2, 2), 1) + eps))
    loss = loss_v0 + loss_v1 + loss_v2
    loss /= batch_size
    return loss


def iou(data1, data2, p):
    # target, prediction
    axes = tuple(range(data1.ndimension())[1:])
    intersection = torch.sum(data1 * data2, dim=axes)
    union = torch.sum(data1 + data2 - data1 * data2, dim=axes)
    if p is None:
        loss = torch.sum(intersection / union) / intersection.numel()
    else:
        loss = torch.sum(p * intersection / union) / (intersection.numel()/24.)

    return loss


def iou_loss(data1, data2, p=None):
    return 1 - iou(data1, data2, p)


def Laplacian_loss_parameters(n_vertices, faces, fn):
    # extract per vertex's neighboring vertices
    # input "faces" needs to be np.array[[],[]...]
    per_vert_neighbors = []
    edges = []
    for i in range(n_vertices):
        per_vert_neighbors.append(set())
    for face in faces:
        per_vert_neighbors[face[0]].add(face[1])
        per_vert_neighbors[face[0]].add(face[2])
        per_vert_neighbors[face[1]].add(face[0])
        per_vert_neighbors[face[1]].add(face[2])
        per_vert_neighbors[face[2]].add(face[0])
        per_vert_neighbors[face[2]].add(face[1])

    idxs = [list(vert_neighbors) for vert_neighbors in per_vert_neighbors]
    degrees = [idx.__len__() for idx in idxs]
    where_6 = np.where(np.array(degrees) == 6)[0]
    where_5 = np.where(np.array(degrees) == 5)[0]
    assert where_5.size + where_6.size == degrees.__len__()  # all vertices need to have a degree of 5 or 6
    idxs_5 = np.array([idx[0:5] for idx in idxs])
    idxs_6 = np.array(
        [idx[5] for idx in idxs if idx.__len__() == 6])  # idxs is like [[6-list], [6-list], [5-list], ...]

    for i in range(idxs.__len__()):
        for j in idxs[i]:
            if i<j:         # think about adjacency matrix is symetric
                edges.append([i,j])
    edges = np.array(edges)

    np.save(fn, (idxs_5, idxs_6, where_5, where_6, edges))
    return idxs_5, idxs_6, where_5, where_6, edges


def Laplacian_edge_loss(vertices, parameters):
    idxs_5, idxs_6, where_5, where_6, edges = parameters
    where_5 = torch.from_numpy(where_5).to('cuda:0')
    where_6 = torch.from_numpy(where_6).to('cuda:0')
    v0s = vertices[:, idxs_5[:, 0], :]
    v1s = vertices[:, idxs_5[:, 1], :]
    v2s = vertices[:, idxs_5[:, 2], :]
    v3s = vertices[:, idxs_5[:, 3], :]
    v4s = vertices[:, idxs_5[:, 4], :]
    v5s = vertices[:, idxs_6, :]

    vertices_pred_5 = (v0s.index_select(1,where_5)+v1s.index_select(1,where_5)+v2s.index_select(1,where_5)+ \
                       v3s.index_select(1,where_5)+v4s.index_select(1,where_5))/5.
    vertices_pred_6 = (v0s.index_select(1,where_6)+v1s.index_select(1,where_6)+v2s.index_select(1,where_6)+ \
                       v3s.index_select(1,where_6)+v4s.index_select(1,where_6)+v5s)/6.
    loss_5 = torch.sum((vertices.index_select(1,where_5) - vertices_pred_5)**2)
    loss_6 = torch.sum((vertices.index_select(1,where_6) - vertices_pred_6)**2)
    batch_size = vertices.shape[0]
    Lap_loss = (loss_5 + loss_6)/batch_size

    v0s = vertices[:, edges[:,0], :]
    v1s = vertices[:, edges[:,1], :]
    # edge_loss = torch.sum((v0s - v1s)**2)/batch_size
    edge_lens = torch.sqrt(torch.sum((v0s - v1s)**2, 2))
    mean_len = torch.mean(edge_lens, 1, keepdim=True)
    edge_loss = torch.sum((edge_lens - mean_len.repeat((1, edge_lens.shape[1])))**2)
    return Lap_loss, edge_loss


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