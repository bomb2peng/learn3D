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
    return loss


def iou(data1, data2):
    # target, prediction
    axes = tuple(range(data1.ndimension())[1:])
    intersection = torch.sum(data1 * data2, dim=axes)
    union = torch.sum(data1 + data2 - data1 * data2, dim=axes)
    return torch.sum(intersection / union) / intersection.numel()


def iou_loss(data1, data2):
    return 1 - iou(data1, data2)


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
