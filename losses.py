import numpy as np
import torch


def smoothness_loss_parameters(faces):
    if hasattr(faces, 'get'):
        faces = faces.get()
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
    np.save('smoothness_params', (v0s, v1s, v2s, v3s))
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


def Laplacian_loss_parameters(n_vertices, faces):
    # extract per vertex's neighboring vertices
    # input "faces" needs to be np.array[[],[]...]
    per_vert_neighbors = []
    for i in range(n_vertices):
        per_vert_neighbors.append(set())
    for face in faces:
        per_vert_neighbors[face[0]].add(face[1])
        per_vert_neighbors[face[0]].add(face[2])
        per_vert_neighbors[face[1]].add(face[0])
        per_vert_neighbors[face[1]].add(face[2])
        per_vert_neighbors[face[2]].add(face[0])
        per_vert_neighbors[face[2]].add(face[1])
    return per_vert_neighbors


def Laplacian_loss(vertices, parameters):
    