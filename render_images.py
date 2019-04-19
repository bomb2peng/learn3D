import numpy as np
import torch
import neural_renderer as nr
import os
from torchvision.utils import  save_image

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
ShapeNet_dir = '/hd2/pengbo/ShapeNet/ShapeNetCore.v2/'
elevation = 30.
distance = 2.732
texture_size = 2
renderer = nr.Renderer(camera_mode='look_at', image_size=64, viewing_angle=15)

for class_id in CLASS_IDS_ALL.split(','):
    for obj_id in os.listdir(os.path.join(ShapeNet_dir, class_id)):
        vertices, faces = nr.load_obj(os.path.join(ShapeNet_dir, class_id, obj_id, \
                                                   'models/model_normalized.obj'))
        vertices = torch.Tensor.repeat(vertices[None,:,:], (24,1,1))
        vertices = vertices*0.5
        faces = torch.Tensor.repeat(faces[None,:,:], (24,1,1))
        textures = torch.ones(faces.shape[0], faces.shape[1], texture_size, texture_size, texture_size, \
                              3, dtype=torch.float32).cuda()
        azimuths = -15.*torch.arange(0,24)
        azimuths = torch.Tensor.float(azimuths)
        elevations = elevation*torch.ones((24))
        distances = distance*torch.ones((24))
        viewpoints = nr.get_points_from_angles(distances, elevations, azimuths)
        renderer.eye = viewpoints
        images_rgb = renderer(vertices, faces, textures, mode='rgb')
        images_silh = renderer(vertices, faces, textures, mode='silhouettes')
        images_silh = images_silh[:, None, :, :]
        images = torch.cat((images_rgb, images_silh), 1)
        save_image(images_rgb, './rendering_rgb.png', nrow=4)
        save_image(images_silh, './rendering_silh.png', nrow=4)