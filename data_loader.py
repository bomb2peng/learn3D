import torchvision.transforms as T
from torchvision import datasets
import torch.utils.data as data
import torch
import numpy as np
from torchvision.utils import save_image
import os
import glob
from PIL import Image
import cv2
import random
import tqdm
import neural_renderer as nr
import skimage.transform as skT


class ShapeNet(data.Dataset):
    def __init__(self, directory=None, class_ids=None, set_name=None, img_resize=64):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732
        self.img_resize = img_resize

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name)))['arr_0'])
            voxels.append(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name)))['arr_0'])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64))
        images = np.ascontiguousarray(images)
        # images = images[range(5,images.shape[0],24),:,:,:]      # only take a certain pose as training data. REMEMBER
        #  the #*24 in line 188
        self.images = images
        self.n_objects = count
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels

    def __len__(self):
        N = 0
        for class_id in self.class_ids:
            N = N + self.num_data[class_id] *24
        return N

    def __getitem__(self, item):
        image = self.images[item,:,:,:].astype('float32') / 255.
        image = skT.resize(image.transpose((1,2,0)), (self.img_resize,self.img_resize), anti_aliasing=True)
        image = np.float32(image.transpose((2,0,1)))
        imageT = torch.from_numpy(image)
        view_id = item % 24
        viewpoints = nr.get_points_from_angles(self.distance, self.elevation, -view_id * 15)
        voxel = self.voxels[item//24,:,:,:]
        voxelT = torch.from_numpy(voxel.astype(np.int32))
        return imageT, torch.Tensor(viewpoints), view_id, voxelT


class ShapeNet_Sampler_Batch(data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        if self.batch_size%2 is not 0:
            raise ValueError("batch_size needs to be an even number, but got {}".format(self.batch_size))

        n_objects = self.batch_size//2
        data_ids_a = []
        data_ids_b = []
        for i in range(len(self.data_source)//2):
            class_id = np.random.choice(self.data_source.class_ids)
            object_id = np.random.randint(0, self.data_source.num_data[class_id])
            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.data_source.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.data_source.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a.append(data_id_a)
            data_ids_b.append(data_id_b)
            if len(data_ids_a) == n_objects:
                data_ids = data_ids_a + data_ids_b
                yield data_ids
                data_ids_a = []
                data_ids_b = []

    def __len__(self):
        return (len(self.data_source)-1)//self.batch_size + 1


class ShapeNet_sampler_all(data.Sampler):
    def __init__(self, data_source, batch_size, class_id):
        self.data_source = data_source
        self.batch_size = batch_size
        self.class_id = class_id

    def __iter__(self):
        data_ids = np.arange(self.data_source.num_data[self.class_id]) + self.data_source.pos[self.class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids
        for i in range(self.__len__()):
            img_ids = data_ids[i * self.batch_size:min((i + 1) * self.batch_size,
                                                       self.data_source.num_data[self.class_id]*24)]
            yield img_ids

    def __len__(self):
        return (self.data_source.num_data[self.class_id]*24-1) // self.batch_size +1