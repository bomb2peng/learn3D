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
import json
import scipy.io as sio
import math

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
    def __init__(self, data_source, batch_size, class_id, nViews=24):
        self.data_source = data_source
        self.batch_size = batch_size
        self.class_id = class_id
        self.nViews = nViews

    def __iter__(self):
        data_ids = np.arange(self.data_source.num_data[self.class_id]) + self.data_source.pos[self.class_id]
        viewpoint_ids = np.tile(np.arange(self.nViews), data_ids.size)
        data_ids = np.repeat(data_ids, self.nViews) * self.nViews + viewpoint_ids
        for i in range(self.__len__()):
            img_ids = data_ids[i * self.batch_size:min((i + 1) * self.batch_size,
                                                       self.data_source.num_data[self.class_id]*self.nViews)]
            yield img_ids

    def __len__(self):
        return (self.data_source.num_data[self.class_id]*self.nViews-1) // self.batch_size +1


def get_split(split_js='data/splits.json'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, split_js), 'r') as f:
        js = json.load(f)
    return js

def viewpoint2bin(azimuth, elevation, wAzimuth, wElevation):
    nAzimuth = math.ceil(360/wAzimuth)
    nElevation = math.ceil(60/wElevation)     # actual dataset view range is different from described in paper, elevation is [-20, 40)
    if elevation == 40:
        elevation = 39.9
    if azimuth == 180:
        azimuth = 179.9
    bin = ((azimuth+180)//wAzimuth)*nElevation + (elevation+20)//wElevation
    bin = int(bin)
    # print([azimuth, elevation, bin])
    assert bin < nAzimuth*nElevation
    return bin

class ShapeNet_LSM(data.Dataset):
    """dataset from NIPS17 paper LSM: https://github.com/akar43/lsm/blob/01edb3ce70a989207fd843bacf7693c057eb073e/shapenet.py#L13"""
    def __init__(self, dataDir=None, splitFile=None, class_ids=None, set_name=None, img_resize=64, N_views=1,
                 wAzimuth=15, wElevation=10):
        self.set_name = set_name
        self.img_resize = img_resize
        self.N_views = N_views
        self.splits_all = get_split(splitFile)
        self.class_ids = (self.splits_all.keys()
                          if class_ids is None else class_ids)
        self.splits = {k: self.splits_all[k] for k in self.class_ids}
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        self.imgDirs = []
        self.depDir = []
        self.views = []
        self.voxelDirs = []
        self.num_data = {}
        self.pos = {}
        self.wAzimuth=wAzimuth
        self.wElevation=wElevation
        count = 0
        for class_id in loop:
            obj_ids = self.splits[class_id][set_name]
            for obj_id in obj_ids:
                renderDir = os.path.join(dataDir, 'renders', class_id, obj_id)
                voxelDir = os.path.join(dataDir, 'voxels', 'modelVoxels32', class_id, '%s.mat' % obj_id)
                self.voxelDirs.append(voxelDir)
                viewDir = os.path.join(renderDir, 'view.txt')
                views = np.loadtxt(viewDir)
                for i in range(N_views):
                    self.imgDirs.append(os.path.join(renderDir, 'render_%d.png' % i))
                    self.depDir.append(os.path.join(renderDir, 'depth_%d.png' % i))
                    self.views.append(views[i,:])
            self.num_data[class_id] = len(obj_ids)
            self.pos[class_id] = count
            count = count + self.num_data[class_id]
        # views = np.array(self.views)
        # print(np.min(views, axis=0))
        # print(np.max(views, axis=0))

    def __len__(self):
        N = len(self.imgDirs)
        return N

    def __getitem__(self, item):
        img = cv2.imread(self.imgDirs[item]).astype('float32') / 255.
        depth = cv2.imread(self.depDir[item], cv2.IMREAD_GRAYSCALE)
        mask = np.float32(depth < 250)[:,:,np.newaxis]
        image = np.concatenate((img, mask), 2)
        image = skT.resize(image, (self.img_resize,self.img_resize), anti_aliasing=True)
        image = np.float32(image.transpose((2,0,1)))
        imageT = torch.from_numpy(image)
        viewpoints = nr.get_points_from_angles(self.views[item][3], self.views[item][1], self.views[item][0])
        view_id = viewpoint2bin(self.views[item][0], self.views[item][1], self.wAzimuth, self.wElevation)
        voxel = sio.loadmat(self.voxelDirs[item//self.N_views])['Volume']
        voxelT = torch.from_numpy(voxel.astype(np.int32))
        return imageT, torch.Tensor(viewpoints), view_id, voxelT


import numpy as np
import os
import scipy.misc


def crop_image(image_in, image_ref, rotation, bounding_box, padding=0.15, jitter=0.15, flip=True, image_size=224):
    # based on https://github.com/shubhtuls/drc/blob/master/utils/cropUtils.lua

    left, top, width_box, height_box = bounding_box
    height_image, width_image = image_in.shape[:2]

    # random cropping
    y_min = int(top + (np.random.uniform(-jitter, jitter) - padding) * height_box)
    y_max = int(top + height_box + (np.random.uniform(-jitter, jitter) + padding) * height_box)
    x_min = int(left + (np.random.uniform(-jitter, jitter) - padding) * width_box)
    x_max = int(left + width_box + (np.random.uniform(-jitter, jitter) + padding) * width_box)
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(height_image, y_max)
    x_max = min(width_image, x_max)
    image_in = image_in[y_min:y_max, x_min:x_max]

    # random flipping
    if flip:
        if np.random.uniform(0, 1) < 0.5:
            image_in = image_in[:, ::-1]

    image_in = scipy.misc.imresize(image_in, (image_size, image_size)).astype('float32') / 255.
    # image_in = image_in.astype('float32') / 255.
    image_in = image_in.transpose((2, 0, 1))
    return image_in, image_ref, rotation


class Pascal(object):
    def __init__(self, directory, class_ids, set_name):
        self.name = 'pascal'
        self.directory = directory
        self.class_ids = class_ids
        self.set_name = set_name
        self.image_size = 224

        self.images_original = {}
        self.images_ref = {}
        self.bounding_boxes = {}
        self.rotation_matrices = {}
        self.voxels = {}
        self.num_data = {}
        for class_id in class_ids:
            data = np.load(os.path.join(directory, '%s_%s.npz' % (class_id, set_name)), allow_pickle=True, encoding="latin1")
            self.images_original[class_id] = data['images']
            self.images_ref[class_id] = data['images_ref']
            self.bounding_boxes[class_id] = data['bounding_boxes']
            self.rotation_matrices[class_id] = data['rotation_matrices']
            self.voxels[class_id] = data['voxels']
            if set_name == 'train':
                # add ImageNet data
                data = np.load(os.path.join(directory, '%s_%s.npz' % (class_id, 'imagenet')), encoding="latin1")
                self.images_original[class_id] = np.concatenate(
                    (self.images_original[class_id], data['images']), axis=0)
                self.images_ref[class_id] = np.concatenate(
                    (self.images_ref[class_id], data['images_ref']), axis=0)
                self.bounding_boxes[class_id] = np.concatenate(
                    (self.bounding_boxes[class_id], data['bounding_boxes']), axis=0)
                self.rotation_matrices[class_id] = np.concatenate(
                    (self.rotation_matrices[class_id], data['rotation_matrices']), axis=0)
                self.voxels[class_id] = np.concatenate((self.voxels[class_id], data['voxels']), axis=0)
            self.images_ref[class_id] = self.images_ref[class_id].transpose((0, 3, 1, 2))
            self.num_data[class_id] = self.images_ref[class_id].shape[0]

    def get_random_batch(self, batch_size):
        labels = np.zeros(batch_size, 'int32')
        images_in = np.zeros((batch_size, 3, self.image_size, self.image_size), 'float32')
        images_ref = np.zeros((batch_size, 4, self.image_size, self.image_size), 'float32')
        rotation_matrices = np.zeros((batch_size, 3, 3), 'float32')
        rotation_matrices_random = np.zeros((batch_size, 3, 3), 'float32')
        voxels = np.zeros((batch_size, 32, 32, 32), 'float32')
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_num = np.random.choice(range(self.num_data[class_id]))
            # object_num = 0
            image_in = self.images_original[class_id][object_num]
            image_ref = self.images_ref[class_id][object_num].astype('float32') / 255.
            rotation = self.rotation_matrices[class_id][object_num]
            bonding_box = self.bounding_boxes[class_id][object_num]
            images_in[i], images_ref[i], rotation_matrices[i] = crop_image(
                image_in,
                image_ref,
                rotation,
                bonding_box,
                padding=0.15,
                jitter=0.15,
                flip=False,
            )
            labels[i] = self.class_ids.index(class_id)
            voxels[i] = self.voxels[class_id][object_num]

        rms = [rotation_matrices[np.nonzero(labels == i)[0]] for i in range(3)]
        counts = np.zeros(3, 'int32')
        for i in range(batch_size):
            label = labels[i]
            rotation_matrices_random[i] = rms[label][(counts[label] + 1) % len(rms[label])].copy()
            counts[label] += 1

        # delete masked pixels
        images_ref[:, :3, :, :] *= images_ref[:, 3][:, None, :, :]

        return images_in, images_ref, rotation_matrices, rotation_matrices_random, labels, voxels

    def get_all_batches_for_evaluation(self, batch_size, class_id, num_views=None):
        num_objects = self.num_data[class_id]
        for batch_num in range((num_objects - 1) / batch_size + 1):
            batch_size2 = min(num_objects - batch_num * batch_size, batch_size)
            images_in = np.zeros((batch_size2, 3, self.image_size, self.image_size), 'float32')
            voxels = np.zeros((batch_size2, 32, 32, 32), 'float32')
            for i in range(batch_size2):
                object_num = batch_num * batch_size + i
                image_in = self.images_original[class_id][object_num]
                image_ref = self.images_ref[class_id][object_num].astype('float32') / 255.
                rotation = self.rotation_matrices[class_id][object_num]
                bounding_box = self.bounding_boxes[class_id][object_num]
                images_in[i], _, _ = crop_image(
                    image_in,
                    image_ref,
                    rotation,
                    bounding_box,
                    padding=0.15,
                    jitter=0,
                    flip=False,
                )
                voxels[i] = self.voxels[class_id][object_num].astype('float32')

            yield images_in, voxels