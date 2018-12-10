import torchvision.transforms as T
from torchvision import datasets
import torch.utils.data as data
import torch
import numpy as np
from torchvision.utils import  save_image
import os
import glob
from PIL import Image

def rescale(x):
    # rescale image to [-1, 1]
    range = x.max()-x.min()
    x = (x-x.min())/range*2-1
    return x


def get_CelebA_loader(data_dir, img_size, crop_size, batch_size, n_cpu):
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(img_size))
    transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform.append(T.Lambda(lambda x:rescale(x)))
    transform = T.Compose(transform)
    celebA = datasets.ImageFolder(data_dir, transform)
    CelebA_loader = data.DataLoader(celebA, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    return CelebA_loader


def generate(G, N, data_dir, cuda, batch_size, latent_dim, device):
    # this function generate a dataset using given G network
    if os.path.isdir(data_dir):
        print('GAN image dataset already exists in {}'.format(data_dir))
        pass
    else:
        os.makedirs(data_dir, exist_ok=False)
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        k = 0
        for i in range(int(np.ceil(N/batch_size))):
            z = Tensor(np.random.normal(0, 1, (batch_size, latent_dim)), device=device)
            imgs = G(z)
            for j in range(batch_size):
                save_image(torch.squeeze(imgs[j,]), os.path.join(data_dir, '%06d.jpg' % k), nrow=1, padding=0,
                           normalize=True)
                k = k+1
                if k%1000 == 0:
                    print('Generated and saved {} images...'.format(k))
                if k == N:
                    break

class ImageFolderSingle(data.Dataset):
    # This data class returns images in a single folder and split it into train-test parts
    def __init__(self, data_dir, train_perc, mode, transform, label):
        # label should be one int, [0, 1]
        self.data_dir = data_dir
        self.train_perc = train_perc
        self.mode = mode
        self.transform = transform
        self.label = label
        self.img_paths = []
        for f in glob.glob(os.path.join(data_dir, '*.jpg')):
            self.img_paths.append(f)

    def __len__(self):
        N = len(self.img_paths)
        if self.mode is 'train':
            return int(np.ceil(N*self.train_perc))
        elif self.mode is 'test':
            return N-int(np.ceil(N*self.train_perc))
        else:
            exit('ImageFolderSingle mode must be one of [train test]')

    def __getitem__(self, index):
        N = len(self.img_paths)
        if self.mode is 'train':
            index_ = index
        elif self.mode is 'test':
            index_ = index+int(np.ceil(N*self.train_perc))
        else:
            exit('GAN_data mode must be one of [train test]')
        image = Image.open(self.img_paths[index_])
        return self.transform(image), torch.FloatTensor(1).fill_(self.label)
