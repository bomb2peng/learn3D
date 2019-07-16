## Learn3D Introduction

Learning category-specific 3D object shapes using 2D images is an active research topic, with merits of not requiring expensive 3D supervision. However, most work in this direction assumes there exist multiple images of the same instance from different views, which oftentimes does not apply in practice. In this paper, we explore a more challenging problem of learning 3D shape models from single-view images of the same object category. We show that this a very hard problem even with the knowledge of  viewpoint information. The major difficulty lies in insufficient constraints on shape provided by single view images, which leads to pose entanglement in learned shape space. To address this problem, we take a novel domain adaptation perspective, and propose an effective adversarial domain confusion training method for learning pose-invariant compact shape space. Experiments on single-view shape reconstruction obtain promising results that demonstrate effectiveness of proposed method.

## Requirements

Python 3.x

Pytorch 0.4.x

[Neural_renderer](https://github.com/daniilidis-group/neural_renderer)

[visdom](https://github.com/facebookresearch/visdom)

## Dataset

We use the same dataset as Kato's. To download the dataset, go to [mesh_reconstruction](https://github.com/hiroharu-kato/mesh_reconstruction) project and see *download_dataset.sh*

## How to Run

* First, compile the CUDA voxelization code which converts mesh models to voxel models. This CUDA code is borrowed from [here](https://github.com/hiroharu-kato/mesh_reconstruction/blob/master/mesh_reconstruction/voxelization.py). To compile, 

   `cd ./cuda`

  `python setup.py`

* To train, evaluate, reconstruct, t-SNE visualize ..., run *train.sh* and read the comments with codes.

