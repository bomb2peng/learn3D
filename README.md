## Learn3D Introduction

Learning category-specific 3D object shapes using 2D images is an active research topic, with merits of not requiring expensive 3D supervision. However, most work in this direction assumes there exist multiple images of the same instance from different views, which oftentimes does not apply in practice. In this paper, we explore a more challenging problem of learning 3D shape models from single-view images of the same object category. We show that this a very hard problem even with the knowledge of  viewpoint information. The major difficulty lies in insufficient constraints on shape provided by single view images, which leads to pose entanglement in learned shape space. To address this problem, we take a novel domain adaptation perspective, and propose an effective adversarial domain confusion training method (denoted as 3D-AE-featGAN in code) for learning pose-invariant compact shape space. Experiments on single-view shape reconstruction obtain promising results that demonstrate effectiveness of proposed method. 

The reconstruction result of proposed method is visulized in the following compared to vinilla AE method.

![result](./reconCompare_whiteBG_sep.jpg)

## Requirements

Python 3.x

Pytorch 0.4.x

[Neural_renderer Pytorch version](https://github.com/daniilidis-group/neural_renderer)(If its API is changed, please try an older version. Mine version of NR is 8e2078754ea572aa9bcf0c09eb6793421ad9c2db)

[visdom](https://github.com/facebookresearch/visdom)

## Dataset

We use the same dataset as Kato's (denoted as "CVPR18" in this code). To download the dataset, go to [mesh_reconstruction](https://github.com/hiroharu-kato/mesh_reconstruction) project and see *download_dataset.sh*

We also use another dataset from [here](https://github.com/akar43/lsm/blob/01edb3ce70a989207fd843bacf7693c057eb073e/shapenet.py#L13) (denoted as NIPS17 in code) to compare with the CVPR19 paper (VPL).

## How to Run

* First, compile the CUDA voxelization code which converts mesh models to voxel models. This is for evaluation of reconstrction accuracy using voxel IoU. This CUDA code is borrowed from [here](https://github.com/hiroharu-kato/mesh_reconstruction/blob/master/mesh_reconstruction/voxelization.py). To compile, 

   `cd ./cuda`

  `python setup.py`

* To train, evaluate, reconstruct, t-SNE visualize ..., run *train.sh* and read the comments in "3D-GAN.py". More specifically:  
`python 3D-GAN.py --mode=train ...` for training our proposed model (denoted as "3D-AE-featGAN" in code)  
`python 3D-GAN.py --mode=trainCVPR19` for training the compared CVPR19 method VPL.  
`python 3D-GAN.py --mode=evaluation` evaluate reconstruction accuracy using voxel IoU   
`python 3D-GAN.py --mode=reconstruct` reconstruct sample images and save models for visulization   
`python 3D-GAN.py --mode=t_SNE` t_SNE visualization of learned shape embedings  
`python 3D-GAN.py --mode=interpolation` shape space interpolation visualization  
`python 3D-GAN.py --mode=MMD` caculate MMD distance of shape embeddings in different domains/poses

## Terms of Use
This code and its derived model is only for non-profit acdemic research purposes.

## Citation
If you find this code usful, please cite our paper as 
 
