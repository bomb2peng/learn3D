#!/usr/bin/env bash

# training of DCGAN
#python 3D-GAN.py --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model_DCGAN2' \
#--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_DCGAN2' \
#--sample_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/sample_DCGAN2' --device_id=0 --model=DCGAN --n_epochs=40

## training of WGAN
#python 3D-GAN.py --model=WGAN --ckpt_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN \
#--log_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/log_WGAN --sample_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/sample_WGAN \
#--G_every=5 --sample_step=500 --lr=0.0001 --n_epochs=20 \
#--load_G=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN/121000-G.ckpt \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN/121000-D.ckpt --batches_done=121000

# training of 3D-GAN
# 02958343--car, 02691156--airplane, 02828884--bench, 02933112--dresser, 03001627--chair, 03211117--display,
# 03636649--lamp, 03691459--loudspeaker, 04090263--riffle, 04256520--sofa, 04379243--table, 04401088--telephone,
# 04530566--boat
CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=trainGAN --model=WGAN-GP --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
--log_dir=/hd2/pengbo/mesh_reconstruction/models/log3D_GAN --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_GAN \
--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_GAN  \
--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_162.obj \
--G_every=1 --sample_step=100 --log_step=10 --ckpt_step=500  --batch_size=64 \
--device_id=0 --class_ids=02958343 --img_size=64 --lambda_smth=10 --latent_dim=512 \
--n_epochs=20 --decay_epoch=2 --decay_every=1 --decay_order=0.8 --conditioned=0 --channels=1

## training of img-GAN
#CUDA_VISIBLE_DEVICES=1 python imgGAN.py --model=DCGAN --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--log_dir=/hd2/pengbo/mesh_reconstruction/models/log_temp --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample_temp \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt_temp --ckpt_step=1000 \
#--G_every=1 --sample_step=500 --n_epochs=50 --decay_epoch=15 --decay_every=1 --decay_order=0.9 --device_id=0 \
#--class_ids=02958343 --channels=25 --img_size=64

### training of 3D-AE
#CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=trainAE --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--log_dir=/hd2/pengbo/mesh_reconstruction/models/log3D_AE --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_AE \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_AE \
#--obj_dir=/home/pengbo/allProjects/mesh_reconstruction/data/obj/sphere_642.obj \
#--sample_step=100 --ckpt_step=500 --n_epochs=20 --decay_epoch=2 --decay_every=1 --decay_order=0.8 --device_id=0 \
#--class_ids=02958343 --img_size=64 --lambda_smth=0.01 --latent_dim=512

## training of D
#python 3D-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/DCGAN' \
#--log_dir='/hd1/xuanxinsheng/result/celeba-dcgan/log/' --ckpt_dir='/hd1/xuanxinsheng/result/celeba-dcgan/model/' \
#--n_epochs=1 --device_id=0

## training of D on JPEG VS png
#python 3D-GAN.py --mode='trainD' --load_G='/hd2/pengbo/StarGAN/DCGAN_celebA/model/47490-G.ckpt' \
#--dir_generated='/hd2/pengbo/CelebA/img_align_celeba_png/' --N_generated=200000 \
#--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_D' --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model' \
#--n_epochs=10 --device_id=1 --log_step=1

## testing of D
#python 3D-GAN.py --mode=testD --dir_generated=/hd1/xuanxinsheng/data/DCGAN --log_step=100 \
#--load_D=/hd1/xuanxinsheng/result/celeba-wgangp/model/Forensics-D.ckpt --device_id=1

## testing of D on JPEG VS png
#python 3D-GAN.py --mode=testD --dir_generated=/hd2/pengbo/CelebA/img_align_celeba_png --log_step=100 \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model/Forensics-D.ckpt --device_id=1

## training of G
#python 3D-GAN.py --mode=trainG --load_G=/hd2/pengbo/StarGAN/DCGAN_celebA/model/47490-G.ckpt \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model/Forensics-D.ckpt \
#--log_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/log_G --ckpt_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/model \
#--sample_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/sample2 --log_step=1 --sample_step=1

#----------------------------------------------------------------------------------------------------

### training of D celebA VS DCGAN
#python 3D-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/DCGAN' \
#--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_trainD_DCGAN' --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model_trainD_DCGAN' \
#--n_epochs=1 --device_id=0

## training of D celebA VS WGANGP
#python 3D-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/WGANGP' \
#--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_trainD_WGAN' --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model_trainD_WGAN' \
#--n_epochs=1 --device_id=1

## testing of D celebA VS DCGAN
#python 3D-GAN.py --mode=testD --dir_generated=/hd1/xuanxinsheng/data/DCGAN --log_step=100 \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model_trainD_DCGAN/Forensics-D.ckpt --device_id=0

### testing of D celebA VS WGANGP
#python 3D-GAN.py --mode=testD --dir_generated=/hd1/xuanxinsheng/data/WGANGP --log_step=100 \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model_trainD_WGAN/Forensics-D.ckpt --device_id=0

# training of D for xuan
#python 3D-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/DCGAN' \
#--log_dir='/hd1/xuanxinsheng/result/xxs/log/' --ckpt_dir='/hd1/xuanxinsheng/result/xxs/model/' \
#--n_epochs=1 --device_id=0

## testing of D for xuan
#python 3D-GAN.py --mode=testD --dir_generated=/hd1/xuanxinsheng/data/DCGAN --log_step=100 \
#--load_D=/hd1/xuanxinsheng/result/celeba-wgangp/model/Forensics-D.ckpt --device_id=0
