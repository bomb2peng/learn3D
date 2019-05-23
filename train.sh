#!/usr/bin/env bash
# training of 3D-GAN
# 02958343--car, 02691156--airplane, 02828884--bench, 02933112--dresser, 03001627--chair, 03211117--display,
# 03636649--lamp, 03691459--loudspeaker, 04090263--riffle, 04256520--sofa, 04379243--table, 04401088--telephone,
# 04530566--vessel
#CUDA_VISIBLE_DEVICES=2 python 3D-GAN.py --mode=trainGAN --model=WGAN-GP --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--log_dir=/hd2/pengbo/mesh_reconstruction/models/log3D_GAN --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_GAN \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_GAN  \
#--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_42.obj \
#--G_every=5 --sample_step=100 --log_step=10 --ckpt_step=100  --batch_size=128 --device_id=0 --class_ids=02958343 \
#--img_size=64 --lambda_smth=1 --lambda_Lap=1 --lambda_edge=1 --lambda_feat=0 --latent_dim=512 \
#--n_epochs=20 --decay_epoch=2 --decay_every=1 --decay_order=0.8 --conditioned=0 --channels=1 \
#--iter_divide1=500 --iter_divide2=2000
##--load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_GAN/300-G.ckpt \
##--load_D=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_GAN/300-D.ckpt --batches_done=300

## batch runs
#ids="02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04090263 04256520 04379243 04401088 04530566"
#for j in $ids
#do
##    CUDA_VISIBLE_DEVICES=2 python 3D-GAN.py --mode=trainAE --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
##    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_${j}_AE521 \
##    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_${j}_AE521  \
##    --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
##    --sample_step=500 --log_step=10 --ckpt_step=1  --batch_size=128 --device_id=0 --class_ids=${j} \
##    --img_size=64 --lambda_smth=1e-3 --lambda_KLD=1e-5 --latent_dim=512 \
##    --n_epochs=30 --decay_epoch=20 --decay_every=10 --decay_order=0.1 --visdom_env=log3D_${j}-AE521 #--use_VAE
#
##    CUDA_VISIBLE_DEVICES=2 python 3D-GAN.py --mode=evaluation --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
##    --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
##    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_${j}_VAE521 \
##    --device_id=0 --class_ids=${j} --img_size=64 --latent_dim=512 --use_VAE
#
#    # training of 3D-AE-featGAN
#    CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=trainAE_featGAN --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_${j}_AEfeatGAN \
#    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_${j}_AEfeatGAN \
#    --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#    --sample_step=500 --n_epochs=30 --decay_epoch=20 --decay_every=10 --decay_order=0.1 --device_id=0 \
#    --class_ids=${j} --img_size=64 --lambda_smth=0.001 --latent_dim=512 --visdom_env=log3D_${j}_AEfeatGAN \
#    --gp=0.1 --G_every=5 #--use_VAE
#
##    ## reconstruction results on an image
##    for i in {0..9}
##    do
##        CUDA_VISIBLE_DEVICES=0 python 3D-GAN.py --mode=reconstruct \
##        --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
##        --sample_dir=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples_train_VAE521 \
##        --device_id=0 --img_size=64 --latent_dim=512 --use_VAE \
##        --load_im=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples_train/${j}_${i}.png \
##        --load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_${j}_VAE521/best-G.ckpt \
##        --load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_${j}_VAE521/best-E.ckpt
##    done
#done

### sampling from trained model
#CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=sampleGAN \
#--sample_dir=/hd2/pengbo/mesh_reconstruction/models/samples \
#--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_42.obj \
#--load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_GAN/10000-G.ckpt \
#--batch_size=32 --device_id=0 --latent_dim=512 --sample_prefix=car

## training of img-GAN
#CUDA_VISIBLE_DEVICES=1 python imgGAN.py --model=DCGAN --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--log_dir=/hd2/pengbo/mesh_reconstruction/models/log_temp --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample_temp \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt_temp --ckpt_step=1000 \
#--G_every=1 --sample_step=500 --n_epochs=50 --decay_epoch=15 --decay_every=1 --decay_order=0.9 --device_id=0 \
#--class_ids=02958343 --channels=25 --img_size=64

### training of 3D-AE
#CUDA_VISIBLE_DEVICES=2 python 3D-GAN.py --mode=trainAE --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_02691156_AE1 \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_02691156_AE1 \
#--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#--sample_step=500 --ckpt_step=1000 --n_epochs=50 --decay_epoch=40 --decay_every=10 --decay_order=0.1 --device_id=0 \
#--class_ids=02691156 --img_size=64 --lambda_smth=0.001 --latent_dim=512 --visdom_env=log3D_02691156-AE1 --use_VAE
##--load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_AE/last-G.ckpt \
##--load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_AE/last-E.ckpt --batches_done=52000 --lr=0.00001

## training of 3D-AE-GAN
#CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=trainAE_GAN --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_02691156_AE-GAN --model=WGAN-GP \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_02691156_AE-GAN \
#--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#--sample_step=500 --ckpt_step=1000 --n_epochs=50 --decay_epoch=40 --decay_every=10 --decay_order=0.1 --device_id=0 \
#--class_ids=02691156 --img_size=64 --lambda_smth=0.001 --latent_dim=512 --visdom_env=log3D_02691156_AE-GAN --channels=25 \
#--load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_02691156_AE1/last-G.ckpt \
#--load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_02691156_AE1/last-E.ckpt --use_VAE --G_every=1 --log_step=1

# training of 3D-AE-featGAN
CUDA_VISIBLE_DEVICES=2 python 3D-GAN.py --mode=trainAE_featGAN --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
--sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_temp \
--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp \
--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
--sample_step=500 --n_epochs=30 --decay_epoch=20 --decay_every=10 --decay_order=0.1 --device_id=0 \
--class_ids=02958343 --img_size=64 --lambda_smth=0.001 --latent_dim=512 --visdom_env=log3D_temp \
--gp=10 --G_every=5 #--use_VAE


### batch evaluation of 3D-AE
#CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=evaluation --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_03691459_VAE \
#--device_id=0 --class_ids=03691459 --img_size=64 --latent_dim=512 --use_VAE

### t-SNE visualization
#CUDA_VISIBLE_DEVICES=2 python 3D-GAN.py --mode=t_SNE --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--n_epochs=1 --device_id=0 \
#--class_ids=02691156 --img_size=64 --latent_dim=512 --use_VAE \
#--load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_02691156_AE1/last-E.ckpt

### reconstruction results on an image
#CUDA_VISIBLE_DEVICES=0 python 3D-GAN.py --mode=reconstruct --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#--obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#--sample_dir=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples_VAE521 \
#--device_id=0 --img_size=64 --latent_dim=512 --use_VAE \
#--load_im=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples/02691156_1.png \
#--load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_02691156_VAE521/best-G.ckpt \
#--load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_02691156_VAE521/best-E.ckpt