#!/usr/bin/env bash
# 02691156--airplane, 02828884--bench, 02933112--dresser, 02958343--car, 03001627--chair, 03211117--display,
# 03636649--lamp, 03691459--loudspeaker, 04090263--riffle, 04256520--sofa, 04379243--table, 04401088--telephone,
# 04530566--vessel

## batch runs
#ids="02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04090263 04256520 04379243 04401088 04530566"
#ids="02691156 02958343 03001627"
ids="02691156"

for j in ${ids}
do
    # training of 3D-AE-featGAN on CVPR18 dataset
    CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=train \
    --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ --trainViews=24 \
    --dataset=CVPR18 \
    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN/sample3D_${j} \
    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN/ckpt3D_${j} \
    --obj_dir=sphere_642.obj \
    --sample_step=500 --ckpt_step=500 --n_iters=20000 --decay_batch=40000 --decay_every=5000 --decay_order=0.1 --device_id=0 \
    --class_ids=${j} --img_size=64 --lambda_smth=0.001 --lambda_Gprior=1. --lambda_adv=1 --latent_dim=512 \
    --visdom_env=log3D-AEfeatGAN_${j} --G_every=2
#    --lambda_smth=0.001
#
    # Evaluations each model
    CUDA_VISIBLE_DEVICES=0 python 3D-GAN.py --mode=evaluation \
    --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ --dataset=CVPR18 --class_id=${j} \
    --obj_dir=sphere_642.obj \
    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN \
    --device_id=0 --img_size=64 --latent_dim=512 --eval_flag=last

#    # training of 3D-AE-featGAN on NIPS17 dataset
#    CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=train \
#    --data_dir=/hd3/pengbo/shapenet_LSM/lsm/data/shapenet_release/ --trainViews=1 \
#    --dataset=NIPS17 --split_file=/hd3/pengbo/shapenet_LSM/lsm/data/splits.json \
#    --wAzimuth=15 --wElevation=20 \
#    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Kar/sample3D_${j} \
#    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Kar/ckpt3D_${j} \
#    --obj_dir=sphere_642.obj \
#    --sample_step=500 --ckpt_step=500 --n_iters=20000 --decay_batch=40000 --decay_every=5000 --decay_order=0.1 --device_id=0 \
#    --class_ids=${j} --img_size=64 --lambda_smth=0.001 --lambda_Gprior=1. --lambda_adv=1. --latent_dim=512 \
#    --visdom_env=log3D-AEfeatGAN-Kar_${j} --G_every=2
##    --batches_done=20000 \
##    --load_G=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN616/ckpt3D_${j}_AEfeatGAN616/last-G.ckpt \
##    --load_E=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN616/ckpt3D_${j}_AEfeatGAN616/last-E.ckpt \
##    --load_D=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN616/ckpt3D_${j}_AEfeatGAN616/last-D.ckpt

#    # training of 3D-AE-featGAN on Pascal3D+ dataset
#    CUDA_VISIBLE_DEVICES=0 python 3D-GAN.py --mode=train \
#    --data_dir=/hd1/pengbo/Pascal3D_Kato/ --trainViews=1 \
#    --dataset=Pascal3D \
#    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D/sample3D_${j} \
#    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D/ckpt3D_${j} \
#    --obj_dir=sphere_642.obj \
#    --sample_step=500 --ckpt_step=500 --n_iters=40000 --decay_batch=60000 --decay_every=5000 --decay_order=0.1 --device_id=0 \
#    --class_ids=${j} --img_size=224 --lambda_smth=0.001 --lambda_Gprior=0.01 --lambda_adv=0.01 --latent_dim=512 \
#    --visdom_env=log3D-AEfeatGAN-Pascal3D_${j} --G_every=2
##    --batches_done=8000 \
##    --load_G=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D-try2/ckpt3D_${j}/last-G.ckpt \
##    --load_E=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D-try2/ckpt3D_${j}/last-E.ckpt \
##    --load_D=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D-try2/ckpt3D_${j}/last-D.ckpt


#    ## t-SNE visualization
#    CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=t_SNE --data_dir=/hd1/pengbo/Pascal3D_Kato/ \
#    --dataset=Pascal3D --device_id=0 \
#    --class_ids=${j} --img_size=224 --latent_dim=512 \
#    --load_E=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D-baseline/ckpt3D_${j}/best-E.ckpt

## training of CVPR19 model: AE+GANonImageDomain
#    CUDA_VISIBLE_DEVICES=0 python 3D-GAN.py --mode=trainCVPR19 --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/VPL1111/sample3D_${j} \
#    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/VPL1111/ckpt3D_${j} \
#    --obj_dir=sphere_642.obj \
#    --sample_step=500 --ckpt_step=500 --n_iters=20000 --decay_batch=40000 --decay_every=5000 --decay_order=0.1 --device_id=0 \
#    --class_ids=${j} --img_size=64 --lambda_smth=0.001 --lambda_adv=1. --latent_dim=512 \
#    --visdom_env=log3D-VPL1111_${j} --G_every=2 --channels=25

#
#    ## MMD between features of different poses
#    CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=MMD --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#    --device_id=0 --class_ids=${j} --img_size=64 --latent_dim=512 \
#    --load_E=/hd2/pengbo/mesh_reconstruction/models/AE616/ckpt3D_${j}_AE616/last-E.ckpt

#    ## reconstruction results on an image
#    for i in {0..9}
#    do
#        CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=reconstruct \
#        --obj_dir=sphere_642.obj \
#        --sample_dir=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples_temp \
#        --device_id=0 --img_size=64 --latent_dim=512 \
#        --load_im=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples/${j}_${i}.png \
#        --load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-G.ckpt \
#        --load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-E.ckpt
#    done

#    ## reconstruction results for Pascal 3D+ image
#    CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=reconstruct_Pascal \
#    --obj_dir=sphere_642.obj \
#    --data_dir=/hd1/pengbo/Pascal3D_Kato/ --dataset=Pascal3D --class_ids=${j} \
#    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples_temp \
#    --device_id=0 --img_size=224 --latent_dim=512 \
#    --load_G=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D-baseline-try2/ckpt3D_${j}/best-G.ckpt \
#    --load_E=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN-Pascal3D-baseline-try2/ckpt3D_${j}/best-E.ckpt

#    ## interplolation results on two images
#    for i in {1..5}
#    do
#        CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=interpolation \
#        --obj_dir=sphere_642.obj \
#        --sample_dir=/hd2/pengbo/mesh_reconstruction/models/reconstruction/interpo_temp \
#        --device_id=0 --img_size=64 --latent_dim=512 \
#        --load_im=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples/${j}_0.png \
#        --load_im1=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples/${j}_${i}.png \
#        --load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-G.ckpt \
#        --load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-E.ckpt
#    done

done

## Evaluations ALL
#CUDA_VISIBLE_DEVICES=1 python 3D-GAN.py --mode=evaluation \
#--data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ --dataset=CVPR18 \
#--obj_dir=sphere_642.obj \
#--ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/noSmooth/ \
#--device_id=0 --img_size=64 --latent_dim=512 --eval_flag=last

