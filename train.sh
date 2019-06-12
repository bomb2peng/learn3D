#!/usr/bin/env bash
# 02691156--airplane, 02828884--bench, 02933112--dresser, 02958343--car, 03001627--chair, 03211117--display,
# 03636649--lamp, 03691459--loudspeaker, 04090263--riffle, 04256520--sofa, 04379243--table, 04401088--telephone,
# 04530566--vessel

# batch runs
#ids="02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04090263 04256520 04379243 04401088 04530566"
ids="02691156"
for j in ${ids}
do
##    training of 3D-AE
#    CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=trainAE --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_temp \
#    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp  \
#    --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#    --sample_step=500 --log_step=10 --ckpt_step=1  --batch_size=128 --device_id=0 --class_ids=${j} \
#    --img_size=64 --lambda_smth=1e-3 --latent_dim=512 \
#    --n_epochs=1 --decay_epoch=15 --decay_every=5 --decay_order=0.1 --visdom_env=main \
#    --AE_Gprior
##    --load_G=/hd2/pengbo/mesh_reconstruction/models/AE529/ckpt3D_${j}_AE529/last-G.ckpt \
##    --load_E=/hd2/pengbo/mesh_reconstruction/models/AE529/ckpt3D_${j}_AE529/last-E.ckpt

## Evaluations
#    CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=evaluation --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#    --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp \
#    --device_id=0 --class_ids=${j} --img_size=64 --latent_dim=512 --eval_flag=last

#    # training of 3D-AE-featGAN
#    CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=trainAE_featGAN --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#    --sample_dir=/hd2/pengbo/mesh_reconstruction/models/sample3D_temp \
#    --ckpt_dir=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp \
#    --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#    --sample_step=500 --n_epochs=20 --decay_epoch=15 --decay_every=10 --decay_order=0.1 --device_id=0 \
#    --class_ids=${j} --img_size=64 --lambda_smth=0.001 --latent_dim=512 --visdom_env=main \
#    --G_every=1 --lambda_Gprior=1
##    --load_G=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN604/ckpt3D_${j}_AEfeatGAN604/last-G.ckpt \
##    --load_E=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN604/ckpt3D_${j}_AEfeatGAN604/last-E.ckpt \
##    --load_D=/hd2/pengbo/mesh_reconstruction/models/AEfeatGAN604/ckpt3D_${j}_AEfeatGAN604/last-D.ckpt


#    ## reconstruction results on an image
#    for i in {0..9}
#    do
#        CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=reconstruct \
#        --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#        --sample_dir=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples_temp \
#        --device_id=0 --img_size=64 --latent_dim=512 \
#        --load_im=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples/${j}_${i}.png \
#        --load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-G.ckpt \
#        --load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-E.ckpt
#    done

#    ## interplolation results on two images
#    for i in {1..5}
#    do
#        CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=interpolation \
#        --obj_dir=/hd2/pengbo/mesh_reconstruction/models/template_obj/sphere_642.obj \
#        --sample_dir=/hd2/pengbo/mesh_reconstruction/models/reconstruction/interpo_temp \
#        --device_id=0 --img_size=64 --latent_dim=512 \
#        --load_im=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples/${j}_0.png \
#        --load_im1=/hd2/pengbo/mesh_reconstruction/models/reconstruction/examples/${j}_${i}.png \
#        --load_G=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-G.ckpt \
#        --load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-E.ckpt
#    done

    ## t-SNE visualization
    CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=t_SNE --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
    --n_epochs=1 --device_id=0 \
    --class_ids=${j} --img_size=64 --latent_dim=512 \
    --load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-E.ckpt

#    ## MMD between features of different poses
#    CUDA_VISIBLE_DEVICES=3 python 3D-GAN.py --mode=MMD --data_dir=/hd2/pengbo/mesh_reconstruction/dataset/ \
#    --n_epochs=1 --device_id=0 \
#    --class_ids=${j} --img_size=64 --latent_dim=512 \
#    --load_E=/hd2/pengbo/mesh_reconstruction/models/ckpt3D_temp/last-E.ckpt
done