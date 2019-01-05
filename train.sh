#!/usr/bin/env bash

# training of DCGAN
#python DC-GAN.py --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model_DCGAN2' \
#--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_DCGAN2' \
#--sample_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/sample_DCGAN2' --device_id=0 --model=DCGAN --n_epochs=40

## training of WGAN
#python DC-GAN.py --model=WGAN --ckpt_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN \
#--log_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/log_WGAN --sample_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/sample_WGAN \
#--G_every=5 --sample_step=500 --lr=0.0001 --n_epochs=20 \
#--load_G=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN/121000-G.ckpt \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN/121000-D.ckpt --batches_done=121000

## training of WGAN-GP
#python DC-GAN.py --model=WGAN-GP #--ckpt_dir=/hd3/xuanxinsheng/result/WGAN_celebA/v2/model/ \
#--log_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/log_WGAN-GP_I --sample_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/sample_WGAN-GP_I \
#--G_every=5 --sample_step=500 --n_epochs=50 --decay_epoch=50

## training of D
#python DC-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/DCGAN' \
#--log_dir='/hd1/xuanxinsheng/result/celeba-dcgan/log/' --ckpt_dir='/hd1/xuanxinsheng/result/celeba-dcgan/model/' \
#--n_epochs=1 --device_id=0

## training of D on JPEG VS png
#python DC-GAN.py --mode='trainD' --load_G='/hd2/pengbo/StarGAN/DCGAN_celebA/model/47490-G.ckpt' \
#--dir_generated='/hd2/pengbo/CelebA/img_align_celeba_png/' --N_generated=200000 \
#--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_D' --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model' \
#--n_epochs=10 --device_id=1 --log_step=1

## testing of D
#python DC-GAN.py --mode=testD --dir_generated=/hd1/xuanxinsheng/data/DCGAN --log_step=100 \
#--load_D=/hd1/xuanxinsheng/result/celeba-wgangp/model/Forensics-D.ckpt --device_id=1

## testing of D on JPEG VS png
#python DC-GAN.py --mode=testD --dir_generated=/hd2/pengbo/CelebA/img_align_celeba_png --log_step=100 \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model/Forensics-D.ckpt --device_id=1

## training of G
#python DC-GAN.py --mode=trainG --load_G=/hd2/pengbo/StarGAN/DCGAN_celebA/model/47490-G.ckpt \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model/Forensics-D.ckpt \
#--log_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/log_G --ckpt_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/model \
#--sample_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/sample2 --log_step=1 --sample_step=1

#----------------------------------------------------------------------------------------------------

## training of D celebA VS DCGAN blur
#python DC-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/DCGAN' \
#--log_dir='/hd1/xuanxinsheng/result/celeba-dcgan/log/' --ckpt_dir='/hd1/xuanxinsheng/result/celeba-dcgan/model/' \
#--n_epochs=1 --device_id=0

# training of D celebA VS WGANGP blur
#python DC-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/WGANGP' \
#--log_dir='/hd1/xuanxinsheng/result/celeba-wgangp/log/' --ckpt_dir='/hd1/xuanxinsheng/result/celeba-wgangp/model/' \
#--n_epochs=1 --device_id=1

## testing of D celebA VS DCGAN
python DC-GAN.py --mode=testD --dir_generated=/hd1/xuanxinsheng/data/DCGAN --log_step=100 \
--load_D=/hd1/xuanxinsheng/result/celeba-dcgan/model/Forensics-D.ckpt --device_id=0

## testing of D celebA VS WGANGP
#python DC-GAN.py --mode=testD --dir_generated=/hd1/xuanxinsheng/data/WGANGP --log_step=100 \
#--load_D=/hd1/xuanxinsheng/result/celeba-wgangp/model/Forensics-D.ckpt --device_id=1

## training of D for xuan
#python DC-GAN.py --mode='trainD' \
#--load_G='/hd1/xuanxinsheng/result/model/wgan-gp/Epoch_(50).ckpt' \
#--dir_generated='/hd1/xuanxinsheng/data/DCGAN' \
#--log_dir='/hd1/xuanxinsheng/result/xxs/log/' --ckpt_dir='/hd1/xuanxinsheng/result/xxs/model/' \
#--n_epochs=1 --device_id=0