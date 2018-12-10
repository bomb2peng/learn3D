#!/usr/bin/env bash

## training of DCGAN
#python DC-GAN.py --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model4' --log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log4' \
#--sample_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/sample4' --device_id=1

# training of WGAN
python DC-GAN.py --model=WGAN --ckpt_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN \
--log_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/log_WGAN --sample_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/sample_WGAN \
--G_every=5 --sample_step=500 --lr=0.0001 --n_epochs=20 \
--load_G=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN/121000-G.ckpt \
--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model_WGAN/121000-D.ckpt --batches_done=121000

## training of D
#python DC-GAN.py --mode='trainD' --load_G='/hd2/pengbo/StarGAN/DCGAN_celebA/model2/47490-G.ckpt' \
#--dir_generated='/hd2/pengbo/StarGAN/DCGAN_celebA/generated' --N_generated=200000 \
#--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_D' --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model2' \
#--n_epochs=1

## testing of D
#python DC-GAN.py --mode=testD --dir_generated=/hd2/pengbo/StarGAN/DCGAN_celebA/generated --log_step=100 \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model2/12000-D.ckpt --device_id=0

## training of G
#python DC-GAN.py --mode=trainG --load_G=/hd2/pengbo/StarGAN/DCGAN_celebA/model/47490-G.ckpt \
#--load_D=/hd2/pengbo/StarGAN/DCGAN_celebA/model/Forensics-D.ckpt \
#--log_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/log_G --ckpt_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/model \
#--sample_dir=/hd2/pengbo/StarGAN/DCGAN_celebA/sample2 --log_step=1 --sample_step=1