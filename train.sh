#!/usr/bin/env bash

## training of GAN
#python DC-GAN.py --ckpt_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/model2' --log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log2' \
#--sample_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/sample2'

# training of D
python DC-GAN.py --mode='trainD' --load_G='/hd2/pengbo/StarGAN/DCGAN_celebA/model/94980-G.ckpt' \
--dir_generated='/hd2/pengbo/StarGAN/DCGAN_celebA/generated' --N_generated=200000 \
--log_dir='/hd2/pengbo/StarGAN/DCGAN_celebA/log_D' --log_step=10