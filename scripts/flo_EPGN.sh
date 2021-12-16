cd ..

# FLO w/o finetuning GZSL: H 86.6
python train.py --dataset FLO_EPGN --ga 15 --beta 1 --dis 3 --nSample 1200 --gpu 0 --S_dim 1024 --NS_dim 1024 \
  --lr 0.0001  --classifier_lr 0.005 --dis_step 2 --kl_warmup 0.01 --tc_warmup 0.001 --gen_nepoch 400 \
  --vae_dec_drop 0.4 --vae_enc_drop 0.4 --ae_drop 0.2  #

# FLO w/o finetuning ZSL: T1 85.4
python train.py --dataset FLO_EPGN --ga 15 --beta 1 --dis 3 --nSample 1200 --gpu 0 --S_dim 1024 --NS_dim 1024 \
  --lr 0.0001  --classifier_lr 0.005 --dis_step 2 --kl_warmup 0.01 --tc_warmup 0.001 --gen_nepoch 400 \
  --vae_dec_drop 0.4 --vae_enc_drop 0.4 --ae_drop 0.2 --zsl true #

# FLO w/ finetuning GZSL: H 87.8
python train.py --dataset FLO_EPGN  --ga 3 --beta 0.1 --dis 0.1 --nSample 1200 --gpu 0 --S_dim 2048 --NS_dim 2048 \
  --lr 0.0001  --classifier_lr 0.003 --dis_step 3 --kl_warmup 0.001 --vae_dec_drop 0.4 --vae_enc_drop 0.4 \
  --ae_drop 0.2 --finetune true #

# FLO w/ finetuning ZSL: T1 86.9
python train_SUN.py --dataset FLO_EPGN  --ga 3 --beta 0.1 --dis 0.1 --nSample 1200 --gpu 0 --S_dim 2048 --NS_dim 2048 \
  --lr 0.0001  --classifier_lr 0.003 --dis_step 3 --kl_warmup 0.001 --vae_dec_drop 0.4 --vae_enc_drop 0.4 \
  --ae_drop 0.2 --finetune true --zsl true #
