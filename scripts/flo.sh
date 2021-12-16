cd ..

# FLO w/o finetuning GZSL: H 69.75
python train.py --dataset FLO  --ga 3 --beta 0.1 --dis 0.1 --nSample 1200 --gpu 0 --S_dim 2048 --NS_dim 2048 \
  --lr 0.0001  --classifier_lr 0.003 --dis_step 3 --kl_warmup 0.001 --vae_dec_drop 0.4 --vae_enc_drop 0.4 \
  --ae_drop 0.2 #

# FLO w/o finetuning ZSL: T1 71.33
python train.py --dataset FLO  --ga 1 --beta 0.1 --dis 0.1 --nSample 1200 --gpu 0 --S_dim 2048 --NS_dim 2048 \
  --lr 0.0001  --classifier_lr 0.003 --dis_step 3 --zsl true --kl_warmup 0.001 --vae_dec_drop 0.4 \
  --vae_enc_drop 0.4 --ae_drop 0.4 #

# FLO w/ finetuning GZSL: H 80.21
python train.py --dataset FLO  --ga 10 --beta 1.0 --dis 3.0 --nSample 1200 --gpu 0 --S_dim 1024 --NS_dim 1024 \
  --lr 0.0001  --classifier_lr 0.003 --finetune True --dis_step 1 #

# FLO w/ finetuning ZSL: T1 76.6
python train.py --dataset FLO  --ga 10 --beta 1.0 --dis 3.0 --nSample 1200 --gpu 0 --S_dim 1024 --NS_dim 1024 \
  --lr 0.0001  --classifier_lr 0.003 --finetune True --dis_step 1 --zsl true #


