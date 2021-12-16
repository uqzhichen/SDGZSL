cd ..

## CUB w/o finetuning GZSL: H 63.00
python train.py --dataset CUB_STC  --ga 3 --beta 0.3 --dis 0.3 --nSample 1200 --gpu 0 --S_dim 312 --NS_dim 312 \
  --lr 0.0001  --classifier_lr 0.005 --gen_nepoch 400 --kl_warmup 0.02 --tc_warmup 0.001 --weight_decay 1e-6 \
  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --dis_step 2 --ae_drop 0.2 # 63.58

## CUB w/o finetuning ZSL: T1 75.5
python train.py --dataset CUB_STC  --ga 3 --beta 0.3 --dis 0.3 --nSample 1200 --gpu 0 --S_dim 312 --NS_dim 312 \
  --lr 0.0001  --classifier_lr 0.005 --gen_nepoch 400 --kl_warmup 0.02 --tc_warmup 0.001 --weight_decay 1e-6 \
  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --dis_step 2 --ae_drop 0.2 --zsl true #

# CUB w/ finetuning GZSL: H 75.1
python train.py --dataset CUB_STC  --ga 3 --beta 0.3 --dis 0.3 --nSample 1200 --gpu 0 --S_dim 1024 --NS_dim 1024 \
  --lr 0.0001  --classifier_lr 0.005 --gen_nepoch 400 --kl_warmup 0.02 --tc_warmup 0.001 --weight_decay 1e-6 \
  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --dis_step 2 --ae_drop 0.2 --finetune true #

# CUB w/ finetuning ZSL: T1 78.5
python train.py --dataset CUB_STC  --ga 3 --beta 0.3 --dis 0.3 --nSample 1200 --gpu 0 --S_dim 1024 --NS_dim 1024 \
  --lr 0.0001  --classifier_lr 0.005 --gen_nepoch 400 --kl_warmup 0.02 --tc_warmup 0.001 --weight_decay 1e-6 \
  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --dis_step 2 --ae_drop 0.2 --finetune true --zsl true #

