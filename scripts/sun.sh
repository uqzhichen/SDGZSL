cd ..

# SUN w/o finetuning GZSL: H 41.27
python train.py --dataset SUN --ga 30 --beta 0.3 --dis 0.5 --nSample 400 --gpu 1 --S_dim 2048 --NS_dim 2048 --lr 0.0003 \
  --kl_warmup 0.001 --tc_warmup 0.0003 --vae_dec_drop 0.2 --dis_step 3 --ae_drop 0.4 # 

# SUN w/o finetuning ZSL: T1 62.43
python train.py --dataset SUN --ga 15 --beta 0.1 --dis 3.0 --nSample 400 --gpu 1 --S_dim 2048 --NS_dim 2048 --lr 0.0003 \
  --zsl True --classifier_lr 0.005 # 

# SUN w/ finetuning GZSL: H 45
python train.py --dataset SUN --ga 3 --beta 0.1 --dis 0.5 --nSample 400 --gpu 0 --S_dim 2048 --NS_dim 2048 --lr 0.0003 \
 --finetune True # 

# SUN w/ finetuning ZSL: T1 65.21
python train.py --dataset SUN --ga 3 --beta 0.1 --dis 0.03 --nSample 400 --gpu 0 --S_dim 2048 --NS_dim 2048 --lr 0.0003 \
 --finetune True --zsl True --classifier_lr 0.0003 #  

