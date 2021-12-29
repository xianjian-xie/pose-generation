export CUDA_VISIBLE_DEVICES=0,1;
python train.py --dataroot ./datasets/fashion_data/ --name fashion_selectiongan --model PATN --lambda_GAN 5 --lambda_A 10 --lambda_B 10 --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 12 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --niter 500 --niter_decay 200 --checkpoints_dir ./checkpoints --pairLst ./datasets/fashion_data/fasion-resize-pairs-train.csv --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 0 --gpu_ids 0,1;--
# --which_epoch 60 --epoch_count 61
