# please modify the sdf_folder and sketch_folder to your own path

# uncondition，  airplane
#CUDA_VISIBLE_DEVICES=9 
CUDA_VISIBLE_DEVICES=0 python train.py --data_class airplane --name mine_airplane_U --batch_size 16 --new True --continue_training False --image_size 64 --training_epoch 4000 --ema_rate 0.999 --base_channels 32  --save_last False --save_every_epoch 200 --with_attention False --use_text_condition False --use_sketch_condition False --split_dataset True  --lr 1e-4 --optimizier adamw --sdf_folder /home/yyy/Projects/dataset_shapenet/sdf --verbose False





