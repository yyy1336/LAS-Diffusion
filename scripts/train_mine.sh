# please modify the sdf_folder and sketch_folder to your own path

# uncondition
#CUDA_VISIBLE_DEVICES=9 
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python train.py --data_class class_2 --name mine_datasetv02_all --batch_size 32 --new True --continue_training True --image_size 64 --training_epoch 8000 --ema_rate 0.999 --base_channels 32  --save_last True --save_every_epoch 200 --with_attention True --use_text_condition False --use_sketch_condition False --split_dataset False  --lr 5e-5 --optimizier adamw --sdf_folder /home/yyy/Projects/dataset/microstructure_v02/data_set_0.2 --verbose False --data_form 1


#CUDA_VISIBLE_DEVICES=9 python train.py --data_class bulk_02 --name mine_bulk02_try --batch_size 16 --new True --continue_training True --image_size 64 --training_epoch 4000 --ema_rate 0.999 --base_channels 32  --save_last False --save_every_epoch 200 --with_attention False --use_text_condition False --use_sketch_condition False --split_dataset False  --lr 1e-4 --optimizier adamw --sdf_folder /home/yyy/Projects/dataset/microstructure/sdf --verbose False --data_form 0
#TODO: stiffness tensor conditioned
## occpuancy-diffuion module
#python train.py --data_class bulk_02 --name try_bulk_02_C --batch_size 16 --new True --continue_training False --image_size 64 --training_epoch 300 --ema_rate 0.995 --base_channels 32 --noise_schedule linear --save_last False --save_every_epoch 50 --with_attention True --use_text_condition False --use_sketch_condition True --kernel_size 4.0  --view_information_ratio 2.0  --lr 2e-4 --optimizier adam --data_augmentation True --sdf_folder /home/yyy/Projects/dataset/microstructure/sdf --sketch_folder /home/D/dataset/shapenet_edge_our_new
#
# SDF-diffuion module
#CUDA_VISIBLE_DEVICES=6,7,8,9 python train_super_resolution.py --data_class bulk_02 --name super_mine_bulk_02_U_v2 --batch_size 4 --new True --continue_training False --training_epoch 500  --split_dataset True --sdf_folder /home/yyy/Projects/dataset/microstructure/sdf


