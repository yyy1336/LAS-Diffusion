# sketch-conditioned generation
# please give a rough view information of the object in the sketch

## creative design
python generate.py --model_path /home/yyy/Projects/LAS-Diffusion/checkpoints/mine_airplane_sketch/epoch\=1999.ckpt --generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path /home/yyy/Projects/dataset_shapenet/sketch_test/edge/02691156/fd7c74a05072d3befef192e05c55dcd3/edge_4_4.png --view_information 4
python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path /home/yyy/Projects/LAS-Diffusion/outputs/mine_airplane_sketch_epoch=1999.ckpt_True_fd7c74a05072d3befef192e05c55dcd3_edge_4_4_1.0_4/ --save_npy True --save_mesh True --level 0.0 --steps 20

#python generate.py --model_path /home/yyy/Projects/LAS-Diffusion/checkpoints/mine_airplane_sketch/epoch\=1999.ckpt --generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path ./demo_data/01.png --view_information 1
#python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_sketch_epoch=299.ckpt_True_demo_data_01_1.0_1 --save_npy True --save_mesh True --level 0.0 --steps 20
#
### add bars
#python generate.py --model_path /home/yyy/Projects/LAS-Diffusion/checkpoints/mine_airplane_sketch/epoch\=1999.ckpt--generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path ./demo_data/02.png --view_information 1
#python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_sketch_epoch=299.ckpt_True_demo_data_02_1.0_1 --save_npy True --save_mesh True --level 0.0 --steps 20
#
#
#python generate.py --model_path /home/yyy/Projects/LAS-Diffusion/checkpoints/mine_airplane_sketch/epoch\=1999.ckpt--generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path ./demo_data/03.png --view_information 1
#python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_sketch_epoch=299.ckpt_True_demo_data_03_1.0_1 --save_npy True --save_mesh True --level 0.0 --steps 20
