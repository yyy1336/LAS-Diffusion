# unconditional generation
# replace airplane with chair, car, rifle, table etc. to generate other categories

#python generate.py --model_path /home/yyy/Projects/LAS-Diffusion/results/mine_uncond_occ64/epoch\=199.ckpt --generate_method generate_unconditional --num_generate 16 --steps 50
#python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/airplane_epoch\=2799_199.ckpt_True_50_0.0_unconditional --save_npy True --save_mesh True --level 0.0 --steps 20

python generate.py --model_path /home/yyy/Projects/LAS-Diffusion/results/mine_datasetv02_all/last.ckpt --generate_method generate_unconditional --num_generate 16 --steps 100
#python generate_super_resolution.py --model_path /home/yyy/Projects/LAS-Diffusion/results/super_mine_bulk_02_U/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/mine_bulk_02_U_epoch=3799.ckpt_True_50_0.0_unconditional --save_npy True --save_mesh True --level 0.0 --steps 20
##generate_super_resolution.py --npy_path ./outputs/xxx  This is not to name the output folder, which is named automatically, but just to tell the upper sampler where the output of the corse phase is.

