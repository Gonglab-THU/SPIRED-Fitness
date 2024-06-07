echo start `date`
CUDA_VISIBLE_DEVICES=0 python -u make_esm2_3B_embed.py \
	  --PDB_coords_seq_hdf5 fitness_samples_seq_xyz.h5 \
	  --gpu_esm 0 --train_list fitness_samples.xls \
	  --out_hdf5 fitness_samples_esm2_3B_embed.h5
echo end `date`
