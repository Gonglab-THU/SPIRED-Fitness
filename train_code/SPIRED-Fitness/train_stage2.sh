#!/usr/bin/env bash

echo start `date`
outdir="./output_stage2"
train_list="./data/stage2/epoch_train_samples_1483/" # training samples lists for SPIRED-Fitness
fitness_list="./data/stage2/fitness_samples.xls" # protein list indicating the fitness samples
valid_list="./data/stage2/CAMEO_L50-700_fitness_samples.xls" # validation samples lists, consists of CAMEO samples and fitness samples

PDB_coords_seq_hdf5="./data/stage2/PDB_till2022-03_xyz_tokens_phi_psi.h5" # coors and seq data of samples for structure prediction training
fit_coords_seq_hdf5="./data/stage2/fitness_data/fitness_samples_seq_xyz.h5" # coors and seq data of samples for Fitness training
CATH_coords_seq_hdf5="./data/stage2/CATH4.2_xyz_tokens_psi_phi.h5" # coors and seq data of CATH samples for structure prediction training
valid_coords_seq_hdf5="./data/stage2/CAMEO_202208_202308.h5" # validation sets for structure prediction training

esm2_3B_h5="./data/stage2/fitness_samples_esm2_3B_embed.h5" # ESM2 3B embedding for fitness samples, use data/stage2/make_esm2_3b_embed.sh to generate
fitness_data_h5="./data/stage2/fitness_input_label_data.h5" # input feature and label for Fitness training

SPIRED_model="./data/stage2/SPIRED.pth" # checkpoint of SPIRED model
Fitness_checkpoint="./data/stage2/Fitness.pth" # checkpoint of Fitness model from stage1 training

mkdir -p $outdir
CUDA_VISIBLE_DEVICES=1 python -u /export/disk5/chenyinghui/SPIRED-Fitness_union/SPIRED-Fitness_doubleMut_h5/train_SPIRED_fitness_PDBmix.py \
	    --SPIRED_checkpoint $SPIRED_model --Fitness_checkpoint $Fitness_checkpoint \
		--ESM_length_cutoff 800 \
		--lr 0.0001 \
        --weight_loss_struct 0.05 --weight_loss_spearman 1 \
		--epoch_start 1 --epoch_end 120 \
        --batch 1 --train_length 230 --valid_length 1400 \
		--train_cycle 1 --valid_cycle 1 \
        --train_sample_num 1483 --valid_sample_num 1146 \
		--train_list $train_list --valid_list $valid_list --fitness_list $fitness_list \
		--PDB_coords_seq_hdf5 $PDB_coords_seq_hdf5 --CATH_coords_seq_hdf5 $CATH_coords_seq_hdf5 \
		--fit_coords_seq_hdf5 $fit_coords_seq_hdf5 --fitness_data_h5 $fitness_data_h5 \
		--valid_coords_seq_hdf5 $valid_coords_seq_hdf5 \
		--esm2_3B_hdf5 $esm2_3B_h5 \
	    --model_dir $outdir --gpu_spired 0 --gpu_esm 0 --write_pdb 0
echo end `date`
