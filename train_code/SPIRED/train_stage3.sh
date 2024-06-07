
echo start `date`
outdir="./output_stage3"
train_list="./stage3_sample_lists/PDB_till2022_3_CATH_24183.xls"
valid_list="./stage3_sample_lists/PDB_2022_05-2023-05.xls"
# users can use ./data/PDB_coord_seq.sh make hdf5 files for training and validation
coords_seq_hdf5="./data/PDB_till2022-03_xyz_tokens_phi_psi.h5"
CATH_coords_seq_hdf5="./data/xyz_tokens_label8_psi_phi_CATH4.2_24185.h5"
valid_coords_seq_hdf5="./data/PDB_2022_05-2023-05.h5"
SPIRED_saved_model="output_stage2/SPIRED_Model_Para_epoch40.pth"

mkdir $outdir
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_SPIRED_stage3.py \
	    --SPIRED_saved_model $SPIRED_saved_model \
		--epoch_start 41 --epoch_end 60 \
        --batch 64 --maxlen 256 \
        --lr_start 0.0001 --lr_end 0.0001 \
		--train_sample_num 137792 --valid_sample_num 500 \
		--train_list $train_list --valid_list $valid_list \
		--PDB_coords_seq_hdf5 $coords_seq_hdf5 \
		--CATH_coords_seq_hdf5 $CATH_coords_seq_hdf5 \
		--valid_coords_seq_hdf5 $valid_coords_seq_hdf5 \
        --valid_length 600 \
	    --out_dir $outdir --gpu_SPIRED 0 --gpu_esm 0
echo end `date`
