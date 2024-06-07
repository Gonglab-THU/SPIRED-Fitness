
echo start `date`
outdir="./output_stage1"
train_list="./stage1_train_sample_lists/"
valid_list="./stage1_validation_sample_lists/valid_PDB_before2020_5.xls"
# users can use ./data/PDB_coord_seq.sh make hdf5 files for training and validation
coords_seq_hdf5="./data/xyz_tokens_label8_psi_phi_before2020_5.h5"
SPIRED_saved_model="output_stage1/SPIRED_Model_Para_epoch23.pth"

mkdir $outdir
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_SPIRED_stage1.py \
		--SPIRED_saved_model $SPIRED_saved_model \
		--epoch_start 24 --epoch_end 31 \
        --batch 64 --maxlen 256 \
        --lr_start 0.0005 --lr_end 0.0005 \
		--train_sample_num 20998 --valid_sample_num 2000 \
		--train_list_dir $train_list --valid_list $valid_list \
		--PDB_coords_seq_hdf5 $coords_seq_hdf5 \
        --valid_length 600 \
	    --out_dir $outdir --gpu_SPIRED 0 --gpu_esm 0
echo end `date`
