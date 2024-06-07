
echo start `date`
outdir="./output_stage2"
train_list="./stage2_sample_lists/all_length400_resolution3_before2020_5.xls"
valid_list="./stage2_sample_lists/CAMEO_2020_2021.xls"
# users can use ./data/PDB_coord_seq.sh make hdf5 files for training and validation
coords_seq_hdf5="./data/xyz_tokens_label8_psi_phi_before2020_5.h5"
valid_coords_seq_hdf5="./data/CAMEO_2020_2021.h5"
SPIRED_saved_model="output_stage1/SPIRED_Model_Para_epoch31.pth"

mkdir $outdir
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_SPIRED_stage2.py \
                --SPIRED_saved_model $SPIRED_saved_model \
		--epoch_start 32 --epoch_end 40 \
                --batch 64 --maxlen 256 \
                --lr_start 0.0005 --lr_end 0.0005 \
		--train_sample_num 63161 --valid_sample_num 575 \
		--train_list $train_list --valid_list $valid_list \
		--PDB_coords_seq_hdf5 $coords_seq_hdf5 \
                --valid_coords_seq_hdf5 $valid_coords_seq_hdf5 \
                --valid_length 600 \
	        --out_dir $outdir --gpu_SPIRED 0 --gpu_esm 0
echo end `date`
