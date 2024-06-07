# SPIRED Deep Learning Model Training Repository

This repository provides the necessary scripts and instructions for training the SPIRED deep learning model on protein structures. Below are the detailed steps and environment settings required to successfully train and validate the model across various stages.

## Data Preparation

### Making the Training and Validation Data
- Run `./data/PDB_coord_seq.sh` to create an HDF5 file containing the coordinates and sequences data of training or validation samples.
- Users can utilize their own sample list (`./data/sample_list.txt`) and PDB files (`./data/pdb/`) to construct the HDF5 file as input and label for SPIRED.

## Environment Setup
- It is recommended to use the `spired_fitness` conda environment.
- Install necessary Python packages using pip:
  ```bash
  pip install h5py scipy
  ```

## Training Stages

### Training Stage 1
- **Data**: 24,179 clusters of protein chains before May 2020. This includes 22,179 clusters for training and 2,000 for validation.
  - **Training Set**: 22,179 clusters in each sample list file of directory 'stage1_train_sample_lists' (e.g., `epoch_1.txt`).
  - **Validation Set**: 2,000 clusters in 'stage1_validation_sample_lists/valid_PDB_before2020_5.xls'.
- **Learning Rate**:
  - `train_stage1_epoch1-23.sh`: Learning rate starts at 1e-6, increasing to 1e-3 and remains until the 23rd epoch.
  - `train_stage1_epoch24-31.sh`: Learning rate is set at 5e-4 from the 24th to 31st epoch.
- **Cropping Size**: 256 amino acids.
- **GPU Requirements**: Single NVIDIA 80G A100 GPU for stages 1 to 3.

### Training Stage 2
- **Data**:
  - **Training Set**: 63,161 protein chains (easy subset) with length < 400 and resolution < 3 Å (`stage2_sample_lists/all_length400_resolution3_before2020_5.txt`).
  - **Validation Set**: 575 proteins from CAMEO (2020~2021) (`stage2_sample_lists/CAMEO_2020_2021.xls`).
- **Learning Rate**:
  - From 32nd to 35th epoch: 5e-4.
  - From 36th to 40th epoch: 1e-4.
- **Run Script**: `train_stage2.sh`.
- **Cropping Size**: 256 amino acids.

### Training Stage 3
- **Data**: 113,609 PDB chains (before March 2022) and 24,183 CATH domains.
  - **Training Duration**: ~23,000 updates.
- **Learning Rate**: Adjusts from 1e-4 to 5e-5.
- **Run Script**: `train_stage3.sh`.
- **Cropping Size**: 256 amino acids.

### Training Stage 4
- **Data**: Same as Stage 3.
- **Learning Rate**: Adjusts from 5e-5 to 1e-5.
- **Run Script**: `train_stage4.sh`.
- **Cropping Sizes**:
  - 350 amino acids for 18,000 updates.
  - 420 amino acids for 12,000 updates.
- **GPU Requirements**: 3 NVIDIA 80G A100 GPUs for Stage 4.

## General Notes
- Ensure that you activate the correct conda environment and have all dependencies installed before starting the training process.
- For large datasets and extended training periods, monitor resource utilization to optimize GPU usage and prevent potential bottlenecks.
