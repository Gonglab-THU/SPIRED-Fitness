# Training Repository for Deep Learning Model

This repository contains scripts and data paths required for training a deep learning model in multiple stages. Detailed instructions for each training stage are provided below to ensure a smooth training process.

## Training Stage 1

### Script
- **Script Name**: `train_stage1.sh`

### Input Data
- **File Path**: `./data/stage1`
  - **fitness_samples_train_valid.csv**: Sample list for training and validation.
  - **mutation_data.pt**: Contains input features and labels for training and validation. This file includes data for 2 samples as examples due to the large size of the complete file (128G).

### Learning Rate
- **Initial Learning Rate**: 1e-3.
- **Scheduler**: `ReduceLROnPlateau` with a reduction factor of 0.5 and a patience parameter of 10 epochs.

## Training Stage 2

### Script
- **Script Name**: `train_stage2.sh`

### Input Data
- **File Path**: `./data/stage2`
  - **epoch_train_samples_1483**: Sample lists for each epoch, consisting of 1,000 samples from SPIRED training and 482 samples from fitness training (stage 1).
  - **Note**: Input files required by `train_stage2.sh` (except for `esm2_3B_h5` and `fitness_data_h5`) can be downloaded from Zenodo at [this link](https://doi.org/10.5281/zenodo.12560925).

### Learning Rate
- **SPIRED Module**: The learning rate is fixed at 1e-5.
- **Fitness Module**: 
  - **Initial Learning Rate**: 1e-4.
  - **Adjustment**: Manually adjusted to 1e-5 based on performance and training feedback.

## General Guidelines
- Before starting the training process, ensure that all necessary data files are properly downloaded and placed in the correct directories as specified.
- Monitor the learning rate adjustments and model performance closely to make timely optimizations and achieve the best results.
