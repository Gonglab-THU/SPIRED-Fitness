echo start `date`
python script_stage1/run.py \
        --train_valid_list data/stage1/fitness_samples_train_valid.csv \
        --training_data data/stage1/mutation_data.pt \
        --outdir output_stage1 \ 
echo end `date`