<h1 align="center">SPIRED-Fitness</h1>
<p align="center">An end-to-end framework for the prediction of protein structure and fitness from single sequence</p>

## Install software on Linux

1. download `SPIRED-Fitness`

```bash
git clone https://github.com/Gonglab-THU/SPIRED-Fitness.git
cd SPIRED-Fitness
```

2. install `Anaconda` / `Miniconda` software

3. install Python packages

```bash
conda create -n spired_fitness python=3.11
conda activate spired_fitness

conda install pytorch cpuonly -c pytorch
pip install click
pip install einops
pip install pandas
pip install biopython
```

4. install [GDFold2](https://github.com/Gonglab-THU/GDFold2)

## Usage

* You should download the [model parameters](https://zenodo.org/records/10589086) and move it into the `model` folder

```bash
# run SPIRED-Fitness
bash run_spired_fitness.sh -i example_fitness/test.fasta -o example_fitness

# run SPIRED-Stab
bash run_spired_stab.sh -i example_stab/test.fasta -o example_stab
```

## Reference

[SPIRED-Fitness: an end-to-end framework for the prediction of protein structure and fitness from single sequence](https://doi.org/10.1101/2024.01.31.578102)
