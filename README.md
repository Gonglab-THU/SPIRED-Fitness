<h1 align="center">SPIRED-Fitness</h1>
<p align="center">An end-to-end framework for the prediction of protein structure and fitness from single sequence</p>

## Install software on Linux (Conda)

1. Download `SPIRED-Fitness`

```bash
git clone https://github.com/Gonglab-THU/SPIRED-Fitness.git
cd SPIRED-Fitness
```

2. Install `Anaconda` / `Miniconda` software

3. Follow the steps below to install the Python package or run the command `conda env create -f environment/spired_fitness.yml`

```bash
conda create -n spired_fitness python=3.11
conda activate spired_fitness

conda install pytorch cpuonly -c pytorch
pip install click==8.1.7
pip install einops==0.7.0
pip install pandas==2.1.4
pip install biopython==1.82
```

4. Follow the steps below to install [GDFold2](https://github.com/Gonglab-THU/GDFold2) or run the command `conda env create -f environment/gdfold2.yml`

:exclamation: install `PyRosetta` at [PyRosetta LICENSE](https://www.pyrosetta.org/home/licensing-pyrosetta)

```bash
conda create -n gdfold2 python=3.11
conda activate gdfold2

conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install biopython==1.82
```

## Install software on Linux (Docker)

1. Download `SPIRED-Fitness`

```bash
git clone https://github.com/Gonglab-THU/SPIRED-Fitness.git
cd SPIRED-Fitness
```

2. Build Docker image from Dockerfile

```bash
docker build -t spired -f ./environment/Dockerfile .
```

3. Start Docker container from the image for the first time

**Note**: Please ensure that the container has a minimum of 50 GB of memory to run the models. Also, remember to adjust the Docker memory setting accordingly.

```bash
docker run -it --name=spired --memory=50g spired /bin/bash
```

## Usage

- You should download the [model parameters](https://zenodo.org/doi/10.5281/zenodo.10589085) and move it into the `model` folder. If you install from docker, then the model parameters have been pre-downloaded.

```bash
# run SPIRED
bash run_spired.sh -i example_spired/test.fasta -o example_spired

# run SPIRED-Fitness
bash run_spired_fitness.sh -i example_fitness/test.fasta -o example_fitness

# run SPIRED-Stab
bash run_spired_stab.sh -i example_stab/test.fasta -o example_stab
```

## Reference

[SPIRED-Fitness: an end-to-end framework for the prediction of protein structure and fitness from single sequence](https://doi.org/10.1101/2024.01.31.578102)
