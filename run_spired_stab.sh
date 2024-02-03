#!/usr/bin/env bash

set -Eeuo pipefail

#######################################################################
# conda environment
#######################################################################

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

#######################################################################
# software
#######################################################################

export SPIRED_DIR=$(dirname $(realpath $0))

#######################################################################
# pasre parameters
#######################################################################

help() {
    echo -e "Usage:\n"
    echo -e "bash run_spired_stab.sh [-i INPUT] [-o FOLDER]\n"
    echo -e "Description:\n"
    echo -e " \e[1;31m-i\e[0m input the fasta file (e.g. -i ./example_stab/test.fasta)"
    echo -e " \e[1;31m-o\e[0m output folder (e.g. -o example_stab)"
    echo -e "\e[1;31mAll parameters must be set!\e[0m"
    exit 1
}

# check the number of parameters
if [ $# -ne 4 ]; then
    echo -e "\e[1;31mThe number of parameters is wrong!\e[0m"
    help
fi

# check the validity of parameters
while getopts 'i:o:' PARAMETER
do
    case ${PARAMETER} in
        i)
        input=${OPTARG};;
        o)
        folder=$(realpath -e ${OPTARG});;
        ?)
        help;;
    esac
done

shift "$(($OPTIND - 1))"

#######################################################################
# run SPIRED-Stab
#######################################################################

echo -e "Input fasta file: \e[1;31m${input}\e[0m"
echo -e "Output folder: \e[1;31m${folder}\e[0m"

# predict Cα protein structure and ddG/dTm
conda activate spired_fitness
python ${SPIRED_DIR}/run_SPIRED-Stab.py --fasta_file ${input} --saved_folder ${folder}

# predict full-atom protein structure by GDFold2
conda activate gdfold2
python ${SPIRED_DIR}/scripts/GDFold2/fold.py ${input} ${folder} -d cuda:6
python ${SPIRED_DIR}/scripts/GDFold2/relax.py --input ${input} --output ${folder}
