FROM registry.codeocean.com/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

# install dependencies
ARG DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -n spired_fitness python=3.11 && \
    echo '# >>> conda initialize >>>' >> ~/.bashrc && \
    echo '# !! Contents within this block are managed by "conda init" !!' >> ~/.bashrc && \
    echo '__conda_setup="$('\''/opt/conda/bin/conda'\'' '\''shell.bash'\'' '\''hook'\'' 2> /dev/null)"' >> ~/.bashrc && \
    echo 'if [ $? -eq 0 ]; then' >> ~/.bashrc && \
    echo '    eval "$__conda_setup"' >> ~/.bashrc && \
    echo 'else' >> ~/.bashrc && \
    echo '    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then' >> ~/.bashrc && \
    echo '        . "/opt/conda/etc/profile.d/conda.sh"' >> ~/.bashrc && \
    echo '    else' >> ~/.bashrc && \
    echo '        export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc && \
    echo '    fi' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc && \
    echo 'unset __conda_setup' >> ~/.bashrc && \
    echo '# <<< conda initialize <<<' >> ~/.bashrc && \
    . ~/.bashrc && \
    conda activate spired_fitness && \
    conda install -y --channel=pytorch cpuonly==2.0 pytorch==2.1.0 && \
    conda clean -ya && \
    pip3 install -U --no-cache-dir biopython==1.83 click==8.1.7 einops==0.7.0 fair-esm==2.0.0 pandas==2.2.0

# create a new user
RUN useradd -ms /bin/bash test
USER test
RUN echo "conda activate spired_fitness" >> ~/.bashrc && \
    sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc

# Set up SPIRED working directory
WORKDIR /home/test/spired
COPY --chown=test:test ./scripts ./scripts
COPY --chown=test:test ./run_SPIRED.py ./run_SPIRED.py
COPY --chown=test:test ./run_SPIRED-Fitness.py ./run_SPIRED-Fitness.py
COPY --chown=test:test ./run_SPIRED-Stab.py ./run_SPIRED-Stab.py
COPY --chown=test:test ./example_fitness/test.fasta ./example_fitness/test.fasta
COPY --chown=test:test ./example_stab/test.fasta ./example_stab/test.fasta
COPY --chown=test:test ./example_spired/test.fasta ./example_spired/test.fasta

RUN echo "## Try SPIRED in docker" > README.md && \
    echo "" >> README.md && \
    echo "\`\`\`bash" >> README.md && \
    echo "# run SPIRED" >> README.md && \
    echo "python run_SPIRED.py --fasta_file example_spired/test.fasta --saved_folder example_spired" >> README.md && \
    echo "" >> README.md && \
    echo "# run SPIRED-Fitness" >> README.md && \
    echo "python run_SPIRED-Fitness.py --fasta_file example_fitness/test.fasta --saved_folder example_fitness" >> README.md && \
    echo "" >> README.md && \
    echo "# run SPIRED-Stab" >> README.md && \
    echo "python run_SPIRED-Stab.py --fasta_file example_stab/test.fasta --saved_folder example_stab" >> README.md && \
    echo "\`\`\`" >> README.md

# Download model parameter
WORKDIR /home/test/spired/model
RUN wget https://zenodo.org/records/10675405/files/model.zip && \
    unzip model.zip && \
    rm -f model.zip

WORKDIR /home/test/spired
