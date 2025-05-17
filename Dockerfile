FROM continuumio/miniconda3

# Set working dir
WORKDIR /ml

# Copy and create Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate env, install pip deps
COPY requirements.txt .
RUN /bin/bash -c "source activate ml && pip install -r requirements.txt"

# Set path so environment is always active
ENV PATH /opt/conda/envs/ml/bin:$PATH

# Copy project files
COPY . .
