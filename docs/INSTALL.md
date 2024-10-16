# Installation

This codebase is tested on Ubuntu 16.04.7 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n transagent python=3.10

# Activate the environment
conda activate transagent

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone TransAgent code repository and install requirements
```bash
# Clone PromptSRC code base
git clone https://github.com/markywg/transagent.git
cd transagent/

# Install requirements
pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
```