#!/bin/bash
sudo apt-get update
sudo apt-get install git python-pip python-virtualenv python3-dev python3-numpy python3-scipy python3-matplotlib ipython3-notebook libxml2-dev zlib1g-dev

mkdir orange3env
virtualenv -p python3 --system-site-packages orange3env
source orange3env/bin/activate

git clone https://github.com/biolab/orange3
cd orange3
pip install -r requirements.txt
python setup.py develop
pip install Orange-Bioinformatics networkx python-igraph

ipython3 notebook
