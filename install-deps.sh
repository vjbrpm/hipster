#!/bin/bash

source ./.venv/bin/activate

#notebooks
pip install jupyter
pip install ipympl

#basic stuff
pip install pandas
pip install polars
pip install openpyxl
pip install fastexcel
pip install numpy
pip install scikit-learn

#jit for accelerating python code
pip install numba

#plotting
pip install matplotlib
pip install SciencePlots

#neural networks
pip install torch
pip install transformers
pip install tensorboard

#this is needed for tensorboard to work, because it is still using pkg_resources that latest setuptools no longer supports
pip install setuptools==80.10.1

#adaptive gradient clipping
pip install git+https://github.com/bluorion-com/ZClip.git

#pytorch lightning with metrics
pip install lightning
pip install torchmetrics

#for using minishlab/potion-multilingual-128M
pip install model2vec
pip install sentence-transformers

#for BART50 tokenizer
pip install sentencepiece

#interpretability
pip install captum

#for JSON serializable classes
pip install pydantic

#sqlalchemy with drivers
pip install SQLAlchemy
pip install psycopg

#hdf5 storage support
pip install h5py

#web sevice
pip install fastapi
pip install uvicorn
