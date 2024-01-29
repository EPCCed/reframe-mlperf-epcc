#!/bin/bash

#SBATCH --job-name=prepare-imagenet
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=[CODE]

eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

DATADIR = /path/to/data/

tar -xzf $DATADIR/train_images_1.tar.gz -C $DATADIR/train
rm $DATADIR/train_images_1.tar.gz
python data_prep.py

tar -xzf $DATADIR/train_images_2.tar.gz -C $DATADIR/train
rm $DATADIR/train_images_2.tar.gz
python data_prep.py

tar -xzf $DATADIR/train_images_3.tar.gz -C $DATADIR/train
rm $DATADIR/train_images_3.tar.gz
python data_prep.py

tar -xzf $DATADIR/train_images_4.tar.gz -C $DATADIR/train
rm $DATADIR/train_images_4.tar.gz
python data_prep.py

tar -xzf $DATADIR/val_images.tar.gz -C $DATADIR/val
rm $DATADIR/val_images.tar.gz
python val_data_prep.py