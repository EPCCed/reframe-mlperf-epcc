#!/bin/bash

export DATADIR=/home/eidf095/eidf095/crae-ml/imagenet-1k/data/

tar -xzf $DATADIR/train_images_0.tar.gz -C $DATADIR/train
rm $DATADIR/train_images_0.tar.gz
python data_prep.py

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

