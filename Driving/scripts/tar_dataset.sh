#!/bin/bash

function tar_with_pbar () {
    tar -cf - $1 -P | (pv -s $(du -sb $1 | awk '{print $1}') > $2)
}

DATA_ROOT=$HOME/data/bnet_dataset_full
cd $DATA_ROOT
tar_with_pbar control control.tar
tar_with_pbar images images.tar