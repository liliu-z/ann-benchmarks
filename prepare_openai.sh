#!/bin/bash

sudo apt-get install python3-pip -y
pip3 install pandas
pip3 install h5py
pip3 install numpy
pip3 install pyarrow

if [ -e ./data ]; then
  cd ./data
else
  mkdir data
  cd ./data
fi

wget assets.zilliz.com/benchmark/openai_medium_500k/shuffle_train.parquet
wget assets.zilliz.com/benchmark/openai_medium_500k/test.parquet
wget assets.zilliz.com/benchmark/openai_medium_500k/neighbors.parquet

cp ../make_hdf5.py ./
python3 make_hdf5.py

rm shuffle_train.parquet
rm test.parquet
rm neighbors.parquet
rm make_hdf5.py

