import pandas as pd
import h5py
import numpy as np

train_df = pd.read_parquet('shuffle_train.parquet')

df_sorted = train_df.sort_values(by='id')

test_df = pd.read_parquet('test.parquet')
neighbors_df = pd.read_parquet('neighbors.parquet')

with h5py.File('openai500k-1536-angular.hdf5', 'w') as h5f:
    train_emb = np.stack(df_sorted['emb'].to_numpy()).astype('f4')
    h5f.create_dataset('train', data=train_emb)

    test_emb = np.stack(test_df['emb'].to_numpy()).astype('f4')
    h5f.create_dataset('test', data=test_emb)

    neighbors_id = np.stack(neighbors_df['neighbors_id'].to_numpy()).astype('i4')
    h5f.create_dataset('neighbors', data=neighbors_id)

    h5f.attrs["distance"] = "angular"
