#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import numpy as np
import pandas as pd


def load_nytaxi_data_df(csv_path, val_split_ratio=0, test_split_ratio=0.1):
    full_df = pd.read_csv(csv_path)
    full_df['datetime'] = pd.to_datetime(full_df['timestamp'])
    val_size = int(len(full_df) * val_split_ratio)
    test_size = int(len(full_df) * test_split_ratio)
    train_df = full_df.iloc[:-(test_size+val_size), :]
    val_df = full_df.iloc[-(test_size+val_size):-test_size, :]
    test_df = full_df.iloc[-test_size:, :]
    return train_df[["datetime", "value"]].copy(), val_df[["datetime", "value"]].copy(), test_df[["datetime", "value"]].copy()


def load_nytaxi_data(npz_path):
    data = np.load(npz_path)
    return data['x_train'], data['y_train'], data['x_test'], data['y_test']
