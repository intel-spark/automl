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

from zoo.automl.feature.abstract import BaseFeatures

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import featuretools as ft
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Text, Numeric, DatetimeTimeIndex
import json


class TimeSequenceFeatures(BaseFeatures):
    """
    TimeSequence feature engineering
    """

    def __init__(self, dt_col="datetime", target_col="value", drop_missing=True):
        """
        Constructor.
        :param drop_missing: whether to drop missing values in the curve, if this is set to False, an error will be
        reported if missing values are found. If True, will drop the missing values and won't raise errors.
        """
        self.scalar = MinMaxScaler()
        # self.scalar = StandardScaler()
        self.config = None
        self.dt_col = dt_col
        self.target_col = target_col
        self.feature_data = None

    def fit(self, input_df, **config):
        """
        Fit data and transform the raw data to features. This is used in training for hyper parameter searching.
        This method will refresh the parameters (e.g. min and max of the MinMaxScaler) if any
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param config: tunable parameters
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
            length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        """
        self.config = self._get_feat_config(**config)
        self._check_input(input_df)
        self.feature_data = self._get_features(input_df, **self.config)
        self.scalar.fit(self.feature_data)


    def transform(self, input_df):
        """
        Transform data into features using the preset of configurations from fit_transform
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
            length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        """
        if self.config is None:
            raise Exception("Needs to call fit_transform first before calling transform")
        # select and standardize data
        data_n = self._scale(self.feature_data)
        # roll data and prepare into array x and y
        (x, y) = self._roll(data_n)

        return x, y

    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler = StandardScaler()
    # scaler.mean_ = 1.0
    # scaler.scale_ = 5.0
    # # scaler.fit(data)
    # print(scaler.transform(data))

    # with open('data.json', 'w') as outfile:
    #     json.dump({"mean": scaler.mean_, "scale": scaler.scale_, "seq": [1, 2, 3, 4], "dic": {"a": 1}}, outfile)

    with open('data.json', 'r') as inputfile:
        result = json.load(inputfile)

    scaler.mean_ = result["mean"]
    scaler.scale_ = result["scale"]
    print(scaler.transform(data))

    def save(self, file):
        """
        save the feature tools internal variables.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file : the file to be saved
        :return:
        """
        with open(file, 'w') as output_file:
            # for StandardScalar()
            json.dump({"mean": self.scaler.mean_, "scale": self.scaler.scale_}, save_file)
            # for minmaxScalar()
            # json.dump({"min": self.scaler.min_, "scale": self.scalar.scale_}, save_file)

    def restore(self, file):
        """
        Restore variables from file
        :param file: the dumped variables file
        :return:
        """
        with open(file, 'r') as input_file:
            result = json.load(input_file)

        # for StandardScalar()
        self.scaler = StandardScaler()
        self.scaler.mean_ = result["mean"]
        self.scaler.scale_ = result["scale"]
        # print(self.scaler.transform(input_data))

        # for MinMaxScalar()
        # self.scaler = MinMaxScaler()
        # self.scaler.min_ = result["min"]
        # self.scaler.scale_ = result["scale"]
        # print(self.scaler.transform(input_data))

    def _get_feat_config(self, config):
        """
        Get feature related arguments from global hyper parameter config and do necessary error checking
        :param config: the global config (usually from hyper paramter tuning)
        :return: config only for feature engineering
        """
        feat_config = {"dummy_arg1": 1, "dummy_arg2": 2}
        return feat_config

    def _check_input(self, input_df):
        """
        Check dataframe for integrity. Requires time sequence to come in uniform sampling intervals.
        :param input_df:
        :return:
        """
        # check missing values
        is_nan = pd.isna(input_df)
        if is_nan.any(axis=None):
            raise ValueError("Missing value in input dataframe!")

        datetime = input_df[self.dt_col]

        interval = datetime[1] - datetime[0]

        if not all([datetime[i] - datetime[i-1] == interval for i in range(1, len(datetime))]):
            raise ValueError("Input time sequence intervals are not uniform!")

        return input_df

    def _roll_data(self, data, seq_len):
        result = []
        for i in range(len(data) - seq_len):
            result.append(data[i : i + seq_len])
        return np.asarray(result)

    def _roll(self, dataframe, past_seqlen, future_seqlen):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seqlen: the length of the past sequence
        :param future_seqlen: the length of the future sequence
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
            length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        """
        x = dataframe[0:-future_seqlen].values
        y = dataframe[past_seqlen-future_seqlen:][0].values
        output_x = self._roll_data(x)
        print(output_x).shape
        output_y = self._roll_data(y)
        print(output_y).shape
        assert output_x.shape[0] == output_y.shape[0], "The shape of output_x and output_y doesn't match! "

        return output_x, output_y

    def _scale(self, data):
        """
        Scale the data
        :param data:
        :return:
        """
        np_scaled = self.scalar.transform(data)
        data_s = pd.DataFrame(np_scaled)
        return data_s

    def _rearrange_data(self, input_df):
        """
        change the input_df column order into [datetime, target, feature1, feature2, ...]
        :param input_df:
        :return:
        """
        cols = input_df.columns.tolist()
        new_cols = [self.dt_col, self.target_col] + [col for col in cols if col != self.dt_col and col != self.target_col]
        rearranged_data = input_df[new_cols].copy
        return rearranged_data

    def _generate_features(self, input_df):
        df = input_df.copy()
        df["id"] = df.index + 1

        es = ft.EntitySet(id="data")
        es = es.entity_from_dataframe(entity_id="time_seq",
                                      dataframe=df,
                                      index="id",
                                      time_index=self.dt_col)

        def is_awake(column):
            hour = column.dt.hour
            return (((hour >= 6) & (hour <= 23)) | (hour == 0)).astype(int)

        def is_busy_hours(column):
            hour = column.dt.hour
            return (((hour >= 7) & (hour <= 9)) | (hour >= 16) & (hour <= 19)).astype(int)

        IsAwake = make_trans_primitive(function=is_awake,
                                       input_types=[DatetimeTimeIndex],
                                       return_type=Numeric)
        IsBusyHours = make_trans_primitive(function=is_busy_hours,
                                           input_types=[DatetimeTimeIndex],
                                           return_type=Numeric)

        feature_matrix, feature_defs = ft.dfs(entityset=es,
                                              target_entity="counts",
                                              agg_primitives=["count"],
                                              trans_primitives=["month", "weekday", "day", "hour", "percentile",
                                                                "is_weekend", IsAwake, IsBusyHours])
        return feature_matrix, feature_defs

    def get_feature_names(self, feature_defs):
        # todo: get feature names from feature_defs
        return ["MONTH(datetime)", "WEEKDAY(datetime)", "DAY(datetime)", "HOUR(datetime)",
                                       "PERCENTILE(value)", "IS_WEEKEND(datetime)",
                                       "IS_AWAKE(datetime)", "IS_BUSY_HOURS(datetime)"]

    def _get_features(self, input_df, **config):
        feature_matrix, feature_defs = self._generate_features(input_df)
        feature_names = self.get_feature_names(feature_defs)
        feature_cols = config.get("selected_features", np.array[feature_names])
        target_cols = np.array([self.target_col])
        cols = np.concatenate([target_cols, feature_cols])
        return feature_matrix[cols]


from zoo.automl.common.util import load_nytaxi_data


class DummyTimeSequenceFeatures(BaseFeatures):
    """
    A Dummy Feature Transformer that just load prepared data
    use flag train=True or False in config to return train or test
    """

    def __init__(self, file_path):
        """
        the prepared data path saved by in numpy.savez
        file contains 4 arrays: "x_train", "y_train", "x_test", "y_test"
        :param file_path: the file_path of the npz
        """
        x_train, y_train, x_test, y_test = load_nytaxi_data(file_path)
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        self.is_train = False

    def _get_data(self, train=True):
        if train:
            return self.train_data
        else:
            return self.test_data

    def fit(self, input_df, **config):
        """

        :param input_df:
        :param config:
        :return:
        """
        self.is_train = True

    def transform(self, input_df):
        x, y = self._get_data(self.is_train)
        if self.is_train is True:
            self.is_train = False
        return x, y

    def _get_optional_parameters(self):
        return set()

    def _get_required_parameters(self):
        return set()

    def save(self, file_path, **config):
        """
        save nothing
        :param file_path:
        :param config:
        :return:
        """
        pass

    def restore(self, file_path, **config):
        """
        restore nothing
        :param file_path:
        :param config:
        :return:
        """
        pass
