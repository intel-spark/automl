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
from featuretools import TransformFeature

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

    def __init__(self, future_seq_len=1, dt_col="datetime", target_col="value", extra_features_col=None, drop_missing=True):
        """
        Constructor.
        :param drop_missing: whether to drop missing values in the curve, if this is set to False, an error will be
        reported if missing values are found. If True, will drop the missing values and won't raise errors.
        """
        # self.scaler = MinMaxScaler()
        self.scaler = StandardScaler()
        self.config = None
        self.dt_col = dt_col
        self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.feature_data = None
        self.drop_missing = drop_missing
        self.generate_feature_list = None
        self.past_seqlen = None
        self.future_seqlen = future_seq_len


    def fit_transform(self, input_df, **config):
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
        # print(input_df.shape)
        feature_data = self._get_features(input_df, self.config)
        self.scaler.fit(feature_data)
        data_n = self._scale(feature_data)
        assert np.mean(data_n[0]) < 1e-5
        (x, y) = self._roll_train(data_n, past_seqlen=self.past_seqlen, future_seqlen=self.future_seqlen)

        return x, y

    def transform(self, input_df, is_train=True):
        """
        Transform data into features using the preset of configurations from fit_transform
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param is_train: If the input_df is for training.
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
            length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        """
        if self.config is None or self.past_seqlen is None:
            raise Exception("Needs to call fit_transform or restore first before calling transform")
        # generate features
        feature_data = self._get_features(input_df, self.config)
        # select and standardize data
        data_n = self._scale(feature_data)
        if is_train:
            (x, y) = self._roll_train(data_n, past_seqlen=self.past_seqlen, future_seqlen=self.future_seqlen)
            return x, y
        else:
            x = self._roll_test(data_n, past_seqlen=self.past_seqlen)
            self._get_y_pred_dt(input_df, self.past_seqlen)
            return x

    def post_processing(self, y_pred):
        """
        Used only in pipeline predict, after calling self.transform(input_df, is_train=False).
        Post_processing includes converting the predicted array to dataframe and scalar inverse transform.
        :param y_pred: Model prediction result (ndarray).
        :return: Un_scaled dataframe with datetime.
        """
        # for standard scalar
        value_mean = self.scaler.mean_[0]
        value_scale = self.scaler.scale_[0]
        y_unscale = y_pred * value_scale + value_mean
        self.y_pred_dt["value"] = y_unscale
        return self.y_pred_dt

    def save(self, file_path):
        """
        save the feature tools internal variables.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file : the file to be saved
        :return:
        """
        # for StandardScaler()
        np.savez(file_path, mean=self.scaler.mean_, scale=self.scaler.scale_)
        print("The mean and scale value of StandardScalar are saved in " + file_path)
        # with open(file, 'w') as output_file:
            # for StandardScaler()
            # json.dump({"mean": self.scaler.mean_, "scale": self.scaler.scale_}, output_file)
            # for minmaxScaler()
            # json.dump({"min": self.scaler.min_, "scale": self.scaler.scale_}, output_file)

    def restore(self, file_path, **config):
        """
        Restore variables from file
        :param file_path: the dumped variables file
        :return:
        """
        result = np.load(file_path)
        # with open(file, 'r') as input_file:
        #     result = json.load(input_file)

        # for StandardScalar()
        self.scaler = StandardScaler()
        self.scaler.mean_ = result["mean"]
        self.scaler.scale_ = result["scale"]

        self.config = self._get_feat_config(**config)
        # print(self.scaler.transform(input_data))

        # for MinMaxScalar()
        # self.scaler = MinMaxScaler()
        # self.scaler.min_ = result["min"]
        # self.scaler.scale_ = result["scale"]
        # print(self.scaler.transform(input_data))

    def get_feature_list(self, input_df):
        feature_matrix, feature_defs = self._generate_features(input_df)
        return [feat.generate_name() for feat in feature_defs if isinstance(feat, TransformFeature)]

    def _get_feat_config(self, **config):
        """
        Get feature related arguments from global hyper parameter config and do necessary error checking
        :param config: the global config (usually from hyper parameter tuning)
        :return: config only for feature engineering
        """
        self._check_config(**config)
        feature_config_names = ["selected_features", "past_seqlen"]
        feat_config = {}
        for name in feature_config_names:
            if name not in config:
                continue
                # raise KeyError("Can not find " + name + " in config!")
            feat_config[name] = config[name]
        self.past_seqlen = feat_config.get("past_seqlen", 50)
        return feat_config

    def _check_input(self, input_df):
        """
        Check dataframe for integrity. Requires time sequence to come in uniform sampling intervals.
        :param input_df:
        :return:
        """
        # check NaT in datetime
        input_df = input_df.reset_index()
        datetime = input_df[self.dt_col]
        is_nat = pd.isna(datetime)
        if is_nat.any(axis=None):
                raise ValueError("Missing datetime in input dataframe!")

        # check uniform (is that necessary?)
        interval = datetime[1] - datetime[0]

        if not all([datetime[i] - datetime[i - 1] == interval for i in range(1, len(datetime))]):
            raise ValueError("Input time sequence intervals are not uniform!")

        # check missing values
        if not self.drop_missing:
            is_nan = pd.isna(input_df)
            if is_nan.any(axis=None):
                raise ValueError("Missing values in input dataframe!")

        return input_df

    def _roll_data(self, data, seq_len):
        result = []
        mask = []
        for i in range(len(data) - seq_len + 1):
            result.append(data[i: i + seq_len])

            if pd.isna(data[i: i + seq_len]).any(axis=None):
                mask.append(0)
            else:
                mask.append(1)

        return np.asarray(result), np.asarray(mask)

    def _roll_train(self, dataframe, past_seqlen, future_seqlen):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        roll_train: split the whole dataset apart to build (x, y).
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
        y = dataframe[past_seqlen:][0].values
        output_x, mask_x = self._roll_data(x, past_seqlen)
        output_y, mask_y = self._roll_data(y, future_seqlen)
        # assert output_x.shape[0] == output_y.shape[0], "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1) & (mask_y == 1)
        return output_x[mask], output_y[mask]

    def _roll_test(self, dataframe, past_seqlen):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        roll_test: the whole dataframe is regarded as x.
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seqlen: the length of the past sequence
        :return: x
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
        """
        x = dataframe.values
        output_x, mask_x = self._roll_data(x, past_seqlen)
        # assert output_x.shape[0] == output_y.shape[0], "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1)
        return output_x[mask]

    def _get_y_pred_dt(self, input_df, past_seqlen):
        """
        :param dataframe:
        :return:
        """
        input_df = input_df.reset_index(drop=True)
        output_df = input_df.loc[past_seqlen:, [self.dt_col]].copy()
        output_df = output_df.reset_index(drop=True)
        time_delta = output_df.iloc[-1] - output_df.iloc[-2]
        last_time = output_df.iloc[-1] + time_delta
        output_df.loc[-1] = last_time
        self.y_pred_dt = output_df

    def _scale(self, data):
        """
        Scale the data
        :param data:
        :return:
        """
        np_scaled = self.scaler.transform(data)
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
                                              target_entity="time_seq",
                                              agg_primitives=["count"],
                                              trans_primitives=["month", "weekday", "day", "hour",
                                                                "is_weekend", IsAwake, IsBusyHours])
        return feature_matrix, feature_defs

    # def write_generate_feature_list(self, feature_defs):
    #     # to be confirmed
    #     self.generate_feature_list = [feat.generate_name() for feat in feature_defs if isinstance(feat, TransformFeature)]
    #
    # def get_generate_features(self):
    #     if self.generate_feature_list is None:
    #         raise Exception("Needs to call fit_transform first before calling get_generate_features")
    #     return self.generate_feature_list

    def _get_features(self, input_df, config):
        feature_matrix, feature_defs = self._generate_features(input_df)
        # self.write_generate_feature_list(feature_defs)
        feature_cols = np.asarray(config.get("selected_features"))
        target_cols = np.array([self.target_col])
        cols = np.concatenate([target_cols, feature_cols])
        return feature_matrix[cols]

    def _get_optional_parameters(self):
        return set(["past_seqlen"])

    def _get_required_parameters(self):
        return set(["selected_features"])


if __name__ == "__main__":
    from zoo.automl.common.util import load_nytaxi_data_df
    csv_path = "../../../../data/nyc_taxi.csv"
    train_df, val_df, test_df = load_nytaxi_data_df(csv_path)

    feat = TimeSequenceFeatures(dt_col="datetime", target_col="value", drop_missing=True)
    config = {"selected_features": ["MONTH(datetime)", "WEEKDAY(datetime)", "DAY(datetime)",
                                    "HOUR(datetime)", "IS_WEEKEND(datetime)",
                                       "IS_AWAKE(datetime)", "IS_BUSY_HOURS(datetime)"],
              "lr": 0.001}
    feature_list = feat.get_feature_list(train_df)
    # print(feature_list)

    train_X, train_Y = feat.fit_transform(train_df, **config)
    print(train_X.shape)
    feat.save("feature_config.npz")

    # feat.restore("StandardScaler.npz")
    new_ft = TimeSequenceFeatures()
    new_ft.restore("feature_config.npz", **config)

    train_X_1 = new_ft.transform(train_df[:-1], is_train=False)

    # test_X = new_ft.transform(test_df[:-1], is_train=False)
    # print(test_X.shape)
    #
    # # ft_1 = TimeSequenceFeatures()
    # _, test_Y = new_ft.transform(test_df, is_train=True)
    # print(test_Y.shape)
    # print(train_Y)
    # print(np.mean(np.reshape(train_Y, )))
    output_df = new_ft.post_processing(train_Y)
    print("*"*10 + "output_df" + "*"*10)
    print(output_df.loc[:10, "value"])
    print("*"*10 + "train_df" + "*"*10)
    test_df = test_df.reset_index(drop=True)
    print(train_df.loc[50   :60, "value"])



#     feature_list = feat.get_generate_features()
#     print(feature_list)
# #
# class DummyTimeSequenceFeatures(BaseFeatures):
#     """
#     A Dummy Feature Transformer that just load prepared data
#     use flag train=True or False in config to return train or test
#     """
#
#     def __init__(self, file_path):
#         """
#         the prepared data path saved by in numpy.savez
#         file contains 4 arrays: "x_train", "y_train", "x_test", "y_test"
#         :param file_path: the file_path of the npz
#         """
#         from zoo.automl.common.util import load_nytaxi_data
#         x_train, y_train, x_test, y_test = load_nytaxi_data(file_path)
#         self.train_data = (x_train, y_train)
#         self.test_data = (x_test, y_test)
#         self.is_train = False
#
#     def _get_data(self, train=True):
#         if train:
#             return self.train_data
#         else:
#             return self.test_data
#
#     def fit(self, input_df, **config):
#         """
#
#         :param input_df:
#         :param config:
#         :return:
#         """
#         self.is_train = True
#
#     def transform(self, input_df):
#         x, y = self._get_data(self.is_train)
#         if self.is_train is True:
#             self.is_train = False
#         return x, y
#
#     def _get_optional_parameters(self):
#         return set()
#
#     def _get_required_parameters(self):
#         return set()
#
#     def save(self, file_path, **config):
#         """
#         save nothing
#         :param file_path:
#         :param config:
#         :return:
#         """
#         pass
#
#     def restore(self, file_path, **config):
#         """
#         restore nothing
#         :param file_path:
#         :param config:
#         :return:
#         """
#         pass
