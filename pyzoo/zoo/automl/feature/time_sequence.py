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


class TimeSequenceFeatures(BaseFeatures):
    """
    TimeSequence feature engineering
    """

    def __init__(self, drop_missing=True):
        """
        Constructor.
        :param drop_missing: whether to drop missing values in the curve, if this is set to False, an error will be
        reported if missing values are found. If True, will drop the missing values and won't raise errors.
        """
        self.scalar = MinMaxScaler()
        self.config = None

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
        (x, y) = self._run(**self.config)
        return x, y

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
        (x, y) = self._run(input_df, **self.config)
        return x, y

    def save(self, file):
        """
        save the feature tools internal variables.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file : the file to be saved
        :return:
        """
        pass

    def restore(self, file):
        """
        Restore variables from file
        :param file: the dumped variables file
        :return:
        """
        pass

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
        return input_df

    def _roll(self, dataframe, past_seqlen, future_seqlen):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seqlen: the length of the past sequence
        :param future_seqlen: the length of the future sequence
        :return: tuple (x,y)
        """
        return None, None

    def _scale(self, data):
        """
        Scale the data
        :param data:
        :return:
        """

        np_scaled = self.scalar.fit_transform(data)
        data_s = pd.DataFrame(np_scaled)
        return data_s

    def _run(self, input_df, **config):
        # check input dataframe for missing values
        # generate features
        # TODO generate features for input_df using featuretools or other.
        # selected features
        feature_cols = config.get("selected_features",
                                  np.array(
                                      ["MONTH(datetime)", "WEEKDAY(datetime)", "DAY(datetime)", "HOUR(datetime)",
                                       "PERCENTILE(value)", "IS_WEEKEND(datetime)",
                                       "IS_AWAKE(datetime)", "IS_BUSY_HOURS(datetime)"]))
        target_cols = np.array(["value"])
        cols = np.concatenate([target_cols, feature_cols])

        data_n = input_df[cols]
        # select and standardize data
        data_n = self._scale(data_n)
        # roll data and prepare into array x and y
        (x, y) = self._roll(data_n)
        return x, y


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
        self.config = {"train": True}

    def _get_data(self, train=True):
        if train:
            return self.train_data
        else:
            return self.test_data

    def fit_transform(self, input_df, **config):
        """

        :param input_df:
        :param config:
        :return:
        """
        self.config = config
        return self._get_data(train=True)

    def transform(self, input_df):
        return self._get_data(train=False)

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

