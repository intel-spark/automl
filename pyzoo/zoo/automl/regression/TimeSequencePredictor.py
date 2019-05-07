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
from zoo.automl.search.SearchDriver import SearchDriver
from zoo.automl.model import VanillaLSTM
from zoo.automl.feature.time_sequence import TimeSequenceFeatures
from zoo.automl.model.time_sequence import TimeSequenceModel

import os

import ray
from ray import tune


class TimeSequencePredictor(object):
    """
    Trains a model that predicts future time sequence from past sequence.
    Past sequence should be > 1. Future sequence can be > 1.
    For example, predict the next 2 data points from past 5 data points.
    Output have only one target value (a scalar) for each data point in the sequence.
    Input can have more than one features (value plus several features)
    Example usage:
        tsp = TimeSequencePredictor()
        tsp.fit(input_df)
        result = tsp.predict(test_df)

    """

    def __init__(self, logs_dir="~/zoo_automl_logs", drop_missing=True):
        """
        Constructor of Time Sequence Predictor
        :param logs_dir where the automl tune logs file located
        """
        self.logs_dir = logs_dir
        self.pipeline = None
        self.tune = SearchDriver()

    def fit(self, input_df,
            dt_col="datetime",
            target_col="value",
            extra_features_col=None,
            validation_df=None,
            metric="mean_squared_error"):
        """
        Trains the model for time sequence prediction.
        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param dt_col: the datetime index column
        :param target_col: the target col (to be predicted)
        :param extra_features_col: extra features
        :param validation_df: validation data
        :param metric: String. Metric used for train and validation. Available values are "mean_squared_error" or
        "r_square"
        :return: self
        """
        self.pipeline = self._hp_search(input_df, dt_col, target_col, extra_features_col, validation_df, metric)
        return self

    def evaluate(self, input_df,
                 dt_col="datetime",
                 target_col="value",
                 extra_features_col=None,
                 metric=None
                 ):
        """
        Evaluate the model on a list of metrics.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param dt_col: the datetime index column
        :param target_col: the target col (to be predicted)
        :param extra_features_col: extra features
        :param metric: A list of Strings Available string values are "mean_squared_error", "r_square".
        :return: a list of metric evaluation results.
        """
        return self.pipeline.evaluate(input_df, dt_col, target_col, extra_features_col, metric)

    def predict(self, input_df,
                dt_col="datetime",
                target_col="value",
                extra_features_col=None):
        """
        Predict future sequence from past sequence.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param dt_col: the datetime index column
        :param target_col: the target col (to be predicted)
        :param extra_features_col: extra features
        :return: a dataframe with 2 columns, the 1st is the datetime, which is the last datetime of the past sequence.
            values are the predicted future sequence values.
            Example :
            datetime    values
            2019-01-03  np.array([2, 3, ... 9])
        """
        return self.pipeline.evaluate(input_df, dt_col, target_col, extra_features_col)

    def _hp_search(self, input_df, dt_col, target_col, extra_features_col, validation_df, metric):
        # we may have to retrain thie tune.sample_from
        feature_list = ["WEEKDAY(datetime)", "HOUR(datetime)",
                        "PERCENTILE(value)", "IS_WEEKEND(datetime)",
                        "IS_AWAKE(datetime)", "IS_BUSY_HOURS(datetime)"
                        # "DAY(datetime)","MONTH(datetime)", #probabaly not useful
                        ]
        target_list = ["value"]

        space = {
            # "lr": tune.grid_search([0.001, 0.01]),
            # "lr" : tune.sample_from(lambda spec: numpy.random.uniform(0.001,0.1)),
            "selected_features": tune.sample_from(
                lambda spec: np.random.choice(
                    feature_list,
                    size=np.random.randint(low=3, high=len(feature_list), size=1),
                    replace=False)),
            # "lstm_1_units":tune.grid_search([16,32,64]),
            # "dropout_1": tune.sample_from(lambda spec: np.random.uniform(0.1,0.6)),
            # "lstm_2_units": tune.grid_search([16,32,64]),
            # "dropout_2": tune.sample_from(lambda spec: np.random.uniform(0.1,0.6)),
            "batch_size": tune.grid_search([32, 1024]),
            # "lstm_1_units":tune.sample_from(lambda spec: int(np.random.uniform(16, 256))),
            # "lstm_2_units": tune.sample_from(lambda spec: int(np.random.uniform(16, 256))),
        }

        stop = {
            "reward_metric": -0.05,
            "training_iteration": 20
        }

        ##may have problems here
        config = {'feature_list': feature_list,
                  'target_list': target_list,
                  'space': space,
                  'stop': stop}

        searcher = SearchDriver()
        searcher.run(input_df,
                     feature_transformers=TimeSequenceFeatures,
                     model=TimeSequenceModel,
                     validation_df=validation_df,
                     metric=metric,
                     **config)
        return searcher.get_pipeline()


if __name__ == "__main__":
    tsp = TimeSequencePredictor()
