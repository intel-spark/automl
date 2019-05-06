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
from zoo.automl.tune import TuneDriver
from zoo.automl.model import VanillaLSTM
from zoo.automl.common.tunehelper import *
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
        tsp.fit(X_train,Y_train)
        result = tsp.predict(X_test)

    """

    def __init__(self, logs_dir="~/zoo_automl_logs"):
        """
        Constructor of Time Sequence Predictor
        :param logs_dir where the automl tune logs file located
        """
        self.logs_dir = logs_dir

    def fit(self, x, y, validation_data=None, metric="mean_squared_error"):
        """
        Trains the model for time sequence prediction.
        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.
        :param x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
        dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        :param y: 2-d numpy array in format (no. of samples, future sequence length) if future sequence length > 1,
        or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        :param validation_data: tuple in format (x_test,y_test), data used for validation. If this is specified,
        validation result will be the optimization target for automl. Otherwise, train metric will be the optimization
        target.
        :param metric: String. Metric used for train and validation. Available values are "mean_squared_error" or
        "r_square"
        :return: self
        """

        train_func = self.__prepare_train_func()
        self.best_model = self.tune(train_func)

        return self

    def evaluate(self, x, y, metric=["mean_squared_error"]):
        """
        Evaluate the model on a list of metrics.
        :param x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
        dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        :param y: y: 2-d numpy array in format (no. of samples, future sequence length) if future sequence length > 1,
        or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        :param metric: A list of Strings Available string values are "mean_squared_error", "r_square".
        :return: a list of metric evaluation results.
        """
        return [0.0]

    def predict(self, x, expand_timeindex=False):
        """
        Predict future sequence from past sequence.
        :param x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
        dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        :param expand_timeindex: whether or not append the time index for output. Default is False, output is
        in same format as y in fit and evaluate. If enabled, will automatically extend the time index from the last
        point of past sequence. Time interval will be calculated from the last 2 data points of past sequence.
        For example if we're predicting future 2 data points from last 5 data points.
        if the time indexes of the last 2 data points in past sequence are 2019-01-01 00:30 and 2019-01-01 00:40,
        the generated future sequence will be 2019-01-01 00:50, 019-01-01 01:00
        :return: if expand_timeindex=True, returns 3-d array in format (no. of samples, future sequence length, 2).
        Otherwise, returns 2-d array (no. of samples, future sequence length) if future sequence lengths > 1,
        or 1-d array if future sequence length = 1.
        """
        return np.array([])

    def __prepare_data(self, x, y):
        # all_features = ray.get(feature_matrix_id)
        # feature_cols = config.get("selected_features",
        #                          np.array(
        #                              ["MONTH(datetime)", "WEEKDAY(datetime)", "DAY(datetime)", "HOUR(datetime)",
        #                               "PERCENTILE(value)", "IS_WEEKEND(datetime)",
        #                               "IS_AWAKE(datetime)", "IS_BUSY_HOURS(datetime)"]))
        # target_cols = np.array(["value"])
        # cols = np.concatenate([target_cols, feature_cols])
        # print(cols)
        # data_n = all_features[cols]
        # select and standardize data
        # data_n, _ = scale_df(data_n)
        # (x_train, y_train), (x_test, y_test) = prepare_data(data_n)
        return (x, y)

    def __prepare_train_func(self, x, y, validation_data=None, metric="mean_squared_error"):
        """
        Prepare the train function for ray tune
        :param metric: the rewarding metric
        :return: the train function
        """
        model = VanillaLSTM()

        def load_data(x, y):
            return (x, y)

        def train_func(config, tune_reporter):
            # prepare data
            (x_train, y_train) = load_data()
            # build model
            inputshape_x = x_train.shape[1]
            inputshape_y = x_train.shape[-1]
            model.build_model(inputshape_x, inputshape_y, out_units=1,
                              lr=config.get("lr", 0.001),
                              lstm_1_units=config.get("lstm_1", 20),
                              dropout_1=config.get("dropout_1", 0.2),
                              lstm_2_units=config.get("lstm_2", 10),
                              dropout_2=config.get("dropout_2", 0.2))
            # print(model.metrics_names)
            # callbacks = [TuneCallback(tune_reporter)]
            # fit model
            best_reward_m = -999
            reward_m = -999
            for i in range(1, 101):
                result = model.fit_iter(x_train, y_train, validation_data=(x_test, y_test), verbose=0, epochs=1,
                                        batch_size=config.get("batch_size", 32),
                                        # validation_data=(x_test,y_test),
                                        # callbacks=callbacks
                                        )
                if metric == "mean_squared_error":
                    reward_m = (-1) * result
                    # print("running iteration: ",i)
                elif metric == "r_square":
                    reward_m = result
                else:
                    raise ValueError("metric can only be \"mean_squared_error\" or \"r_square\"")
                if reward_m > best_reward_m:
                    model.save_weights("weights_tune_tmp.h5")
                    os.rename("weights_tune_tmp.h5", "weights_tune.h5")
                tune_reporter(
                    training_iteration=i,
                    reward_metric=reward_m,
                    checkpoint="weights_tune.h5"
                )

        return train_func

    def tune(self, train_func):
        # feature engineering
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
            "neg_mse": -0.05,
            "training_iteration": 20
        }

        ###############################################

        trials = tune.run(
            train_func,
            stop=stop,
            config=space,
            num_samples=5,
            resources_per_trial={"cpu": 2},
            verbose=1,
            reuse_actors=True
        )
        sorted_trials = get_sorted_trials(trials, metric="neg_mse")
        best_trial = sorted_trials[0]
        # load best model
        #TODO




class GoodError(Exception):
    pass


def test_reporter(train_func):
    def mock_reporter(**kwargs):
        assert "neg_mse" in kwargs, "Did not report proper metric"
        assert "checkpoint" in kwargs, "Accidentally removed `checkpoint`?"
        raise GoodError("This works.")

    try:
        train_func({}, mock_reporter)
    except TypeError as e:
        print("Forgot to modify function signature?")
        raise e
    except GoodError:
        print("Works!")
        return 1
    raise Exception("Didn't call reporter...")


if __name__ == "__main__":
    tsp = TimeSequencePredictor()
    assert test_reporter(tsptrain_func)
