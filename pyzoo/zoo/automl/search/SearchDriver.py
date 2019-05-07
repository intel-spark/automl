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

import os
import numpy as np
from ray import tune

from zoo.automl.common.tunehelper import *


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


class SearchDriver(object):
    """
    Tune driver
    """

    def __init__(self):
        self.pipeline = None

    def run(self, input_df,
            feature_transformers=None,
            model=None,
            validation_df=None,
            metric="mean_squared_error",
            **config):
        """
        Run tuning
        :param input_df: input dataframe
        :param feature_transformers: feature transformers
        :param model: model or model selector
        :param validation_df: validation dataframe
        :param metric: evaluation metric
        :param config: hyper paramter configurations
        :return:
        """
        train_func = self.__prepare_train_func(self, input_df, feature_transformers, model, validation_df, metric)
        trials = self.__run_trials(train_func, **config)
        # TODO ensemble models
        return trials

    def get_pipeline(self, trials):
        sorted_trials = get_sorted_trials(trials, metric="reward_metric")
        best_trial = sorted_trials[0]
        # load best model TODO
        best = None
        # TODO
        return self.pipeline

    def __prepare_train_func(self, input_df,
                             feature_transformers,
                             model,
                             validation_df=None,
                             metric="mean_squared_error"
                             ):
        """
        Prepare the train function for ray tune
        :param input_df: input dataframe
        :param feature_transformers: feature transformers
        :param model: model or model selector
        :param validation_df: validation dataframe
        :param metric: the rewarding metric
        :return: the train function
        """

        def train_func(config, tune_reporter):
            # prepare data
            (x_train, y_train) = feature_transformers.fit_transform(input_df, **config)
            validation_data = None
            if (validation_df != None):
                validation_data = feature_transformers.transform(validation_df)

            # build model

            inputshape_x = x_train.shape[1]
            inputshape_y = x_train.shape[-1]
            # TODO add input_shape_x and into config

            model.build(**config)
            # print(model.metrics_names)
            # callbacks = [TuneCallback(tune_reporter)]
            # fit model
            best_reward_m = -999
            reward_m = -999
            for i in range(1, 101):
                result = model.fit_iter(x_train, y_train, validation_data=validation_data, **config),
                if metric == "mean_squared_error":
                    reward_m = (-1) * result
                    # print("running iteration: ",i)
                elif metric == "r_square":
                    reward_m = result
                else:
                    raise ValueError("metric can only be \"mean_squared_error\" or \"r_square\"")
                if reward_m > best_reward_m:
                    model.save("weights_tune.h5")
                tune_reporter(
                    training_iteration=i,
                    reward_metric=reward_m,
                    checkpoint="weights_tune.h5"
                )

        return train_func

    def __run_trials(self, train_func, feature_list, target_list, space, stop):
        # retrieve

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
        return trials
