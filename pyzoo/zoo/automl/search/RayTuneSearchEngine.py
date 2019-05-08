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
import ray
from ray import tune

from zoo.automl.common.tunehelper import *
from zoo.automl.search.abstract import *


class RayTuneSearchEngine(SearchEngine):
    """
    Tune driver
    """

    def __init__(self, logs_dir="", ray_num_cpus=6, resources_per_trial=None):
        """
        Constructor
        :param ray_num_cpus: the total number of cpus for ray
        :param resources_per_trial: resources for each trial
        """
        self.pipeline = None
        self.train_func = None
        self.resources_per_trail = resources_per_trial

        ray.init(num_cpus=ray_num_cpus, include_webui=False, ignore_reinit_error=True)

    def compile(self,
                input_df,
                search_space,
                num_samples=1,
                stop=None,
                feature_transformers=None,
                model=None,
                validation_df=None,
                metric="mean_squared_error"):
        """
        Do necessary preparations for the engine
        :param input_df:
        :param search_space:
        :param num_samples:
        :param stop:
        :param feature_transformers:
        :param model:
        :param validation_df:
        :param metric:
        :return:
        """
        self.search_space = self._prepare_tune_config(search_space)
        self.stop_criteria = stop
        self.num_samples = num_samples
        self.train_func = self._prepare_train_func(input_df,
                                                   feature_transformers,
                                                   model,
                                                   validation_df,
                                                   metric)

    def run(self):
        """
        Run trials
        :return: trials result
        """
        trials = tune.run(
            self.train_func,
            stop=self.stop_criteria,
            config=self.search_space,
            num_samples=self.num_samples,
            resources_per_trial=self.resources_per_trail,
            verbose=1,
            reuse_actors=True
        )

        return trials

    def get_best_trials(self, trials, k=1):
        sorted_trials = get_sorted_trials(trials, metric="reward_metric")
        best_trials = sorted_trials[:k]
        return [self._make_trial_output() for t in best_trials]

    def _make_trial_output(self, trial):
        return TrialOutput(config=trial.config,
                           model_path=os.path.join(trial.logdir, trial.last_result["checkpoint"]))

    def test_run(self):
        def mock_reporter(**kwargs):
            assert "reward_metric" in kwargs, "Did not report proper metric"
            assert "checkpoint" in kwargs, "Accidentally removed `checkpoint`?"
            raise GoodError("This works.")

        try:
            self.train_func({'out_units': 1}, mock_reporter)
        except TypeError as e:
            print("Forgot to modify function signature?")
            raise e
        except GoodError:
            print("Works!")
            return 1
        raise Exception("Didn't call reporter...")

    def _prepare_train_func(self,
                            input_df,
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
            if validation_df is not None:
                validation_data = feature_transformers.transform(validation_df)

            # build model
            # model.build(**config)
            # print(model.metrics_names)
            # callbacks = [TuneCallback(tune_reporter)]
            # fit model
            best_reward_m = -999
            reward_m = -999
            for i in range(1, 101):
                result = model.fit_eval(x_train, y_train, validation_data=validation_data, **config)
                if metric == "mean_squared_error":
                    reward_m = (-1) * result
                    # print("running iteration: ",i)
                elif metric == "r_square":
                    reward_m = result
                else:
                    raise ValueError("metric can only be \"mean_squared_error\" or \"r_square\"")
                if reward_m > best_reward_m:
                    best_reward_m = reward_m
                    model.save("weights_tune.h5", config)
                tune_reporter(
                    training_iteration=i,
                    reward_metric=reward_m,
                    checkpoint="weights_tune.h5"
                )

        return train_func

    def _prepare_tune_config(self, space):
        tune_config = {}
        for k, v in space.items():
            if isinstance(v, RandomSample):
                tune_config[k] = tune.sample_from(v.func)
            elif isinstance(v, GridSearch):
                tune_config[k] = tune.grid_search(v.values)
            else:
                tune_config[k] = v
        return tune_config
