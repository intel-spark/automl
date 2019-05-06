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

from zoo.automl.model.VanillaLSTM import VanillaLSTM
from zoo.automl.model.base import BaseModel


class TimeSequenceModel(BaseModel):
    """
    TimeSequenceModel. Includes model selection (candidates are time sequence or vanila lstm)
    """
    def build(self, **config):
        """
        build a model from config. This operation involves model selection step.
        :param config: tunable arguments for the model
        :return: self
        """
        pass

    def fit_iter(self, x, y, validation_data=None, **config):
        """
        optimize for one iteration for tuning
        :param config: tunable parameters for optimization
        :return: self
        """
        pass

    def evaluate(self, x, y, metric=None):
        """
        Evaluate the model
        :param x: input
        :param y: target
        :param metric:
        :return: a list of metric evaluation results
        """
        pass

    def predict(self, x):
        """
        Prediction.
        :param x: input
        :return: result
        """
        pass

    def save(self, file):
        """
        save model to file.
        :param file: the model file.
        :return:
        """
        pass

    def restore(self, file):
        """
        restore model from file
        :param file: the model file
        :return: the restored model
        """
        pass
