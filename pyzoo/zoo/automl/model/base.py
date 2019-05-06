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


class BaseModel(object):
    """
    base model for automl tuning
    """

    def __init__(self):
        pass

    def build_model(self, **config):
        """
        build a model
        :param config: tunable arguments for the model
        :return: self
        """
        pass

    def fit_iter(self, **config):
        """
        optimize for one step for tune
        :param config: tunable parameters for optimization
        :return: self
        """
        pass
