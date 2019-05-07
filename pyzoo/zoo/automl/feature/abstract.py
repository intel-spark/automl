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

from abc import ABC, abstractmethod


class BaseFeatures(ABC):

    @abstractmethod
    def fit_transform(self, input_df, **config):
        """
        fit and transform the input dataframe.
        Will refit the scalars to this data if any.
        :param input_df:
        :param config:
        :return:
        """
        pass

    @abstractmethod
    def transform(self, input_df):
        """
        transform the data without refitting the scalars (e.g. minmax scalar)
        :param input_df:
        :return:
        """
        pass

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