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


class Pipeline(object):
    """
    The pipeline object which is used to store the series of transformation of features and model
    """
    def __init__(self, feature_transformers, model):
        """
        initialize a pipeline
        :param model: the internal model
        :param feature_transformers: the feature transformers
        """
        self.model = model
        self.feature_transformers = feature_transformers

    def save(self, file):
        """
        save the pipeline to a file
        :param file: the pipeline file
        :return: a pipeline object
        """
        pass

    def restore(self, file):
        """
        restore the pipeline from a file
        :param file: the pipeline file
        :return:
        """
        pass
