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


from zoo.automl.pipeline.abstract import Pipeline


class TimeSequencePipeline(Pipeline):

    def __init__(self, feature_transformers, model):
        """
        initialize a pipeline
        :param model: the internal model
        :param feature_transformers: the feature transformers
        """
        self.feature_transformers = feature_transformers
        self.model = model

    def evaluate(self, input_df,
                 future_seq_len=1,
                 dt_col="datetime",
                 target_col="value",
                 extra_features_col=None,
                 metric=["mean_squared_error"]):
        x, y = self.feature_transformers.transform(input_df)
        return self.model.evaluate(x, y, metric)

    def predict(self, input_df,
                future_seq_len=1,
                dt_col="datetime",
                target_col="value",
                extra_features_col=None):
        # there might be no y in the data, TODO needs to fix in TimeSquenceFeatures
        x, _ = self.feature_transformers.transform(input_df)
        return self.model.predict(x)
