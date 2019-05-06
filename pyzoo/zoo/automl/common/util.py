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


def roll(df, past_seqlen, future_seqlen, y_timeindex=False):
    """
    roll dataframe into sequence samples to be used in TimeSequencePredictor.
    :param df: a dataframe which has been resampled in uniform frequency.
    :param past_seqlen: the length of the past sequence
    :param future_seqlen: the length of the future sequence
    :param y_timeindex: whether to retain the time index col in the y
    :return: tuple (x,y) which can be used directly in TimeSequencePredictor.fit, evaluate
        x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
        dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        y: If y_timeindex=True, y is 3-d numpy array in format (no. of samples, future sequence length, 2).
        Otherwise, y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
        length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
    """
    return (None, None)
