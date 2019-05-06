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

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras

from zoo.automl.model.base import BaseModel


class VanillaLSTM(BaseModel):

    def __init__(self):
        """
        Constructor of Vanilla LSTM model
        :param model: initialize using an existing model
        """
        self.model = None

    @staticmethod
    def __identity(r):
        return r

    @staticmethod
    def __negative(r):
        return (-1) * r

    def build_model(self, **config):
        """
        build vanilla LSTM model
        :param config: model hyper parameters
        :return: self
        """
        model = Sequential()
        model.add(LSTM(
            input_shape=(config['input_shape_x'], config['inputshape_y']),
            units=config['lstm_1_units'],
            return_sequences=True))
        model.add(Dropout(config['dropout_1']))

        model.add(LSTM(
            units=config['lstm_2_units'],
            return_sequences=False))
        model.add(Dropout(config['dropout_2']))

        model.add(Dense(units=config['out_units']))
        model.compile(loss='mse', metrics=[config['metric']], optimizer=keras.optimizers.RMSprop(lr=config['lr']))
        self.model = model
        return self

    def fit_iter(self, x_train, y_train, validation_data=None, verbose=0, epochs=1, batchsize=32):
        """
        fit for one iteration
        :param config: optimization hyper parameters
        :return: the resulting metric
        """
        if (self.model == None):
            raise Exception("Call build_model first before calling fit")

        self.model.fit(x_train, y_train, validation_data, verbose, epochs, batchsize)
        ##TODO, get metrics value from History instead of recalculating
        if validation_data == None:
            results = self.model.evaluate(x_train, y_train)
        else:
            results = self.model.evaluate(validation_data)
        return results[1]
