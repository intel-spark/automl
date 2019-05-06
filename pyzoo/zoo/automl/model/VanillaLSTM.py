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
import os

from zoo.automl.model.base import BaseModel


class VanillaLSTM(BaseModel):

    def __init__(self):
        """
        Constructor of Vanilla LSTM model
        """
        self.model = None

    def build(self, **config):
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

    def fit_iter(self, x, y, validation_data=None, verbose=0, epochs=1, batchsize=32):
        """
        fit for one iteration
        :param x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
        dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        :param y: 2-d numpy array in format (no. of samples, future sequence length) if future sequence length > 1,
        or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        :param validation_data: tuple in format (x_test,y_test), data used for validation. If this is specified,
        validation result will be the optimization target for automl. Otherwise, train metric will be the optimization
        target.
        :param config: optimization hyper parameters
        :return: the resulting metric
        """
        if (self.model == None):
            raise Exception("Model is not initialized. Call build() or restore() first before calling fit")

        self.model.fit(x, y, validation_data, verbose, epochs, batchsize)
        ##TODO, get metrics value from History instead of recalculating
        if validation_data == None:
            results = self.model.evaluate(x, y)
        else:
            results = self.model.evaluate(validation_data)
        return results[1]

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

    def save(self, filename="weights_tune_tmp.h5"):
        """
        save model to file.
        :param file: the model file.
        :return:
        """
        self.model.save_weights("weights_tune_tmp.h5")
        os.rename("weights_tune_tmp.h5", filename)
        pass

    def restore(self, file):
        """
        restore model from file
        :param file: the model file
        :return: the restored model
        """
        pass


    def __get_config(self, config):
        """
        Get config and do necessary checking
        :param config:
        :return:
        """
        #lr = config.get("lr", 0.001),
        #lstm_1_units = config.get("lstm_1", 20),
        #dropout_1 = config.get("dropout_1", 0.2),
        #lstm_2_units = config.get("lstm_2", 10),
        #dropout_2 = config.get("dropout_2", 0.2)
        return config
