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
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras
import os

from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator


class LSTMSeq2Seq(BaseModel):

    def __init__(self, check_optional_config=True):
        """
        Constructor of LSTM Seq2Seq model
        """
        self.model = None
        self.check_optional_config = check_optional_config

    def _build(self, **config):
        """
        build LSTM Seq2Seq model
        :param config:
        :return:
        """
        super()._check_config(**config)
        self.metric = config.get('metric', 'mean_squared_error')

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, config.get("input_len", 20)))
        encoder = LSTM(units=config.get('latent_dim', 256),
                       return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, config.get("target_len", 1)))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(config.get('latent_dim', 256),
                            return_sequences=True,
                            return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(config.get("target_len", 1),
                              activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(loss='mse',
                           metrics=[self.metric],
                           optimizer=keras.optimizers.RMSprop(lr=config.get('lr', 0.001)))
        return self.model

    def fit_eval(self, x, y, validation_data=None, **config):
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
        # if model is not initialized, __build the model
        if self.model is None:
            self._build(**config)

        hist = self.model.fit(x, y,
                              validation_data=validation_data,
                              batch_size=config.get('batch_size', 1024),
                              epochs=config.get('epochs', 1),
                              verbose=0
                              )
        # print(hist.history)

        if validation_data is None:
            # get train metrics
            # results = self.model.evaluate(x, y)
            result = hist.history.get(self.metric)[0]
        else:
            result = hist.history.get('val_' + str(self.metric))[0]
        return result