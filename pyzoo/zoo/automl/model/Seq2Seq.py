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
        self.future_seq_len = None
        self.check_optional_config = check_optional_config

    def _build_train(self, future_seq_len, **config):
        """
        build LSTM Seq2Seq model
        :param config:
        :return:
        """
        super()._check_config(**config)
        self.metric = config.get('metric', 'mean_squared_error')
        self.latent_dim = config.get('latent_dim', 256)
        self.dropout = config.get('dropout', 0.2)

        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, 9), name="encoder_inputs")
        encoder = LSTM(units=self.latent_dim,
                       dropout=self.dropout,
                       return_state=True,
                       name="encoder_lstm")
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None, 1), name="decoder_inputs")
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(self.latent_dim,
                                 dropout=self.dropout,
                                 return_sequences=True,
                                 return_state=True,
                                 name="decoder_lstm")
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                                  initial_state=self.encoder_states)

        self.decoder_dense = Dense(future_seq_len, name="decoder_dense")
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        self.model.compile(loss='mse',
                           metrics=[self.metric],
                           optimizer=keras.optimizers.RMSprop(lr=config.get('lr', 0.001)))
        return self.model

    def _build_inference(self):
        # from our previous model - mapping encoder sequence to state vectors
        encoder_model = Model(self.encoder_inputs, self.encoder_states)

        # A modified version of the decoding stage that takes in predicted target inputs
        # and encoded state vectors, returning predicted target outputs and decoder state vectors.
        # We need to hang onto these state vectors to run the next step of the inference loop.
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model([self.decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)
        return encoder_model, decoder_model

    def decode_sequence(self, input_seq):
        encoder_model, decoder_model = self._build_inference()
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((len(input_seq), 1))

        # Populate the first target sequence with end of encoding series value
        target_seq[:, 0] = input_seq[:, -1, 0]

        # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
        # (to simplify, here we assume a batch of size 1).

        decoded_seq = np.zeros((len(input_seq), self.future_seq_len))

        for i in range(self.future_seq_len):
            output, h, c = decoder_model.predict([target_seq] + states_value)

            decoded_seq[:, i] = output[:, 0, 0]

            # Update the target sequence (of length 1).
            target_seq = np.zeros((len(input_seq), 1))
            target_seq[:, 0] = output[:, 0, 0]

            # Update states
            states_value = [h, c]

        return decoded_seq

    def _get_decoder_inputs(self, x, y):
        """
        lagged target series for teacher forcing
        :param y:
        :return:
        """
        decoder_input_data = np.zeros(y.shape)
        decoder_input_data[1:, ] = y[:-1, ]
        decoder_input_data[0, 0] = x[-1, -1, 0]
        if len(y[1]) > 1:
            decoder_input_data[0, 1:] = y[0, :-1]
        return decoder_input_data

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
        self.future_seq_len = y.shape[1]
        print("future_seq_len is", self.future_seq_len)
        # if model is not initialized, __build the model
        if self.model is None:
            self._build_train(self.future_seq_len, **config)

        decoder_input_data = self._get_decoder_inputs(x, y)
        hist = self.model.fit([x, decoder_input_data], y,
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

    def evaluate(self, x, y, metric=['mean_squared_error']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        e = Evaluator()
        y_pred = self.predict(x)
        return [e.evaluate(m, y, y_pred) for m in metric]

    def predict(self, x):
        """
        Prediction on x.
        :param x: input
        :return: predicted y
        """
        return self.decode_sequence(x)

    def save(self, file_path, **config):
        """
        save model to file.
        :param file_path: the model file.
        :param config: the trial config
        :return:
        """
        self.model.save("seq2seq_tmp.h5")
        os.rename("seq2seq.h5", file_path)
        pass

    def restore(self, file_path, **config):
        """
        restore model from file
        :param file_path: the model file
        :param config: the trial config
        :return: the restored model
        """
        #self.model = None
        #self._build(**config)
        self.model = keras.models.load_model(file_path)
        #self.model.load_weights(file_path)

    def _get_required_parameters(self):
        return {
            # 'input_shape_x',
            # 'input_shape_y',
            # 'out_units'
        }

    def _get_optional_parameters(self):
        return {
            'past_seqlen'
            'latent_dim'
            'dropout',
            'metric',
            'lr',
            'epochs',
            'batch_size'
        }


if __name__ == "__main__":
    model = LSTMSeq2Seq(check_optional_config=False)
    x_train, y_train, x_test, y_test = load_nytaxi_data('../../../../data/nyc_taxi_rolled_split.npz')
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    config = {
        # 'input_shape_x': x_train.shape[1],
        # 'input_shape_y': x_train.shape[-1],
        'out_units': 1,
        'dummy1': 1,
        'batch_size': 1024,
        'epochs': 1
    }

    print("fit_eval:", model.fit_eval(x_train, y_train, validation_data=(x_test, y_test), **config))
    print("evaluate:", model.evaluate(x_test, y_test))
    print("saving model")
    model.save("testmodel.tmp.h5",**config)