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
import os
import shutil
import tempfile

import pytest
from zoo.automl.feature.time_sequence import *


class TestTimeSequenceFeature:

    def test_get_feature_list(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"datetime": dates, "values": values})
        feat = TimeSequenceFeatures(dt_col="datetime", target_col="values", drop_missing=True)
        feature_list = feat.get_feature_list(df)
        assert set(feature_list) == set(['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)', 'DAY(datetime)',
                                'IS_WEEKEND(datetime)', 'WEEKDAY(datetime)', 'MONTH(datetime)'])

    def test_fit_transform(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"datetime": dates, "values": values})
        config = {"selected_features": ['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)'],
                  "past_seqlen": 2}
        feat = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                    target_col="values", drop_missing=True)
        x, y = feat.fit_transform(df, **config)
        assert x.shape == (6, 2, 4)
        assert y.shape == (6, 1)
        assert np.mean(np.concatenate((x[0, :, 0], y[:, 0]), axis=None)) < 1e-5

    def test_fit_transform_input_datetime(self):
        # if the type of input datetime is not datetime64, raise an error
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"datetime": dates.strftime('%m/%d/%Y'), "values": values})
        config = {"selected_features": ['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)'],
                  "past_seqlen": 2}
        feat = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                    target_col="values", drop_missing=True)

        with pytest.raises(ValueError) as excinfo:
            feat.fit_transform(df, **config)
        assert 'np.datetime64' in str(excinfo.value)

        # if there is NaT in datetime, raise an error
        df.loc[1, "datetime"] = None
        with pytest.raises(ValueError, match=r".* datetime .*"):
            feat.fit_transform(df, **config)

    def test_fit_transform_input_data(self):
        # if there is NaN in data other than datetime, drop the training sample.
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"datetime": dates, "values": values})
        df.loc[2, "values"] = None

        config = {"selected_features": ['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)'],
                  "past_seqlen": 2}
        feat = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                    target_col="values", drop_missing=True)

        x, y = feat.fit_transform(df, **config)
        # mask_x = [1, 0, 0, 1, 1, 1]
        # mask_y = [0, 1, 1, 1, 1, 1]
        # mask   = [0, 0, 0, 1, 1, 1]
        assert x.shape == (3, 2, 4)
        assert y.shape == (3, 1)

    def test_transform_train_true(self):
        dates = pd.date_range('1/1/2019', periods=16)
        values = np.random.randn(16)
        df = pd.DataFrame({"datetime": dates, "values": values})

        config = {"selected_features": ['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)'],
                  "past_seqlen": 2}
        feat = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                    target_col="values", drop_missing=True)

        feat.fit_transform(df[:10], **config)
        val_x, val_y = feat.transform(df[10:], is_train=True)
        assert val_x.shape == (4, 2, 4)
        assert val_y.shape == (4, 1)

    def test_transform_train_false(self):
        dates = pd.date_range('1/1/2019', periods=16)
        values = np.random.randn(16)
        df = pd.DataFrame({"datetime": dates, "values": values})

        config = {"selected_features": ['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)'],
                  "past_seqlen": 2}
        feat = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                    target_col="values", drop_missing=True)

        feat.fit_transform(df[:10], **config)
        test_x = feat.transform(df[10:], is_train=False)
        assert test_x.shape == (5, 2, 4)

    def test_save_restore(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"datetime": dates, "values": values})

        config = {"selected_features": ['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)'],
                  "past_seqlen": 2}
        feat = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                    target_col="values", drop_missing=True)

        train_x, train_y = feat.fit_transform(df, **config)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature")
        try:
            feature_scalar_path = os.path.join(dirname, "feature_scalar.json")
            feat.save(file_path=feature_scalar_path)
            new_ft = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                          target_col="values", drop_missing=True)
            new_ft.restore(file_path=feature_scalar_path, **config)
            test_x = new_ft.transform(df[:-1], is_train=False)
            assert np.array_equal(test_x, train_x)
        finally:
            shutil.rmtree(dirname)

    def test_post_processing(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"datetime": dates, "values": values})

        config = {"selected_features": ['IS_AWAKE(datetime)', 'IS_BUSY_HOURS(datetime)', 'HOUR(datetime)'],
                  "past_seqlen": 2}
        feat = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                    target_col="values", drop_missing=True)

        train_x, train_y = feat.fit_transform(df, **config)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature_")
        try:
            feature_scalar_path = os.path.join(dirname, "feature_scalar.npz")
            feat.save(file_path=feature_scalar_path)
            new_ft = TimeSequenceFeatures(future_seq_len=1, dt_col="datetime",
                                          target_col="values", drop_missing=True)
            new_ft.restore(file_path=feature_scalar_path, **config)
            new_ft.transform(df[:-1], is_train=False)
            output_value_df = new_ft.post_processing(train_y)
            target_df = df[2:].copy().reset_index(drop=True)

            assert output_value_df["datetime"].equals(target_df["datetime"])
            assert np.sum(np.square(output_value_df["values"].values - target_df["values"].values)) < 1e-7

        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
