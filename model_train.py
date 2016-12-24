from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, RepeatVector
from keras.layers import Dropout, TimeDistributed
import numpy as np

class Model(object):
    def __init__(self):
        pass

    def save(self, struct_file, weights_file):
        struct_fp = open(struct_file, 'w')
        struct_fp.write(self.model.to_json())

        weights_fp = open(weights_file, 'w')
        model.save_weights(weights_file, overwrite=True)

class TimeModel(Model):
    def __init__(self, data_x, data_y):
        self.model = None
        pass

    def build(self):
        model = Sequential()
        model.add(LSTM(input_dim=1, output_dim=128, return_sequences=False))
        model.add(Dense(128, activation="relu"))
        model.add(RepeatVector(1024))
        model.add(LSTM(128, return_sequences=True))
        model.add(TimeDistributed(Dense(output_dim=1, activation="linear")))
        model.compile(loss="mse", optimizer='adam')

        self.model = model


def build_data(raw_data):
    max_input_size = 0
    for d in raw_data:
        if max_input_size < len(d):
            max_input_size = len(d)

    data = np.zeros()
    for i, d in enumerate(raw_data):
        t = np.zeros()
        tmp = np.array(d)
        np.concatenate((tmp, np.array([[0] * len(d)].T)), axis=1)
        np.concatenate((tmp, np.array([[0, 0, 1]])), axis=0)
        t[: len(d + 1)] = tmp
        data[i] = t

    data_x = data.copy()
    data_y = data[1:] + np.array()
    return data_x, data_y

def main():

    data_x, data_y = build_data()


    model = TimeModel(data_x, data_y)
    model.build()
    model.fit()
    model.save("test.struct", "test.weights")
    pass
