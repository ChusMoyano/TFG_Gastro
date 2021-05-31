import datetime
import time

import tensorflow as tf

from src.methods.LoadData import LoadData
from src.methods.ModelRunning import ModelRunning


def load_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == "__main__":
    load_gpu()

    time1 = time.time()
    load_data = LoadData()

    x_wl, x_nbi, y_b, y_m, y_b_s, y_m_s = load_data.load_data_sets()

    data_sets_names = ["X_WL", "X_NBI"]
    data_sets = [x_wl, x_nbi]
    labels = [y_b, ]
    model_running = ModelRunning(epochs=100, batch_size=32, lr=0.001)
    model_running.run_model(data_sets, data_sets_names, y_b, y_b_s)

    time2 = time.time()

    print(str(datetime.timedelta(seconds=time2 - time1)))

