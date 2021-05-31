import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.backend import image_data_format
import numpy as np


def convert_y(y_labels):
    encoder = LabelEncoder()
    encoder.fit(y_labels)
    encoded_y = encoder.transform(y_labels)
    converted_y = to_categorical(encoded_y)
    return converted_y


def save_numpy_array(wl, nbi, b, m, b_s, m_s):
    np.save('data_numpy/wl.npy', wl)
    np.save('data_numpy/nbi.npy', nbi)
    np.save('data_numpy/b.npy', b)
    np.save('data_numpy/m.npy', m)
    np.save('data_numpy/b_s.npy', b_s)
    np.save('data_numpy/m_s.npy', m_s)


class LoadData:

    def __init__(self):
        self.__num_classes = 3
        self.__input_size = (224, 224)

        self.__paths_wl = ['data/WL/adenoma',
                           'data/WL/hyperplasic',
                           'data/WL/serrated']

        self.__paths_nbi = ['data/NBI/adenoma',
                            'data/NBI/hyperplasic',
                            'data/NBI/serrated']

        self.__valid_images = [".jpg"]

        self.n_adenoma = 40
        self.n_hyperplasic = 21
        self.n_serrated = 15

        # Total Images
        self.n_img = self.n_adenoma + self.n_hyperplasic + self.n_serrated

        self.__input_size_net = (224, 224, 3)
        # Numero de imagenes de adenomas
        # self.n_adenoma = len(next(os.walk(self.__paths_wl[0]))[2])
        # n de img de hyper
        # self.n_hyperplasic = len(next(os.walk(self.__paths_wl[1]))[2])
        # n de img de serrated
        # self.n_serrated = len(next(os.walk(self.__paths_wl[2]))[2])

    def get_num_classes(self):
        return self.__num_classes

    def get_input_size_net(self):
        return self.__input_size_net

    def load_data_sets(self):
        if len(os.listdir('data_numpy')) == 0:
            print("LOAD DATA FROM IMAGES")
            x_wl = self.__load_wl_data()
            x_nbi = self.__load_nbi_data()
            y_b, y_m, y_b_simple, y_m_simple = self.__load_y()

            save_numpy_array(x_wl, x_nbi, y_b, y_m, y_b_simple, y_m_simple)
        else:
            print("LOAD DATA FROM NPY")
            x_wl = np.load('data_numpy/wl.npy')
            x_nbi = np.load('data_numpy/nbi.npy')
            y_b = np.load('data_numpy/b.npy')
            y_m = np.load('data_numpy/m.npy')
            y_b_simple = np.load('data_numpy/b_s.npy')
            y_m_simple = np.load('data_numpy/m_s.npy')

        if image_data_format() == 'channels_first':
            x_wl = x_wl.reshape(x_wl.shape[0], 3, 224, 224)
            self.__input_size_net = (3, 224, 224)

        else:
            x_wl = x_wl.reshape(x_wl.shape[0], 224, 224, 3)
            self.__input_size_net = (224, 224, 3)

        return x_wl, x_nbi, y_b, y_m, y_b_simple, y_m_simple

    def __load_wl_data(self):
        x_wl = list()
        x_wl_names = list()

        for p in self.__paths_wl:
            lst = os.listdir(p)
            lst.sort(key=lambda filt: int(''.join(filter(str.isdigit, filt))))

            for f in lst:
                ext = os.path.splitext(f)[1]

                if ext.lower() not in self.__valid_images:
                    continue

                img = tf.keras.preprocessing.image.load_img(os.path.join(p, f), target_size=self.__input_size)
                x_wl_names.append(f.split(".")[0])
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                x_wl.append(img_array)

        x_wl = np.asarray(x_wl)
        x_wl /= 255
        return x_wl

    def __load_nbi_data(self):
        x_nbi = list()
        x_nbi_names = list()

        for p in self.__paths_nbi:
            lst = os.listdir(p)
            lst.sort(key=lambda filt: int(''.join(filter(str.isdigit, filt))))

            for f in lst:
                ext = os.path.splitext(f)[1]

                if ext.lower() not in self.__valid_images:
                    continue

                img = tf.keras.preprocessing.image.load_img(os.path.join(p, f), target_size=self.__input_size)
                x_nbi_names.append(f.split(".")[0])
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                x_nbi.append(img_array)

        x_nbi = np.asarray(x_nbi)
        x_nbi /= 255
        return x_nbi

    def __load_y(self):
        y_multi = np.concatenate((np.full(self.n_adenoma, 2, dtype=int),
                                  np.full(self.n_hyperplasic, 0, dtype=int),
                                  np.full(self.n_serrated, 1, dtype=int)))

        y_binary = np.concatenate((np.full(self.n_adenoma, 1, dtype=int),
                                   np.full(self.n_hyperplasic, 0, dtype=int),
                                   np.full(self.n_serrated, 1, dtype=int)))

        y_multi_conv = convert_y(y_multi)
        y_binary_conv = convert_y(y_binary)

        return y_binary_conv, y_multi_conv, y_binary, y_multi