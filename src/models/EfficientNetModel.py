from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import efficientnet.keras as efn


class EfficientNetModel:
    def __init__(self):
        self.earlystop_callback = EarlyStopping(
            monitor='accuracy', min_delta=0.001,
            patience=2)

        self.base_model = efn.EfficientNetB0(input_shape=(224, 224, 3), 
                                  include_top=False, weights='imagenet')
        for layer in self.base_model.layers:
            layer.trainable = False

    def model_efficient_net(self, n_class, lr):
        model = Sequential()
        model.add(self.base_model)
        model.add(Flatten())
        model.add(Dense(n_class, activation="sigmoid"))

        model.compile(loss=binary_crossentropy,
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        return model

    def get_early_stop(self):
        return self.earlystop_callback
