from statistics import mean

import numpy as np
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut

from methods.Metrics import *
from models.EfficientNetModel import EfficientNetModel
from aumentations import Augmtetation


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def leave_one_out_binary(x, y, y_no_categorical, epochs, tam_batch, lr):
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    y_predict = list()

    efficient_net_model = EfficientNetModel()

    cnt = 0
    for t_v_i, test_i in loo.split(x):
        print(color.BOLD + 'LOO ' + str(cnt) + ":" + color.END)
        cnt += 1

        # models = createModelMnist(2)
        model = efficient_net_model.model_efficient_net(2, lr)

        x_train = x[t_v_i]
        y_train = y[t_v_i]

        x_train_aug, y_train_aug = Augmtetation.augmentation(x_train, y_train, 100)

        x_test = x[test_i]
        # y_test = y[test_i]

        # no callbacks
        # h = models.fit(X_train, y_train,epochs = epocas ,batch_size= tam_batch, verbose=1)

        h = model.fit(x_train_aug, y_train_aug,
                      epochs=epochs, batch_size=tam_batch,
                      verbose=0, callbacks=[efficient_net_model.get_early_stop()])

        print("Mean Loss ", round(mean(h.history['loss']), 3))
        print("Mean Accuracy ", round(mean(h.history['accuracy']), 3))

        # y_pre = models.predict_classes(X_test)
        # y_pre = (models.predict(X_test) > 0.5).astype("int32")
        y_pre = np.argmax(model(x_test), axis=-1)

        y_predict.append(y_pre)

    # Please use instead:* `np.argmax(models.predict(x), axis=-1)`,   if your models does multi-class classification
    # (e.g. if it uses a `softmax` last-layer activation).* `(models.predict(x) > 0.5).astype("int32")`,
    # if your models does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).

    cm = metrics.confusion_matrix(y_no_categorical, y_predict)

    accuracy, specificity, sensitivity, precision, f1score = get_metrics(cm)

    return round(accuracy, 3), round(specificity, 3), round(sensitivity, 3), round(precision, 3), cm
