import pandas as pd

from src.methods.LeaveOneOut import leave_one_out_binary
from src.methods.Metrics import create_confusion_matrix


class ModelRunning:
    def __init__(self, epochs, batch_size, lr):
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__lr = lr
        self.__df_xls = self.__clean_dt_xls()
        self.__matrix = list()

    def run_model(self, data_sets, data_sets_names, y_binary, y_binary_simple):
        self.__clean_dt_xls()
        self.__matrix = list()

        count = 0
        for name_ds, ds in zip(data_sets_names, data_sets):
            print(name_ds)
            # sc-> score sp->specificity ss->sensitivity
            sc, sp, ss, pr, cm = leave_one_out_binary(ds, y_binary, y_binary_simple, self.__epochs, self.__batch_size,
                                                      self.__lr)

            pond = 0.35 * ss + 0.25 * sc + 0.2 * sp + 0.2 * pr

            self.__df_xls.loc[count] = [name_ds, sc, sp, ss, pr, pond]

            count += 1
            self.__matrix.append(cm)

        self.__save_report()
        for name, mat in zip(data_sets_names, self.__matrix):
            create_confusion_matrix(mat,"EfficientNet-" +
                               "batch" + str(self.__batch_size) +
                               "epochs" + str(self.__epochs) +
                               "lr" + str(self.__lr) + name, 0)

    @staticmethod
    def __clean_dt_xls():
        return pd.DataFrame(columns=["DS", "ACC", "Specificity", "Sensitivity", "Precision", "Ponderacion"])

    def __save_report(self):
        self.__df_xls.to_excel("reports/xls/EfficientNet-" +
                               "batch" + str(self.__batch_size) +
                               "epochs" + str(self.__epochs) +
                               "lr" + str(self.__lr) +
                               ".xlsx")

    def get_matrix(self):
        return self.__matrix
