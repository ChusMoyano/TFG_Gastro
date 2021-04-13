import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_confusion_matrix(cm, name_clf, tipo_de_clas):
    if tipo_de_clas == 0:
        labels = ["hyperplasic", "adenoma/serrated"]
        ruta = "matrices/"

    if tipo_de_clas == 1:
        labels = ["hyperplasic", "serrated", "adenoma"]
        ruta = "matrices/"

    con_mat_df = pd.DataFrame(cm,
                              index=labels,
                              columns=labels)

    figure = plt.figure(figsize=(5, 5))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: " + name_clf)
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.show()
    plt.savefig(ruta + name_clf + ".png", bbox_inches='tight')
    plt.close()


def get_metrics(cmf):
    tp, fp, fn, tn = cmf.ravel()

    accuracy = (tp + tn) / sum([tp, fp, fn, tn])

    specificity = tn / (tn + fp)

    sensitivity = tp / (tp + fn)

    precision = tp / (tp + fp)

    f1 = 2 * ((sensitivity * precision) / (sensitivity + precision))

    return round(accuracy, 3), round(specificity, 3), round(sensitivity, 3), round(precision, 3), round(f1, 3)