from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def visualize_model(y_test, y_pred):
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.title('Confusion Matrix', fontsize=18)

    tick_marks = [0,1,2]
    class_name = ['COLLECTION', 'PAIDOFF', 'COL_PAIDOFF']

    plt.xticks(ticks=tick_marks, labels=class_name, fontsize=14)
    plt.yticks(ticks=tick_marks, labels=class_name, fontsize=14)

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)

    plt.savefig('../../reports/figures/plot.png',
                dpi=300,
                bbox_inches='tight'
                )
