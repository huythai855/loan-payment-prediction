from src.visualization import visualize
import pandas as pd
import logging
import joblib


def evaluate_model(model, test_features, test_labels):
    accuracy = model.score(test_features, test_labels)
    print('Test score =', accuracy)
    return accuracy


def run():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logging.info('Evaluating the model...')

    model_save_path = "../../models/MLR.pkl"
    data_test_path = "../../data/processed/lpd_test.csv"
    df = pd.read_csv(data_test_path)

    # load the model
    mlr = joblib.load(model_save_path)
    X_test = df[['Principal', 'terms', 'past_due_days',
                 'age', 'education', 'Gender']]
    y_test = df[['loan_status']]
    evaluate_model(mlr, X_test, y_test)
    y_pred = mlr.predict(X_test).round()

    visualize.visualize_model(y_test, y_pred)

    # conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.xticks([0, 1, 2], ['COLLECTION', 'PAIDOFF', 'COLLECTION_PAIDOFF'], fontsize=14)
    #
    # plt.savefig('../../reports/figures/plot.png', dpi=300, bbox_inches='tight')
    # plt.show()
