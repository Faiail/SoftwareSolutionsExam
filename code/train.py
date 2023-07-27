import mlflow
import sklearn
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier


def main():
    mlflow.start_run()
    mlflow.set_experiment('style_classification')
    train_set = pd.read_csv('data/train_data.csv', index_col=0).values
    X_train, y_train = train_set[:, :-1], train_set[:, -1]

    test_set = pd.read_csv('data/test_data.csv', index_col=0).values
    X_test, y_test = test_set[:, :-1], test_set[:, -1]
    model_kwargs = {
        'activation': 'relu',
        'solver': 'adam',
        'batch_size': 1024,
        'learning_rate': 'adaptive',
    }
    model = MLPClassifier(**model_kwargs)
    mlflow.log_params(model_kwargs)
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    mlflow.log_artifact('model.pkl')

    train_out = model.predict(X_train)
    train_acc = sklearn.metrics.accuracy_score(y_train, train_out)
    train_f1 = sklearn.metrics.f1_score(y_train, train_out, average='macro')

    # validation
    test_out = model.predict(X_test)
    test_acc = sklearn.metrics.accuracy_score(y_test, test_out)
    test_f1 = sklearn.metrics.f1_score(y_test, test_out, average='macro')

    mlflow.log_metrics(
        {
            'train_acc': train_acc,
            'train_f1': train_f1,

            'test_acc': test_acc,
            'test_f1': test_f1,
        }
    )
    mlflow.end_run()


if __name__ == '__main__':
    main()