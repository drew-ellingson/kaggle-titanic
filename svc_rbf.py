import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import data_helpers as dh


def svc_train(train_data, labels, n_neighbors=15):
    X, y = train_data, labels.values.ravel()
    X_data = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X_data, columns=X.columns)

    clf = SVC(kernel='rbf', gamma='auto', C=3)
    clf.fit(X, y)
    return clf


def svc_test(svc_clf, test_set):
    y = svc_clf.predict(test_set)
    return y


if __name__ == '__main__':
    X, y = dh.transform_train()
    clf = svc_train(X, y)

    test_set = dh.load_data('test')
    test_set_data = StandardScaler().fit_transform(test_set)
    test_set_scaled = pd.DataFrame(test_set_data, columns=test_set.columns)

    y_guess = svc_test(clf, test_set_scaled)

    test_set['Survived'] = y_guess
    output_write = test_set[['Survived']]
    output_write.to_csv('submissions/svc_sub.csv')
