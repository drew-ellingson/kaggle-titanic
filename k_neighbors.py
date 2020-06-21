import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import data_helpers as dh
from disp_tools import pd_samp


def knn_train(train_data, labels, n_neighbors=5):
    X, y = train_data, labels.values.ravel()
    X_data = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X_data, columns=X.columns)

    print(pd_samp(X))

    clf = KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, y)
    return clf


def knn_test(knn_clf, test_set):
    y = knn_clf.predict(test_set)
    return y


if __name__ == '__main__':
    X, y = dh.transform_train()
    clf = knn_train(X, y)

    test_set = dh.load_data('test')
    test_set_data = StandardScaler().fit_transform(test_set)
    test_set_scaled = pd.DataFrame(test_set_data, columns=test_set.columns)

    y_guess = knn_test(clf, test_set_scaled)

    test_set['Survived'] = y_guess
    output_write = test_set[['Survived']]
    output_write.to_csv('submissions/knn_sub.csv')
