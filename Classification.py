# Import data and modules
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def pre_processing():
    iris = datasets.load_iris()
    # We'll use the petal length and width only for this analysis
    test_size = 0.3
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Place the iris data into a pandas dataframe
    iris_df = pd.DataFrame(iris.data[:, [2, 3]], columns=iris.feature_names[2:])

    # # View the first 5 rows of the data
    # print(iris_df.head())

    # Print the unique labels of the dataset
    # print('\n' + 'The unique labels in this data are ' + str(np.unique(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))

    sc = StandardScaler()

    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # print('After standardizing our features, the first 5 rows of our data now look like this:\n')
    # print(pd.DataFrame(X_train_std, columns=iris_df.columns).head())
    return X_train_std, X_test_std, y_train, y_test


class Classification:
    def __init__(self):
        x_train, x_test, y_train, y_test = pre_processing()
        self.x_train_std = x_train
        self.x_test_std = x_test
        self.y_train = y_train
        self.y_test = y_test
        pass


# if __name__ == "__main__":
    # X_train_std, X_test_std, y_train, y_test = pre_processing()
    # classification = Classification(X_train_std, X_test_std, y_train, y_test)
    # cProfile.run('classification.knn_predict(5)')
    # profile.run('print (classification.knn_predict(5)); print')
