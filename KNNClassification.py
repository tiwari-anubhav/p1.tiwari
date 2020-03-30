from sklearn.neighbors import KNeighborsClassifier
from Classification import Classification


class KNN:
    def __init__(self):
        cl = Classification()
        self.x_train_std = cl.x_train_std
        self.x_test_std = cl.x_test_std
        self.y_train = cl.y_train
        self.y_test = cl.y_test
        pass

    def knn_predict(self, k):
        knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
        knn.fit(self.x_train_std, self.y_train)

        # print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(self.x_train_std, self.y_train)))
        # print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(self.x_test_std, self.y_test)))
