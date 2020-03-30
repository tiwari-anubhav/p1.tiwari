from sklearn.svm import SVC

from Classification import Classification


class SVM:
    def __init__(self):
        cl = Classification()
        self.x_train_std = cl.x_train_std
        self.x_test_std = cl.x_test_std
        self.y_train = cl.y_train
        self.y_test = cl.y_test
        pass

    def svm_predict(self, gamma):
        svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=1.0)
        svm.fit(self.x_train_std, self.y_train)

        # print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(self.x_train_std, self.y_train)))

        # print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(self.x_test_std, self.y_test)))
