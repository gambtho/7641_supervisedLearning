from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def train_using_gini(x_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(x_train, y_train)
    return clf_gini


def train_using_entropy(x_train, y_train):
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(x_train, y_train)
    return clf_entropy


def prediction(x_test, clf_object):
    y_pred = clf_object.predict(x_test)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


class Debug:

    def info_print(self, x_test, x_train, y_test, y_train):
        clf_gini = train_using_gini(x_train, y_train)
        clf_entropy = train_using_entropy(x_train, y_train)
        print("Results Using Gini Index:")
        y_pred_gini = prediction(x_test, clf_gini)
        cal_accuracy(y_test, y_pred_gini)
        print("Results Using Entropy:")
        y_pred_entropy = prediction(x_test, clf_entropy)
        cal_accuracy(y_test, y_pred_entropy)
