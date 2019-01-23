import logging
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import graphviz

logger = logging.getLogger(__name__)


class Tree:

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=17)

    clf = tree.DecisionTreeClassifier(random_state=17)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))

    dot = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
                               filled=True, rounded=True, special_characters=True)

    graph = graphviz.Source(dot)
    graph.format = 'png'
    graph.render('iris', view=True)
