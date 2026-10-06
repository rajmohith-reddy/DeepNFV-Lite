"""CART-style baseline: a Gini decision tree on the flattened 900-value packet vectors."""
from sklearn.tree import DecisionTreeClassifier


def train_cart(X_train, y_train, seed=42):
    clf = DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=seed)
    clf.fit(X_train.reshape(len(X_train), -1), y_train)
    return clf


def predict_cart(clf, X):
    return clf.predict(X.reshape(len(X), -1))
