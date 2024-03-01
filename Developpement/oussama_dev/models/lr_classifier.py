from sklearn.linear_model import LogisticRegression


def evaluate_model(X_train, X_test, y_train, y_test):
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)
    predictions = logistic_reg.predict(X_test)
    score = logistic_reg.score(X_test, y_test)
    return predictions, score
