import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

with open("models/logistic_regression.pkl", 'rb') as f:
    logistic_regr = pickle.load(f)

with open("models/tree.pkl", 'rb') as f:
    tree = pickle.load(f)

with open("models/svm.pkl", 'rb') as f:
    svm = pickle.load(f)


df_test = pd.read_csv("sets/test_depression.csv")
df_train = pd.read_csv("sets/train_depression.csv")
df_valid = pd.read_csv("sets/valid_depression.csv")

x_train = df_train.drop(columns=["id", "Depression"])
y_train = df_train["Depression"]

x_valid = df_valid.drop(columns=["id", "Depression"])
y_valid = df_valid["Depression"]

x_test = df_test.drop(columns=["id", "Depression"])
y_test = df_test["Depression"]

y_pred_train_logistic_reg = logistic_regr.predict(x_train)
y_pred_valid_logistic_reg = logistic_regr.predict(x_valid)
y_pred_test_logistic_reg = logistic_regr.predict(x_test)

y_pred_train_tree = tree.predict(x_train)
y_pred_valid_tree = tree.predict(x_valid)
y_pred_test_tree = tree.predict(x_test)

y_pred_train_svm = svm.predict(x_train)
y_pred_valid_svm = svm.predict(x_valid)
y_pred_test_svm = svm.predict(x_test)


print("Logistic Regression - TRAIN")
print(accuracy_score(y_train, y_pred_train_logistic_reg))

print("Logistic Regression - VALIDATION")
print(accuracy_score(y_valid, y_pred_valid_logistic_reg))

print("Logistic Regression - TEST")
print(accuracy_score(y_test, y_pred_test_logistic_reg))

print("Decision tree - TRAIN")
print(accuracy_score(y_train, y_pred_train_tree))

print("Decision tree - VALIDATION")
print(accuracy_score(y_valid, y_pred_valid_tree))

print("Decision tree - TEST")
print(accuracy_score(y_test, y_pred_test_tree))

print("SVM - TRAIN")
print(accuracy_score(y_train, y_pred_train_svm))

print("SVM - VALIDATION")
print(accuracy_score(y_valid, y_pred_valid_svm))

print("SVM - TEST")
print(accuracy_score(y_test, y_pred_test_svm))