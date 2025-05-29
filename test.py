import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, log_loss
from utils import make_table

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

y_proba_train_logistic_reg = logistic_regr.predict_proba(x_train)
y_proba_valid_logistic_reg = logistic_regr.predict_proba(x_valid)
y_proba_test_logistic_reg = logistic_regr.predict_proba(x_test)


y_pred_train_tree = tree.predict(x_train)
y_pred_valid_tree = tree.predict(x_valid)
y_pred_test_tree = tree.predict(x_test)

y_proba_train_tree = tree.predict_proba(x_train)
y_proba_valid_tree = tree.predict_proba(x_valid)
y_proba_test_tree = tree.predict_proba(x_test)


y_pred_train_svm = svm.predict(x_train)
y_pred_valid_svm = svm.predict(x_valid)
y_pred_test_svm = svm.predict(x_test)

y_proba_train_svm = svm.predict_proba(x_train)
y_proba_valid_svm = svm.predict_proba(x_valid)
y_proba_test_svm = svm.predict_proba(x_test)

log_regr_train_acc =  round(accuracy_score(y_train, y_pred_train_logistic_reg), 4)
log_regr_train_loss = round(log_loss(y_train, y_proba_train_logistic_reg), 4)

log_regr_valid_acc = round(accuracy_score(y_valid, y_pred_valid_logistic_reg), 4)
log_regr_valid_loss = round(log_loss(y_valid, y_proba_valid_logistic_reg), 4)

log_regr_test_acc = round(accuracy_score(y_test, y_pred_test_logistic_reg), 4)
log_regr_test_loss = round(log_loss(y_test, y_proba_test_logistic_reg), 4)

make_table("Logistic Regression", "Accuracy", "CE", log_regr_train_acc, log_regr_train_loss, log_regr_valid_acc, log_regr_valid_loss, log_regr_test_acc, log_regr_test_loss)

tree_train_acc = round(accuracy_score(y_train, y_pred_train_tree), 4)
tree_train_loss = round(log_loss(y_train, y_proba_train_tree), 4)

tree_valid_acc = round(accuracy_score(y_valid, y_pred_valid_tree), 4)
tree_valid_loss = round(log_loss(y_valid, y_proba_valid_tree), 4)

tree_test_acc = round(accuracy_score(y_test, y_pred_test_tree), 4)
tree_test_loss = round(log_loss(y_test, y_proba_test_tree), 4)

make_table("Decision Tree", "Accuracy", "CE", tree_train_acc, tree_train_loss, tree_valid_acc, tree_valid_loss, tree_test_acc, tree_test_loss)


svm_train_acc = round(accuracy_score(y_train, y_pred_train_svm), 4)
svm_train_loss = round(log_loss(y_train, y_proba_train_svm), 4)

svm_valid_acc = round(accuracy_score(y_valid, y_pred_valid_svm), 4)
svm_valid_loss = round(log_loss(y_valid, y_proba_valid_svm), 4)

svm_test_acc = round(accuracy_score(y_test, y_pred_test_svm), 4)
svm_test_loss = round(log_loss(y_test, y_proba_test_svm), 4)

make_table("SVM", "Accuracy", "CE", svm_train_acc, svm_train_loss, svm_valid_acc, svm_valid_loss, svm_test_acc, svm_test_loss)
