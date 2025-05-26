import pandas as pd
import numpy as np
from pipeline import call_preprocess
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("sets/train_depression.csv")
valid_df = pd.read_csv("sets/valid_depression.csv")
test_df = pd.read_csv("sets/test_depression.csv")
preprocessor = call_preprocess(["CGPA", "Depression"])

target_column = "CGPA"
drop_columns = ["id", "Depression", "CGPA"]

x_train = train_df.drop(columns=drop_columns)
y_train = train_df[target_column].astype(float)

x_valid = valid_df.drop(columns=drop_columns)
y_valid = valid_df[target_column].astype(float)

x_test = test_df.drop(columns=drop_columns)
y_test = test_df[target_column].astype(float)

x_train_prep = preprocessor.fit_transform(x_train)
x_valid_prep = preprocessor.transform(x_valid)
x_test_prep = preprocessor.transform(x_test)


def closed_form_linear_regression(x, y):
    X_bias = np.hstack([np.ones((x.shape[0], 1)), x])
    return np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y.values

def predict(x, w):
    x_bias = np.hstack([np.ones((x.shape[0], 1)), x])
    return x_bias @ w


w = closed_form_linear_regression(x_train_prep.toarray(), y_train)

y_pred_train = predict(x_train_prep.toarray(), w)
y_pred_valid = predict(x_valid_prep.toarray(), w)
y_pred_test = predict(x_test_prep.toarray(), w)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_valid = mean_squared_error(y_valid, y_pred_valid)
mse_test = mean_squared_error(y_test, y_pred_test)

print("Wyniki własnej regresji liniowej (Closed-form)")
print(f"{'Zbiór':<12}{'MSE':>10}")
print(f"{'Train':<12}{mse_train:>10.4f}")
print(f"{'Validation':<12}{mse_valid:>10.4f}")
print(f"{'Test':<12}{mse_test:>10.4f}")