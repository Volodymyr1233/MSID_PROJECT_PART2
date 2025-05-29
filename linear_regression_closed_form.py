import pandas as pd
from pipeline import call_preprocess
from sklearn.metrics import mean_absolute_error, mean_squared_error
from implemented_models.LinearRegrClosedForm import LinearRegrClosedForm
import numpy as np
from utils import make_table

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

linear_regr_closed = LinearRegrClosedForm()
linear_regr_closed.fit(x_train_prep.toarray(), y_train)


y_pred_train = linear_regr_closed.predict(x_train_prep.toarray())
y_pred_valid = linear_regr_closed.predict(x_valid_prep.toarray())
y_pred_test = linear_regr_closed.predict(x_test_prep.toarray())


mae_train = round(mean_absolute_error(y_train, y_pred_train), 4)
rmse_train = round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4)

mae_valid = round(mean_absolute_error(y_valid, y_pred_valid), 4)
rmse_valid = round(np.sqrt(mean_squared_error(y_valid, y_pred_valid)), 4)

mae_test = round(mean_absolute_error(y_test, y_pred_test), 4)
rmse_test = round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4)

make_table(
    "Regresja liniowa (Closed-form)",
    "MAE", "RMSE",
    mae_train, rmse_train,
    mae_valid, rmse_valid,
    mae_test, rmse_test
)
