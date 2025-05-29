import pandas as pd
from sklearn.metrics import accuracy_score
from pipeline import call_preprocess
from implemented_models.LogisticRegrGradient import LogisticRegrGradient
from utils import make_table

train_df = pd.read_csv("sets/train_depression.csv")
valid_df = pd.read_csv("sets/valid_depression.csv")
test_df = pd.read_csv("sets/test_depression.csv")

target_column = "Depression"
drop_columns = ["id", "CGPA", "Depression"]

x_train = train_df.drop(columns=drop_columns)
y_train = train_df[target_column].astype(float).values
x_valid = valid_df.drop(columns=drop_columns)
y_valid = valid_df[target_column].astype(float).values
x_test = test_df.drop(columns=drop_columns)
y_test = test_df[target_column].astype(float).values

preprocessor = call_preprocess(["CGPA", "Depression"])
x_train_prep = preprocessor.fit_transform(x_train).toarray()
x_valid_prep = preprocessor.transform(x_valid).toarray()
x_test_prep = preprocessor.transform(x_test).toarray()

logistic_regr_gradient = LogisticRegrGradient(lr=0.01, epochs=500, batch_size=16)

logistic_regr_gradient.fit(x_train_prep, y_train)


y_pred_train, y_proba_train = logistic_regr_gradient.predict(x_train_prep)
y_pred_valid, y_proba_valid = logistic_regr_gradient.predict(x_valid_prep)
y_pred_test, y_proba_test = logistic_regr_gradient.predict(x_test_prep)

logreg_train_acc = round(accuracy_score(y_train, y_pred_train), 4)
logreg_train_loss = round(logistic_regr_gradient.compute_cross_entropy(y_train, y_proba_train), 4)

logreg_valid_acc = round(accuracy_score(y_valid, y_pred_valid), 4)
logreg_valid_loss = round(logistic_regr_gradient.compute_cross_entropy(y_valid, y_proba_valid), 4)

logreg_test_acc = round(accuracy_score(y_test, y_pred_test), 4)
logreg_test_loss = round(logistic_regr_gradient.compute_cross_entropy(y_test, y_proba_test), 4)


make_table(
    "Regresja logistyczna (gradient)",
    "Accuracy", "CE",
    logreg_train_acc, logreg_train_loss,
    logreg_valid_acc, logreg_valid_loss,
    logreg_test_acc, logreg_test_loss
)