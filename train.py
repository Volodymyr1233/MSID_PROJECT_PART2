import pandas as pd
import numpy as np
from pipeline import call_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle


preprocessor = call_preprocess(["CGPA", "Depression"])
df = pd.read_csv("depr_dataset.csv")
df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)

train_valid_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.25, random_state=42)

train_df.to_csv("sets/train_depression.csv", index=False)
valid_df.to_csv("sets/valid_depression.csv", index=False)
test_df.to_csv("sets/test_depression.csv", index=False)

x_train = train_df.drop(columns=["id", "Depression"])
y_train = train_df["Depression"]

logistic_regression_pipeline = Pipeline([('preprocess', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
tree_pipeline = Pipeline([('preprocess', preprocessor), ('classifier', DecisionTreeClassifier(max_depth=3))])
svm_pipeline = Pipeline([('preprocess', preprocessor), ('classifier', SVC(probability=True))])


logistic_regression_pipeline.fit(x_train, y_train)
tree_pipeline.fit(x_train, y_train)
svm_pipeline.fit(x_train, y_train)

with open("models/logistic_regression.pkl", 'wb') as f:
    pickle.dump(logistic_regression_pipeline, f)

with open("models/tree.pkl", 'wb') as f:
    pickle.dump(tree_pipeline, f)

with open("models/svm.pkl", 'wb') as f:
    pickle.dump(svm_pipeline, f)