from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_columns = ["Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction", "Work/Study Hours", "Financial Stress", "Depression"]

categoric_columns = ["Gender", "City", "Profession", "Sleep Duration", "Dietary Habits", "Degree", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]

transform_num = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler())
])

transform_cat = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


def call_preprocess(removed_element_arr):
    preprocess = ColumnTransformer([
        ("num", transform_num, [x for x in numeric_columns if x not in removed_element_arr]),
        ("cat", transform_cat, categoric_columns)
    ])

    return preprocess