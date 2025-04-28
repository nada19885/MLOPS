import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_df[features]
y = train_df["Survived"]
X_test = test_df[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ["Age", "SibSp", "Parch", "Fare"]
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_features = ["Pclass", "Sex", "Embarked"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} Validation Accuracy: {accuracy:.4f}")

=    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    joblib.dump(pipeline, f"{model_name}_pipeline.joblib")
    print(f"Saved {model_name} pipeline to {model_name}_pipeline.joblib")

    test_predictions = pipeline.predict(X_test)
    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": test_predictions}
    )
    submission.to_csv(f"submission_{model_name}.csv", index=False)
    print(f"Created submission file: submission_{model_name}.csv")

print("Training pipeline completed.")