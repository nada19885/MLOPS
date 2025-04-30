import os

import hydra
import joblib
import pandas as pd
from hydra.utils import instantiate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="config", config_name="titanic")
def main(cfg):

    os.chdir(HydraConfig.get().runtime.cwd)

    train_df = pd.read_csv(cfg.data.train_file)
    test_df = pd.read_csv(cfg.data.test_file)

    X = train_df[cfg.data.features]
    y = train_df[cfg.data.target]
    X_test = test_df[cfg.data.features]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=cfg.preprocessing.numeric_transformer.imputer_strategy
                ),
            ),
            ("scaler", instantiate(cfg.preprocessing.numeric_transformer.scaler)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=cfg.preprocessing.categorical_transformer.imputer_strategy
                ),
            ),
            (
                "onehot",
                OneHotEncoder(**cfg.preprocessing.categorical_transformer.onehot),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(cfg.preprocessing.numeric_features)),
            (
                "cat",
                categorical_transformer,
                list(cfg.preprocessing.categorical_features),
            ),
        ]
    )

    models = {name: instantiate(config) for name, config in cfg.models.items()}

    os.makedirs(cfg.output.model_path, exist_ok=True)
    os.makedirs(cfg.output.submission_path, exist_ok=True)

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"{model_name} Validation Accuracy: {accuracy:.4f}")

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cfg.evaluation.cross_validation.cv,
            scoring=cfg.evaluation.cross_validation.scoring,
        )
        print(
            f"{model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        model_file = cfg.output.model_file.format(model_name=model_name)
        joblib.dump(pipeline, model_file)
        print(f"Saved {model_name} pipeline to {model_file}")

        test_predictions = pipeline.predict(X_test)
        submission = pd.DataFrame(
            {"PassengerId": test_df["PassengerId"], "Survived": test_predictions}
        )
        submission_file = cfg.output.submission_file.format(model_name=model_name)
        submission.to_csv(submission_file, index=False)
        print(f"Created submission file: {submission_file}")

    print("Training pipeline completed.")


if __name__ == "__main__":
    main()
