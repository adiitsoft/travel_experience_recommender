import os
import joblib
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


# -------------------------
# Load data
# -------------------------

project_root = Path(__file__).resolve().parents[1]
processed_dir = project_root / "data" / "processed"
model_output_dir = project_root / "deployment"
model_output_dir.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (project_root / "mlruns").as_uri()))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "travel-wellness-lead-scoring"))

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = os.getenv("TRAVEL_WELLNESS_XTRAIN_PATH", str(processed_dir / "Xtrain.csv"))
Xtest_path = os.getenv("TRAVEL_WELLNESS_XTEST_PATH", str(processed_dir / "Xtest.csv"))
ytrain_path = os.getenv("TRAVEL_WELLNESS_YTRAIN_PATH", str(processed_dir / "ytrain.csv"))
ytest_path = os.getenv("TRAVEL_WELLNESS_YTEST_PATH", str(processed_dir / "ytest.csv"))

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Separate feature types
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
numeric_features = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']


# Compute class imbalance ratio to use for XGBoost's scale_pos_weight
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print("scale_pos_weight (class 0 / class 1):", scale_pos_weight)

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)


# -------------------------
# Models + param grids
# -------------------------
models = {
    "xgboost": {
        "estimator": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ),
        "param_grid": {
            'xgbclassifier__n_estimators': [75],
            'xgbclassifier__max_depth': [2, 3],
            'xgbclassifier__learning_rate': [0.1],
        }
    },
    "random_forest": {
        # Use 'balanced' to let RandomForest internally weigh classes inversely proportional to class frequencies
        "estimator": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        "param_grid": {
            'randomforestclassifier__n_estimators': [100],
            'randomforestclassifier__max_depth': [None, 10],
            'randomforestclassifier__max_features': ['sqrt'],
        }
    }
}

# best overall scores variables declaration
best_overall_model = None
best_overall_model_name = None
best_overall_recall = -1.0

# -------------------------
# Loop over models, grid-search, MLflow logging
# -------------------------
for model_name, model_info in models.items():
    print(f"\n=== Running model: {model_name} ===")
    estimator = model_info['estimator']
    param_grid = model_info['param_grid']

    # Build pipeline (preprocessor + estimator)
    pipeline = make_pipeline(preprocessor, estimator)

    # Outer MLflow run for this model
    with mlflow.start_run(run_name=model_name):
        # Grid search (5-fold CV)
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="recall")
        grid_search.fit(Xtrain, ytrain)

        # Log each hyperparameter combination as a nested run with its mean/std recall
        results = grid_search.cv_results_
        for i in range(len(results['params'])):
            param_set = results['params'][i]
            mean_score = results['mean_test_score'][i]
            std_score = results['std_test_score'][i]

            with mlflow.start_run(nested=True, run_name=f"{model_name}_params_{i}"):
                mlflow.log_params(param_set)
                mlflow.log_metric("mean_test_recall", float(mean_score))
                mlflow.log_metric("std_test_recall", float(std_score))

        # Log best params in parent run
        mlflow.log_params(grid_search.best_params_)

        # Best estimator and evaluation
        best_model = grid_search.best_estimator_


        # Direct class predictions (no probability thresholding)
        y_pred_train = best_model.predict(Xtrain)
        y_pred_test = best_model.predict(Xtest)

        # Classification reports
        train_report = classification_report(ytrain, y_pred_train, output_dict=True)
        test_report = classification_report(ytest, y_pred_test, output_dict=True)

        test_recall = test_report['1']['recall']

        # Track best model across algorithms
        if test_recall > best_overall_recall:
            best_overall_recall = test_recall
            best_overall_model = best_model
            best_overall_model_name = model_name


        # Log standard metrics
        mlflow.log_metrics({
                   f"{model_name}_train_accuracy": train_report['accuracy'],
                   f"{model_name}_train_precision": train_report['1']['precision'],
                   f"{model_name}_train_recall": train_report['1']['recall'],
                   f"{model_name}_train_f1": train_report['1']['f1-score'],
                   f"{model_name}_test_accuracy": test_report['accuracy'],
                   f"{model_name}_test_precision": test_report['1']['precision'],
                   f"{model_name}_test_recall": test_report['1']['recall'],
                   f"{model_name}_test_f1": test_report['1']['f1-score']})
        # Save and log model artifact
        model_artifact_path = f"{model_name}_best_model.joblib"
        joblib.dump(best_model, model_artifact_path)
        mlflow.log_artifact(model_artifact_path,artifact_path="model")


        print(f"Completed model: {model_name}. Best params: {grid_search.best_params_}")
        print(f"Test recall for {model_name}: {test_report['1']['recall']:.4f}")

print("\n================ FINAL MODEL SELECTION ================")
print(f"Selected Model      : {best_overall_model_name}")
print(f"Best Test Recall    : {best_overall_recall:.4f}")
print("=======================================================")
# Save the model locally for the Streamlit app
model_filename = os.getenv("TRAVEL_WELLNESS_MODEL_FILENAME", "travel_wellness_model.joblib")
model_path = model_output_dir / model_filename
joblib.dump(best_overall_model, model_path)

print(f"Best Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = os.getenv("TRAVEL_WELLNESS_MODEL_REPO_ID", "your-huggingface-username/travel-wellness-model")
repo_type = "model"

if os.getenv("HF_TOKEN"):
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repository '{repo_id}' not found. Creating it...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repository '{repo_id}' created.")

    api.upload_file(
          path_or_fileobj=str(model_path),
          path_in_repo=model_filename,
          repo_id=repo_id,
          repo_type=repo_type,
    )
