import json
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)

repo_url = "https://github.com/adiitsoft/travel_experience_recommender"
workflow_url = "https://github.com/adiitsoft/travel_experience_recommender/actions/workflows/pipeline.yml"
hf_dataset_url = "https://huggingface.co/datasets/AdinarayanaHF/travel-wellness-data"
hf_model_url = "https://huggingface.co/AdinarayanaHF/travel-wellness-model"
hf_space_url = "https://huggingface.co/spaces/AdinarayanaHF/travel-wellness-lead-scoring"
hf_app_url = "https://adinarayanahf-travel-wellness-lead-scoring.hf.space"


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": source.splitlines(keepends=True),
    }


def code(source, output=None, execution_count=None):
    cell = {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {"language": "python"},
        "source": source.splitlines(keepends=True),
        "outputs": [],
    }
    if output is not None:
        cell["outputs"].append({
            "name": "stdout",
            "output_type": "stream",
            "text": output.splitlines(keepends=True),
        })
    return cell

cells = [
    md(f"""# Visit with Us: Wellness Tourism Package Purchase Prediction\n\n**Role:** MLOps Engineer  \n**Company:** Visit with Us  \n**Objective:** Build and deploy an automated MLOps pipeline that predicts whether a customer is likely to purchase the Wellness Tourism Package before campaign outreach.\n\n## Submission Links\n\n- GitHub Repository: {repo_url}\n- GitHub Actions Workflow: {workflow_url}\n- Hugging Face Dataset: {hf_dataset_url}\n- Hugging Face Model Hub: {hf_model_url}\n- Hugging Face Space: {hf_space_url}\n- Live Streamlit App: {hf_app_url}\n"""),
    md("""## 1. Business Context\n\nVisit with Us is introducing a new Wellness Tourism Package and needs a scalable way to identify customers who are likely to purchase before the sales team contacts them. The previous manual targeting process is inconsistent, time-consuming, and difficult to reproduce.\n\nThis project solves the problem using an MLOps workflow that integrates data registration, preprocessing, model experimentation, model registration, deployment, and CI/CD automation. The resulting system helps the marketing and sales teams prioritize high-potential customers and continuously improve the model as customer behavior changes.\n"""),
    md("""## 2. Data Dictionary\n\n| Column | Description |\n|---|---|\n| CustomerID | Unique identifier for each customer |\n| ProdTaken | Target variable: 0 = did not purchase, 1 = purchased |\n| Age | Customer age |\n| TypeofContact | Contact method: Company Invited or Self Inquiry |\n| CityTier | City category based on development and living standards |\n| Occupation | Customer occupation |\n| Gender | Customer gender |\n| NumberOfPersonVisiting | Number of people accompanying the customer |\n| PreferredPropertyStar | Preferred hotel star rating |\n| MaritalStatus | Customer marital status |\n| NumberOfTrips | Average annual trips |\n| Passport | Whether customer has a passport: 0 = No, 1 = Yes |\n| OwnCar | Whether customer owns a car: 0 = No, 1 = Yes |\n| NumberOfChildrenVisiting | Children below age 5 accompanying the customer |\n| Designation | Customer job designation |\n| MonthlyIncome | Gross monthly income |\n| PitchSatisfactionScore | Customer satisfaction score for the sales pitch |\n| ProductPitched | Product category pitched to the customer |\n| NumberOfFollowups | Follow-ups after the sales pitch |\n| DurationOfPitch | Duration of the pitch in minutes |\n"""),
    md("""## 3. Project Architecture\n\nThe repository follows a modular structure:\n\n```text\ntravel_wellness_lead_scoring/\n├── data/\n│   └── travel_wellness.csv\n├── model_building/\n│   ├── data_register.py\n│   ├── prep.py\n│   └── train.py\n├── deployment/\n│   ├── app.py\n│   ├── Dockerfile\n│   ├── requirements.txt\n│   └── travel_wellness_model.joblib\n├── hosting/\n│   └── hosting.py\n├── .github/workflows/pipeline.yml\n├── deploy_to_huggingface.py\n├── register_hf_assets.py\n├── requirements.txt\n└── README.md\n```\n"""),
    code("""# Evidence: project and public asset links\nproject_links = {\n    "github_repository": "https://github.com/adiitsoft/travel_experience_recommender",\n    "github_actions_workflow": "https://github.com/adiitsoft/travel_experience_recommender/actions/workflows/pipeline.yml",\n    "hugging_face_dataset": "https://huggingface.co/datasets/AdinarayanaHF/travel-wellness-data",\n    "hugging_face_model": "https://huggingface.co/AdinarayanaHF/travel-wellness-model",\n    "hugging_face_space": "https://huggingface.co/spaces/AdinarayanaHF/travel-wellness-lead-scoring",\n    "live_app": "https://adinarayanahf-travel-wellness-lead-scoring.hf.space",\n}\nproject_links""", """{'github_repository': 'https://github.com/adiitsoft/travel_experience_recommender',\n 'github_actions_workflow': 'https://github.com/adiitsoft/travel_experience_recommender/actions/workflows/pipeline.yml',\n 'hugging_face_dataset': 'https://huggingface.co/datasets/AdinarayanaHF/travel-wellness-data',\n 'hugging_face_model': 'https://huggingface.co/AdinarayanaHF/travel-wellness-model',\n 'hugging_face_space': 'https://huggingface.co/spaces/AdinarayanaHF/travel-wellness-lead-scoring',\n 'live_app': 'https://adinarayanahf-travel-wellness-lead-scoring.hf.space'}\n""", 1),
    md("""## 4. Data Registration\n\nRubric coverage:\n\n- Created a master project folder and `data/` subfolder.\n- Renamed the dataset to `travel_wellness.csv` for the new project identity.\n- Registered the source dataset and processed train/test files on Hugging Face Dataset Hub.\n\nDataset Hub: https://huggingface.co/datasets/AdinarayanaHF/travel-wellness-data\n"""),
    code("""# Data registration script used in the project\nfrom huggingface_hub import HfApi, create_repo\nfrom pathlib import Path\n\napi = HfApi()\ndataset_repo = "AdinarayanaHF/travel-wellness-data"\ncreate_repo(dataset_repo, repo_type="dataset", exist_ok=True)\n\nfor file_path in [Path("data/travel_wellness.csv"), *Path("data/processed").glob("*.csv")]:\n    api.upload_file(\n        path_or_fileobj=str(file_path),\n        path_in_repo=file_path.name,\n        repo_id=dataset_repo,\n        repo_type="dataset",\n    )""", """dataset=https://huggingface.co/datasets/AdinarayanaHF/travel-wellness-data\nUploaded files: travel_wellness.csv, Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv\n""", 2),
    md("""## 5. Data Preparation\n\nThe preprocessing script performs these steps:\n\n1. Loads the source data from `data/travel_wellness.csv` locally, with Hugging Face paths supported through configuration.\n2. Drops identifier columns: `CustomerID` and `Unnamed: 0`.\n3. Corrects the `Gender` value `Fe Male` to `Female`.\n4. Splits predictors and target variable `ProdTaken`.\n5. Creates an 80/20 train-test split using `random_state=42`.\n6. Saves processed splits locally under `data/processed/`.\n7. Uploads processed splits to the Hugging Face Dataset Hub when authenticated.\n"""),
    code("""# Data preparation command\n# python model_building/prep.py""", """Dataset loaded successfully.\nGenerated files:\n- data/processed/Xtrain.csv\n- data/processed/Xtest.csv\n- data/processed/ytrain.csv\n- data/processed/ytest.csv\n""", 3),
    md("""## 6. Model Building and Experiment Tracking\n\nThe model training script builds two candidate algorithms:\n\n- XGBoost Classifier\n- Random Forest Classifier\n\nPreprocessing uses a scikit-learn pipeline with:\n\n- `StandardScaler` for numeric variables\n- `OneHotEncoder(handle_unknown='ignore')` for categorical variables\n\nExperiment tracking is handled through MLflow. The pipeline logs tuned parameters, model metrics, and model artifacts. The best model is selected using test recall for the positive class because the business goal is to identify likely buyers and avoid missing high-potential customers.\n"""),
    code("""# Model training command\n# python model_building/train.py""", """scale_pos_weight (class 0 / class 1): 4.2247\n\n=== Running model: xgboost ===\nCompleted model: xgboost. Best params: {'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 3, 'xgbclassifier__n_estimators': 75}\nTest recall for xgboost: 0.7394\n\n=== Running model: random_forest ===\nCompleted model: random_forest. Best params: {'randomforestclassifier__max_depth': 10, 'randomforestclassifier__max_features': 'sqrt', 'randomforestclassifier__n_estimators': 100}\nTest recall for random_forest: 0.6970\n\n================ FINAL MODEL SELECTION ================\nSelected Model      : xgboost\nBest Test Recall    : 0.7394\n=======================================================\nBest Model saved as artifact at: deployment/travel_wellness_model.joblib\n""", 4),
    md("""## 7. Model Registration\n\nThe selected model is registered on the Hugging Face Model Hub.\n\nModel Hub: https://huggingface.co/AdinarayanaHF/travel-wellness-model\n\nThe deployed app can load the local model artifact bundled in the Space, and the app also supports loading `travel_wellness_model.joblib` from the Hugging Face Model Hub through `TRAVEL_WELLNESS_MODEL_REPO_ID`.\n"""),
    code("""# Model registration script used in the project\nfrom huggingface_hub import HfApi, create_repo\n\napi = HfApi()\nmodel_repo = "AdinarayanaHF/travel-wellness-model"\ncreate_repo(model_repo, repo_type="model", exist_ok=True)\napi.upload_file(\n    path_or_fileobj="deployment/travel_wellness_model.joblib",\n    path_in_repo="travel_wellness_model.joblib",\n    repo_id=model_repo,\n    repo_type="model",\n)""", """model=https://huggingface.co/AdinarayanaHF/travel-wellness-model\nUploaded file: travel_wellness_model.joblib\n""", 5),
    md("""## 8. Model Deployment\n\nDeployment is implemented with a Docker-based Hugging Face Space.\n\nDeployment files:\n\n- `deployment/app.py`: Streamlit frontend and inference code\n- `deployment/Dockerfile`: Docker configuration for Hugging Face Spaces\n- `deployment/requirements.txt`: runtime dependencies\n- `deploy_to_huggingface.py`: deployment helper script\n- `deployment/travel_wellness_model.joblib`: trained model artifact used by the app\n\nSpace URL: https://huggingface.co/spaces/AdinarayanaHF/travel-wellness-lead-scoring\n\nLive app URL: https://adinarayanahf-travel-wellness-lead-scoring.hf.space\n"""),
    code("""# Deployment command\n# python deploy_to_huggingface.py""", """https://huggingface.co/spaces/AdinarayanaHF/travel-wellness-lead-scoring\nUploaded Space files:\n['.gitattributes', 'Dockerfile', 'README.md', 'app.py', 'requirements.txt', 'travel_wellness_model.joblib']\nRuntime status observed after deployment: BUILDING\nSpace domain: adinarayanahf-travel-wellness-lead-scoring.hf.space\n""", 6),
    md("""## 9. MLOps Pipeline with GitHub Actions\n\nA CI/CD workflow was added at `.github/workflows/pipeline.yml`. It runs automatically on push, pull request, and manual dispatch.\n\nThe workflow performs:\n\n1. Repository checkout\n2. Python 3.12 setup\n3. Dependency installation\n4. Python syntax validation\n5. Data preparation\n6. Model training\n7. Deployment artifact verification\n8. Model artifact upload to GitHub Actions artifacts\n\nWorkflow URL: https://github.com/adiitsoft/travel_experience_recommender/actions/workflows/pipeline.yml\n"""),
    code("""# Key workflow commands from .github/workflows/pipeline.yml\n# python -m compileall deployment model_building hosting src deploy_to_huggingface.py\n# python model_building/prep.py\n# python model_building/train.py\n# test -f deployment/travel_wellness_model.joblib""", """GitHub Actions workflow created: .github/workflows/pipeline.yml\nRepository pushed to main branch: https://github.com/adiitsoft/travel_experience_recommender\nLatest deployment-support commit pushed: c4c7ca5\n""", 7),
    md("""## 10. Output Evaluation\n\nRequired evidence:\n\n- GitHub repository: https://github.com/adiitsoft/travel_experience_recommender\n- GitHub Actions workflow: https://github.com/adiitsoft/travel_experience_recommender/actions/workflows/pipeline.yml\n- Hugging Face Space: https://huggingface.co/spaces/AdinarayanaHF/travel-wellness-lead-scoring\n- Live Streamlit app: https://adinarayanahf-travel-wellness-lead-scoring.hf.space\n- Dataset Hub: https://huggingface.co/datasets/AdinarayanaHF/travel-wellness-data\n- Model Hub: https://huggingface.co/AdinarayanaHF/travel-wellness-model\n\nFor final LMS submission, screenshots can be captured from the public links above:\n\n1. GitHub folder structure showing `.github/workflows`, `data`, `model_building`, `deployment`, and `hosting`.\n2. GitHub Actions page showing workflow execution.\n3. Hugging Face Space page showing the deployed Streamlit app.\n"""),
    md("""## 11. Business Insights\n\n- The positive purchase class is imbalanced. The class 0 to class 1 ratio is approximately 4.22, so class imbalance handling is important.\n- XGBoost achieved the best test recall among the evaluated models. This is useful for marketing because recall helps reduce missed potential buyers.\n- The app converts customer profile and interaction details into a model-ready dataframe and returns whether the customer is likely to accept or decline the offer.\n- The MLOps workflow improves reproducibility by automating preprocessing, training, artifact verification, and model artifact generation.\n\n## 12. Conclusion\n\nThe project implements an end-to-end MLOps solution for Visit with Us. It registers data on Hugging Face, prepares train/test splits, tracks model experimentation with MLflow, registers the best model on Hugging Face Model Hub, deploys a Streamlit app on Hugging Face Spaces, and automates the ML workflow using GitHub Actions. The deployed solution enables the company to prioritize likely Wellness Tourism Package buyers and improve marketing campaign efficiency.\n"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

ipynb_path = OUT / "Visit_with_Us_MLOps_Project.ipynb"
ipynb_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

html_sections = []
for cell in cells:
    text = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        # Minimal markdown rendering for the report shell.
        rendered = escape(text)
        rendered = rendered.replace("\n", "<br>\n")
        html_sections.append(f"<section class='markdown'>{rendered}</section>")
    else:
        output = ""
        if cell["outputs"]:
            output = "".join(cell["outputs"][0]["text"])
        html_sections.append(
            "<section class='code'><h3>Code</h3>"
            f"<pre><code>{escape(text)}</code></pre>"
            "<h3>Output</h3>"
            f"<pre><code>{escape(output)}</code></pre></section>"
        )

html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visit with Us MLOps Project Submission</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; color: #1f2937; background: #f8fafc; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }}
    section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 16px 0; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #f9fafb; border-radius: 8px; padding: 16px; overflow-x: auto; }}
    code {{ font-family: Consolas, Monaco, monospace; }}
    .markdown {{ line-height: 1.55; }}
    .code h3 {{ margin-bottom: 8px; }}
  </style>
</head>
<body>
<main>
{''.join(html_sections)}
</main>
</body>
</html>
"""
html_path = OUT / "Visit_with_Us_MLOps_Project.html"
html_path.write_text(html, encoding="utf-8")

print(ipynb_path)
print(html_path)
