# Travel Wellness Lead Scoring

A Streamlit and scikit-learn project for estimating whether a customer is likely to accept a wellness travel offer. The project includes data preparation, model training with MLflow tracking, and a deployable Streamlit app.

## Project Structure

- `data/` stores the source dataset and generated train/test files.
- `model_building/prep.py` prepares the dataset and writes processed splits.
- `model_building/train.py` trains candidate models, selects the best model, and saves it for the app.
- `deployment/app.py` runs the Streamlit prediction interface.
- `hosting/hosting.py` uploads the deployment folder to your configured Hugging Face Space.

## Configuration

Copy `.env.example` to your local environment settings and replace the placeholder Hugging Face repository names with your own account and repositories.

Required settings for hosted model loading:

- `TRAVEL_WELLNESS_MODEL_REPO_ID`
- `TRAVEL_WELLNESS_MODEL_FILENAME`

Optional settings for publishing data, model artifacts, and the app:

- `HF_TOKEN`
- `TRAVEL_WELLNESS_DATASET_REPO_ID`
- `TRAVEL_WELLNESS_SPACE_REPO_ID`
- `MLFLOW_EXPERIMENT_NAME`

## Local Workflow

Install dependencies:

```powershell
pip install -r requirements.txt
```

Prepare the data:

```powershell
python model_building/prep.py
```

Train the model:

```powershell
python model_building/train.py
```

Run the app:

```powershell
streamlit run deployment/app.py
```

The training script saves the selected model into `deployment/`, and the app loads that local artifact before trying Hugging Face.