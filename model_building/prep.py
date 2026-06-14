

# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
from pathlib import Path
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


api = HfApi(token=os.getenv("HF_TOKEN"))
project_root = Path(__file__).resolve().parents[1]
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

dataset_repo_id = os.getenv("TRAVEL_WELLNESS_DATASET_REPO_ID", "AdinarayanaHF/travel-wellness-data")
DATASET_PATH = os.getenv("TRAVEL_WELLNESS_DATASET_PATH", str(project_root / "data" / "travel_wellness.csv"))
travel_wellness_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")



# Preprocessing data for Gender column and dropping columns which are not required since they are only identifiers.
travel_wellness_dataset.drop(columns=['CustomerID','Unnamed: 0'], inplace=True)
travel_wellness_dataset['Gender'] = travel_wellness_dataset['Gender'].replace('Fe Male', 'Female')



# Define the target variable for the classification task
target = 'ProdTaken'

# get the list all of numeric and categorical/object columns
categorical_features = list(travel_wellness_dataset.select_dtypes(include=['object']).columns)
numeric_features = list(travel_wellness_dataset.select_dtypes(include=['int64', 'float64']).columns)
# remvoving the target variable from the columns
numeric_features.remove(target)


# Define predictor matrix (X) using selected numeric and categorical features
X = travel_wellness_dataset[numeric_features + categorical_features]

# Define target variable
y = travel_wellness_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv(output_dir / "Xtrain.csv", index=False)
Xtest.to_csv(output_dir / "Xtest.csv", index=False)
ytrain.to_csv(output_dir / "ytrain.csv", index=False)
ytest.to_csv(output_dir / "ytest.csv", index=False)


files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

if os.getenv("HF_TOKEN"):
    for file_name in files:
        api.upload_file(
            path_or_fileobj=str(output_dir / file_name),
            path_in_repo=file_name,
            repo_id=dataset_repo_id,
            repo_type="dataset",
        )
