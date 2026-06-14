from pathlib import Path

from huggingface_hub import HfApi, create_repo

api = HfApi()
user = api.whoami()["name"]
dataset_repo = f"{user}/travel-wellness-data"
model_repo = f"{user}/travel-wellness-model"

create_repo(dataset_repo, repo_type="dataset", exist_ok=True)
create_repo(model_repo, repo_type="model", exist_ok=True)

for file_path in [Path("data/travel_wellness.csv"), *Path("data/processed").glob("*.csv")]:
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=file_path.name,
        repo_id=dataset_repo,
        repo_type="dataset",
    )

api.upload_file(
    path_or_fileobj="deployment/travel_wellness_model.joblib",
    path_in_repo="travel_wellness_model.joblib",
    repo_id=model_repo,
    repo_type="model",
)

print(f"dataset=https://huggingface.co/datasets/{dataset_repo}")
print(f"model=https://huggingface.co/{model_repo}")
