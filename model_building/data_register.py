
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path


repo_id = os.getenv("TRAVEL_WELLNESS_DATASET_REPO_ID", "your-huggingface-username/travel-wellness-data")
repo_type = "dataset"
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repository '{repo_id}' created.")

api.upload_folder(
    folder_path=str(data_dir),
    repo_id=repo_id,
    repo_type=repo_type,
)
