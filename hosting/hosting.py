from huggingface_hub import HfApi
import os
from pathlib import Path

api = HfApi(token=os.getenv("HF_TOKEN"))
project_root = Path(__file__).resolve().parents[1]
api.upload_folder(
    folder_path=str(project_root / "deployment"),
    repo_id=os.getenv("TRAVEL_WELLNESS_SPACE_REPO_ID", "your-huggingface-username/travel-wellness-app"),
    repo_type="space",
    path_in_repo="",
)
