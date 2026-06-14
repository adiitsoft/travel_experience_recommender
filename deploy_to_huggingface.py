from pathlib import Path

from huggingface_hub import HfApi, create_repo

api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/travel-wellness-lead-scoring"
deployment_dir = Path("deployment").resolve()

create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(deployment_dir),
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["__pycache__/*", "*.pyc"],
)

print(f"https://huggingface.co/spaces/{repo_id}")
