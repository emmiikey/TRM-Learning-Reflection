""" Function to fetch files from the GAIA (validation set) in huggingface repo. """
import os, requests, huggingface_hub
from huggingface_hub import hf_hub_download

def fetch_files_gaia_validation_hub(filename: str):
    """
    Fetch a file from the GAIA validation set hosted on Hugging Face Hub.
    Args:
        filename (str): The name of the file to fetch (e.g., 'question_1.json').
    Returns:
        str: Local path to the downloaded file.
    """
    if not filename:
        raise ValueError("Please provide a valid filename.")

    repo_id = "gaia-benchmark/GAIA"
    repo_type = "dataset"
    path_in_repo = f"2023/validation/{filename}"

    # If you need a private token: set HF_TOKEN env var or pass token=...
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=path_in_repo,
        # token=os.getenv("HF_TOKEN"),  # uncomment if needed
        force_download=False
    )

    # Copy into ./downloaded to keep a consistent local layout
    os.makedirs("downloaded", exist_ok=True)
    local_path = os.path.join("downloaded", filename)
    if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
        # Lightweight copy
        with open(downloaded_path, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())

    return local_path