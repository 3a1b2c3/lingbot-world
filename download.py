import argparse
from huggingface_hub import hf_hub_download, list_repo_files

def list_files_in_repo(repo_id):
    """
    Lists all files in the specified Hugging Face repository.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.

    Returns:
        list: A list of filenames in the repository.
    """
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model")
        print("Files in repository:")
        for file in files:
            print(f"- {file}")
        return files
    except Exception as e:
        print(f"An error occurred while listing files in the repository: {e}")
        return []

def download_model(repo_id, filename, local_dir):
    """
    Downloads a specific file from Hugging Face Hub to a specified local directory.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub (e.g., 'robbyant/lingbot-world-base-cam').
        filename (str): The name of the file to download from the repository.
        local_dir (str): The local directory where the file will be saved.
    """
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", local_dir=local_dir)
        print(f"File '{filename}' downloaded successfully to {local_dir}")
    except Exception as e:
        print(f"An error occurred while downloading the file '{filename}': {e}")

def download_all_files(repo_id, local_dir):
    """
    Downloads all files from a Hugging Face repository to a specified local directory.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        local_dir (str): The local directory where the files will be saved.
    """
    files = list_files_in_repo(repo_id)
    if not files:
        print("No files found in the repository.")
        return

    for filename in files:
        download_model(repo_id, filename, local_dir)

if __name__ == "__main__":
    # Available models
    MODELS = {
        "base-cam": "robbyant/lingbot-world-base-cam",
        "base-cam-nf4": "cahlen/lingbot-world-base-cam-nf4",
        "base-act": "robbyant/lingbot-world-base-act"
    }

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download Lingbot World models from Hugging Face")
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        choices=list(MODELS.keys()),
        default=["base-act", "base-cam-nf4"],
        help=f"Model(s) to download. Available options: {', '.join(MODELS.keys())} (default: base-act base-cam-nf4)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to save the model (default: ./model-name)"
    )

    args = parser.parse_args()

    for model in args.model:
        repo_id = MODELS[model]
        local_dir = args.local_dir if args.local_dir else f"./{model}"

        print(f"Downloading model: {model}")
        print(f"Repository: {repo_id}")
        print(f"Local directory: {local_dir}")
        print()

        download_all_files(repo_id, local_dir)