import argparse
import os
from huggingface_hub import snapshot_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    MODELS = {
        #"base-cam": "robbyant/lingbot-world-base-cam",
        "base-cam-nf4": "cahlen/lingbot-world-base-cam-nf4",
        "base-act": "robbyant/lingbot-world-base-act",
        "fast": "robbyant/lingbot-world-fast",
    }

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
        help="Local directory to save the model (default: <script_dir>/<model-name>)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/.cache/huggingface/hub"),
        help="Shared HF cache directory (default: ~/.cache/huggingface/hub, matches huggingface_hub's own default so all HF tools share one cache)."
    )

    args = parser.parse_args()

    for model in args.model:
        repo_id = MODELS[model]
        local_dir = args.local_dir if args.local_dir else os.path.join(SCRIPT_DIR, model)

        print(f"Downloading model: {model}")
        print(f"Repository: {repo_id}")
        print(f"Local directory: {local_dir}")
        print(f"Cache directory: {args.cache_dir}")
        print()

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            cache_dir=args.cache_dir,
        )
        print(f"Model '{model}' downloaded to {local_dir}")
