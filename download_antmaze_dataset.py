from pathlib import Path
import argparse
import shutil

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-subpath", required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()

    minari_root = Path.home() / ".minari" / "datasets" / args.namespace / args.dataset_subpath

    print("Downloading dataset snapshot from Hugging Face...")
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=[f"{args.dataset_subpath}/**"],
        )
    )

    src = snapshot_path / args.dataset_subpath
    dst = minari_root

    print(f"Source:      {src}")
    print(f"Destination: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("Destination already exists. Removing old copy first...")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    print("\nDone.")
    print("Local dataset files:")
    for p in sorted(dst.rglob("*")):
        print(" ", p.relative_to(dst))


if __name__ == "__main__":
    main()