import json
from pathlib import Path

import opik

from app.config import get_settings

DATASET_NAME = "business-research-baseline"
DATASET_PATH = Path(__file__).parent / "baseline_dataset.json"


def main() -> None:
    settings = get_settings()

    client = opik.Opik(
        project_name=settings.opik_project_name,
    )

    # Start clean during development.
    try:
        client.delete_dataset(
            name=DATASET_NAME,
            project_name=settings.opik_project_name,
        )
        print(f"Deleted existing dataset: {DATASET_NAME}")
    except Exception:
        print("No existing dataset to delete.")

    dataset = client.create_dataset(
        name=DATASET_NAME,
        description="Baseline evaluation dataset for the AI Business Research system.",
        project_name=settings.opik_project_name,
    )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        items = json.load(file)

    dataset.insert(items)

    print(f"Inserted {len(items)} evaluation items.")
    print(f"Dataset: {dataset.name}")
    print(f"Current version: {dataset.get_current_version_name()}")

    version_info = dataset.get_version_info()
    
    if version_info:
        print(f"Version: {version_info.version_name}")
        print(f"Items: {version_info.items_total}")
    else:
        print("WARNING: Dataset has no version!")


if __name__ == "__main__":
    main()
