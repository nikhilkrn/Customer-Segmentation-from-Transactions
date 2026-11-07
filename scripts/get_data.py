"""
Download the UCI Online Retail dataset.

This script downloads the original Excel dataset and stores it at:
    data/raw/Online_Retail.xlsx

Run from the project root:
    python scripts/get_data.py

To replace an existing downloaded file:
    python scripts/get_data.py --force

The script uses only Python standard library.
"""

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


# UCI Machine Learning Repository: Online Retail (Dataset 352)
DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "Online_Retail.xlsx"


def download_dataset(force: bool = False) -> Path:
    """Download and extract the original Online Retail Excel file."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists() and not force:
        print(f"Dataset already exists: {OUTPUT_FILE}")
        print("Use --force if you want to download it again.")
        return OUTPUT_FILE

    print("Downloading the UCI Online Retail dataset...")


    with tempfile.TemporaryDirectory() as temp_dir:
        temp_archive = Path(temp_dir) / "online_retail.zip"

        request = urllib.request.Request(DATASET_URL,headers={"User-Agent": "CustomerSegmentationProject/1.0"},)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temp_archive.open("wb") as archive_file:
                    shutil.copyfileobj(response, archive_file)
        except Exception as exc:
            raise RuntimeError(f"Dataset download failed. Check your internet connection and try again. Details: {exc}") from exc

        if not zipfile.is_zipfile(temp_archive):
            raise RuntimeError("The downloaded file is not a valid ZIP archive. The dataset URL may have changed.")

        with zipfile.ZipFile(temp_archive, "r") as archive:
            excel_members = [
                member
                for member in archive.namelist()
                if Path(member).name.lower() == "online retail.xlsx"
            ]

            if not excel_members:
                raise FileNotFoundError("Could not find 'Online Retail.xlsx' in the downloaded UCI archive. Inspect the archive contents or verify the dataset URL.")


            source_member = excel_members[0]
            with archive.open(source_member) as source_file:
                with tempfile.NamedTemporaryFile(mode="wb",suffix=".xlsx",dir=RAW_DATA_DIR,delete=False,) as destination_file:
                    temporary_output = Path(destination_file.name)
                    shutil.copyfileobj(source_file, destination_file)


    try:
        temporary_output.replace(OUTPUT_FILE)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    print("Dataset downloaded and extracted successfully.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")

    return OUTPUT_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI Online Retail dataset.")
    parser.add_argument("--force",action="store_true",help="Download again and replace the existing raw Excel file.",)
    args = parser.parse_args()
    download_dataset(force=args.force)


if __name__ == "__main__":
    main()
