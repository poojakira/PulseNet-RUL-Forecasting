import hashlib
import urllib.request
from pathlib import Path

CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
EXPECTED_SHA256 = "55959b00c7ee12fda9d154d1f0f0026136cade9cd14b15b6d85836b7478b6528"


def download_and_verify():
    out_dir = Path("data/official")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "CMAPSSData.zip"

    if out_file.exists():
        print("Data already downloaded, verifying hash...")
    else:
        print(f"Downloading from {CMAPSS_URL}...")
        urllib.request.urlretrieve(CMAPSS_URL, out_file)

    sha256 = hashlib.sha256()
    with open(out_file, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)

    actual_hash = sha256.hexdigest()
    if actual_hash != EXPECTED_SHA256:
        out_file.unlink()
        raise ValueError(
            f"Hash mismatch! Expected {EXPECTED_SHA256}, got {actual_hash}. File deleted."
        )

    print("Download and verification successful.")


if __name__ == "__main__":
    download_and_verify()
