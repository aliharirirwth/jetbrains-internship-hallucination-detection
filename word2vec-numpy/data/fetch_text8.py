import time
import urllib.request
import zipfile
from pathlib import Path

URL = "http://mattmahoney.net/dc/text8.zip"
DIR = Path(__file__).resolve().parent
LOCAL_ZIP = DIR / "text8.zip"
LOCAL_TXT = DIR / "text8"


def main() -> None:
    if LOCAL_TXT.exists():
        print(f"{LOCAL_TXT} already exists. Delete it to re-download.")
        return
    t0 = time.perf_counter()
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, LOCAL_ZIP)
    print("Extracting...")
    with zipfile.ZipFile(LOCAL_ZIP, "r") as z:
        z.extractall(DIR)
    LOCAL_ZIP.unlink()
    print(f"Done in {time.perf_counter() - t0:.1f}s. Corpus at {LOCAL_TXT}")


if __name__ == "__main__":
    main()
