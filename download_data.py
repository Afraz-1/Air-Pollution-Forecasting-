"""Download the two public CSV files used by the original notebook."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests


URL = "https://www.kaggle.com/api/v1/datasets/download/rupakroy/lstm-datasets-multivariate-univariate"
FILES = ("LSTM-Multivariate_pollution.csv", "pollution_test_data1.csv")


def main() -> None:
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    output = Path("data")
    output.mkdir(exist_ok=True)
    with ZipFile(BytesIO(response.content)) as archive:
        for name in FILES:
            (output / name).write_bytes(archive.read(name))
            print(output / name)


if __name__ == "__main__":
    main()
