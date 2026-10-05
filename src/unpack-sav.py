import os
from pathlib import Path
import sys

from JBGModelPersistence import load_model_artifact


def main(model_name: str):
    src_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    filename = src_dir / "model" / model_name
    headers = [
        "Config",
        "Text Converter",
        "Model Components",
        "Pipeline",
        "Keras Model Name",
        "N_Features",
    ]
    try:
        unpacked = load_model_artifact(filename)
    except Exception as e:
        print(f"Something went wrong on loading model: {e}")
        return

    for i, value in enumerate(unpacked):
        header = headers[i] if i < len(headers) else "Undefined"

        if i == 0:
            print(f"# {header}")
        else:
            print(f"\n\n# {header}")

        print(value)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Correct call: python unpack-sav.py <model.sav>")
    else:
        main(sys.argv[1])
