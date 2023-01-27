import os
from pathlib import Path
import sys
import dill

def main(model_name: str):
    src_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    filename = src_dir / "model" / model_name
    headers = [
        "Config",
        "Text Converter",
        "Pipeline Names",
        "Pipeline",
        "N_Features"
    ]
    try:
        unpacked = dill.load(open(filename, 'rb'))
    except Exception as e:
        print(f"Something went wrong on loading model: {e}")
    
    for i, value in enumerate(unpacked):
        
        if i >= len(headers):
            header = "Undefined" # This lets us know if we've changed the .sav in ways we need to handle
        else:
            header = headers[i]

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
    