import os, glob

def find_dataset_dir() -> str:
    # try typical locations from where scripts/notebooks may run
    candidates = ["datasets", "../datasets", "../../datasets"]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "data_names.csv")):
            return c
    raise FileNotFoundError(
        "Couldn't find datasets directory containing data_names.csv. "
        "Move the script/notebook to project root or adjust this function."
    )

def list_data_files(ds_dir: str):
    return sorted(glob.glob(os.path.join(ds_dir, "data_*.csv")))
