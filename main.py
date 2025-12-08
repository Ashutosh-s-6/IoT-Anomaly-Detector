from src.train_sklearn import run_sklearn
from src.train_keras import run_keras

if __name__ == "__main__":
    # Stream chunks across all files until we have a balanced sample
    # You can bump target_per_class to 10000 and chunksize to 200_000 if your machine can handle it
    run_sklearn(target_per_class=5000, chunksize=100_000, max_files=None, model_type="rf")
    run_keras(target_per_class=5000, chunksize=100_000, max_files=None, epochs=10, batch_size=1024)
