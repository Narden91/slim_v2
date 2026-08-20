import time
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from slim_gsgp.main_slim import slim
import sys
import warnings
warnings.filterwarnings('ignore')

def run_benchmark(dataset_name, dataset_loader):
    print(f"--- Benchmarking {dataset_name} (500 gens) ---")
    data = dataset_loader()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    start_time = time.time()
    try:
        slim(
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test, 
            dataset_name=dataset_name, 
            slim_version="SLIM+SIG1",
            pop_size=100,
            n_iter=500,
            log_level=0,
            verbose=0,
            n_jobs=1
        )
    except Exception as e:
        print(f"Error during solve: {e}")
    elapsed = time.time() - start_time
    
    print(f"{dataset_name} took {elapsed:.2f} seconds for 500 iterations.")
    return elapsed

if __name__ == "__main__":
    t1 = run_benchmark("Breast Cancer", load_breast_cancer)
    t2 = run_benchmark("Diabetes", load_diabetes)
    print(f"RESULTS: {t1:.2f}, {t2:.2f}")
