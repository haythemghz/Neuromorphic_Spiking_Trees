import numpy as np
import pandas as pd
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dataset_loader import load_q1_datasets
from latency_aware_tree import LatencyAwareNST, encode_ttfs
from spiking_forest import LatencyAwareNSF

# Configuration
RESULTS_FILE = 'unified_results.csv'
LAMBDAS = [0.0, 0.1, 0.25, 0.5]
N_SPLITS = 5
T_MAX = 50

def run_unified_benchmark():
    datasets = load_q1_datasets()
    all_results = []
    
    # Fresh start
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"Starting Unified Entropy Benchmark (Greedy NST/NSF)...")
    print(f"Protocol: {N_SPLITS}-fold Stratified CV per dataset.")

    for d_name, d_loader in datasets.items():
        print(f"\n>>> Processing Dataset: {d_name}")
        
        try:
            # Load Data
            X_tr, X_te, y_tr, y_te = d_loader()
            X = np.vstack((X_tr, X_te))
            y = np.concatenate((y_tr, y_te))
        except Exception as e:
            print(f"Error loading {d_name}: {e}")
            continue

        # PCA for high-dimensional image data
        if d_name in ['MNIST', 'Fashion-MNIST', 'UCI HAR'] and X.shape[1] > 50:
            print(f"  Applying PCA to {d_name} ({X.shape[1]} -> 50 components)...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            X = pca.fit_transform(X)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        
        fold_idx = 0
        for train_idx, test_idx in skf.split(X, y):
            fold_idx += 1
            print(f"  Fold {fold_idx}/{N_SPLITS}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Encode TTFS
            X_train_spikes = encode_ttfs(X_train, T_max=T_MAX)
            X_test_spikes = encode_ttfs(X_test, T_max=T_MAX)
            
            # 1. Baselines (Run once per fold)
            # CART
            cart = DecisionTreeClassifier(max_depth=10, random_state=42)
            cart.fit(X_train, y_train)
            cart_acc = accuracy_score(y_test, cart.predict(X_test))
            all_results.append({
                'Dataset': d_name, 'Model': 'CART', 'Lambda': 0.0, 'Fold': fold_idx,
                'Accuracy': cart_acc, 'Latency': 0.0, 'Depth': cart.get_depth(), 'FitTime': 0.0
            })
            
            # RF
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test))
            all_results.append({
                'Dataset': d_name, 'Model': 'RF', 'Lambda': 0.0, 'Fold': fold_idx,
                'Accuracy': rf_acc, 'Latency': 0.0, 'Depth': 10.0, 'FitTime': 0.0
            })
            print(f"    [Baselines] CART: {cart_acc:.4f}, RF: {rf_acc:.4f}")

            # 2. NST and NSF sweeps
            for lam in LAMBDAS:
                # NST
                nst = LatencyAwareNST(max_depth=10, lambda_latency=lam, time_steps=T_MAX)
                start_t = time.time()
                nst.fit(X_train_spikes, y_train)
                fit_time = time.time() - start_t
                
                preds, lats = nst.predict_with_latency(X_test_spikes)
                acc = accuracy_score(y_test, preds)
                lat = np.mean(lats)
                depth = nst.get_depth()
                
                all_results.append({
                    'Dataset': d_name, 'Model': 'NST', 'Lambda': lam, 'Fold': fold_idx,
                    'Accuracy': acc, 'Latency': lat, 'Depth': depth, 'FitTime': fit_time
                })
                print(f"    [NST] lam={lam}: Acc={acc:.4f}, Lat={lat:.1f}")

                # NSF
                nsf = LatencyAwareNSF(n_estimators=5, max_depth=10, lambda_latency=lam, time_steps=T_MAX)
                start_t = time.time()
                nsf.fit(X_train_spikes, y_train)
                fit_time = time.time() - start_t
                
                preds, lats = nsf.predict(X_test_spikes)
                acc = accuracy_score(y_test, preds)
                lat = np.mean(lats)
                avg_depth = np.mean([t.get_depth() for t in nsf.trees])
                
                all_results.append({
                    'Dataset': d_name, 'Model': 'NSF', 'Lambda': lam, 'Fold': fold_idx,
                    'Accuracy': acc, 'Latency': lat, 'Depth': avg_depth, 'FitTime': fit_time
                })
                print(f"    [NSF] lam={lam}: Acc={acc:.4f}, Lat={lat:.1f}")
                
                # Checkpoint
                pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)

    print("\nUnified Benchmark Complete. Results saved to", RESULTS_FILE)

if __name__ == "__main__":
    run_unified_benchmark()
