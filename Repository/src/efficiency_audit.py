import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from latency_aware_tree import LatencyAwareNSF, encode_ttfs

def load_digits_data():
    data = load_breast_cancer()
    return data.data, data.target

def calculate_snn_sop_baseline(input_dim, hidden_dim, output_dim):
    r"""
    Theoretical SOP count for a TTFS-coded Spiking MLP (one spike per synapse).
    $N_{SOP} = D \cdot H + H \cdot C$
    """
    return (input_dim * hidden_dim) + (hidden_dim * output_dim)

def run_efficiency_audit():
    print("🚀 Starting HEAV-v2 Efficiency Audit...")
    
    # 1. Load Data
    X, y = load_digits_data()
    X_spikes = encode_ttfs(X, T_max=50)
    
    # 2. Train NST (Small ensemble for interpretability)
    print("Training Neuromorphic Spiking Forest...")
    nsf = LatencyAwareNSF(n_estimators=10, max_depth=10, lambda_latency=0.5)
    nsf.fit(X_spikes, y)
    
    # 3. Collect NST Results
    _, _, nst_sops = nsf.predict_with_latency(X_spikes)
    avg_nst_sops = np.mean(nst_sops)
    print(f"✅ NST Average SOPs: {avg_nst_sops:.2f}")
    
    # 4. Define SNN Baselines (MNIST-scale or Digits-scale)
    # Digits dataset is 8x8 = 64 features.
    input_dim = X.shape[1]
    hidden_dim = 128
    output_dim = len(np.unique(y))
    
    snn_sop = calculate_snn_sop_baseline(input_dim, hidden_dim, output_dim)
    print(f"✅ SNN Baseline SOPs (MLP {input_dim}-{hidden_dim}-{output_dim}): {snn_sop}")
    
    # 5. Comparative Audit Table
    audit_data = [
        {"Model": "Spiking MLP (Baseline)", "Avg_SOPs": snn_sop, "Relative_Complexity": 1.0},
        {"Model": "NST (Ours)", "Avg_SOPs": avg_nst_sops, "Relative_Complexity": avg_nst_sops / snn_sop}
    ]
    df = pd.DataFrame(audit_data)
    df.to_csv("results/efficiency_audit.csv", index=False)
    print("\n📊 Efficiency Audit Results:")
    print(df)
    
    # 6. Generate Pareto-Equivalent Figure
    models = ["Spiking MLP", "NST"]
    sops = [snn_sop, avg_nst_sops]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, sops, color=['#34495e', '#e74c3c'], alpha=0.8)
    plt.yscale('log')
    plt.ylabel('Hardware-Equivalent Synaptic Operations (SOPs)')
    plt.title('HEAV-v2: Computational Footprint Comparison')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add annotations
    plt.text(1, avg_nst_sops, f"{snn_sop/avg_nst_sops:.1f}x Reduction", 
             ha='center', va='bottom', fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig("figures/heav_v2_efficiency.png", dpi=300)
    print("📈 Figure saved to figures/heav_v2_efficiency.png")

if __name__ == "__main__":
    run_efficiency_audit()
