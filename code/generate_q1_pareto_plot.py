import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def generate_q1_pareto():
    print("Generating Top-Tier Q1 Pareto Comparison (Revised Version)...")
    
    # Absolute paths
    base_dir = r"d:\Haythem\Temp\AAAAA-Submitted papers\Neurmorphic Spiking Trees"
    csv_path = os.path.join(base_dir, "results", "top_q1_baselines_results.csv")
        
    df_baselines = pd.read_csv(csv_path)
    
    # Revised NSF and NST Data (cohere with Table 1 of the paper)
    # NST: 93.6% accuracy, 36.6 latency
    # NSF: 94.7% accuracy, 35.4 latency
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot Baselines with smooth lines
    plt.plot(df_baselines['Budget'], df_baselines['BranchyNet_Proxy'], 's-', 
             label='BranchyNet (Proxy)', color='#4169E1', linewidth=2, markersize=8, alpha=0.8)
    plt.plot(df_baselines['Budget'], df_baselines['ACT_Proxy'], '^-', 
             label='Adaptive Comp. Time (Proxy)', color='#2E8B57', linewidth=2, markersize=8, alpha=0.8)
    
    # Plot Our Spiking Tree/Forest models
    # NST (Single Tree)
    plt.scatter(36.6, 0.936, color='#FF8C00', s=150, marker='D', label='NST (Ours)', zorder=5, edgecolors='black')
    # NSF (Forest)
    plt.scatter(35.4, 0.947, color='#FF0000', s=180, marker='o', label='NSF (Ours)', zorder=6, edgecolors='black')
    # Early-exit variant of NSF
    plt.scatter(12.0, 0.850, color='#FF0000', s=120, marker='o', facecolors='none', zorder=5, edgecolors='#FF0000', linewidths=2)
    # Dotted line to connect the early-exit NSF to peak NSF
    plt.plot([12.0, 35.4], [0.850, 0.947], 'r:', alpha=0.6, zorder=1)
    
    # Annotate NSF best
    plt.annotate('NSF (Peak Performance)\nAcc: 94.7%, Latency: 35.4', xy=(35.4, 0.947), xytext=(38, 0.925),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
                 
    # Annotate NST best
    plt.annotate('NST (Single Tree)\nAcc: 93.6%, Latency: 36.6', xy=(36.6, 0.936), xytext=(40, 0.88),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=5, headlength=5),
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5))

    # Annotate early-exit variant
    plt.text(14, 0.84, 'Early-Exit NSF\n(Low Latency)', fontsize=9, color='#B22222', style='italic')

    plt.title("Accuracy-Latency Pareto Comparison (Top-Tier Q1)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Inference Latency (Simulation Time-steps)", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.ylim(0.7, 1.0)
    plt.xlim(0, 60)
    plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save paths
    paths_to_save = [
        os.path.join(base_dir, "figures", "q1_pareto_comparison.png"),
        os.path.join(base_dir, "Revised Version", "figures", "q1_pareto_comparison.png"),
        os.path.join(base_dir, "code", "q1_pareto_comparison.png")
    ]
    for path in paths_to_save:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=300)
            print(f"Saved copy to: {path}")
        except Exception as e:
            print(f"Failed to save copy to {path}: {e}")
            
    print("Plot saved successfully.")

if __name__ == "__main__":
    generate_q1_pareto()
