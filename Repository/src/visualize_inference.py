import numpy as np
import matplotlib.pyplot as plt
import copy
from spiking_tree import NeuromorphicTree

def plot_decision_trace(tree, sample_spikes, feature_names=None):
    """
    Visualizes the temporal decision path of a single sample.
    """
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(len(sample_spikes))]
        
    path = []
    node = tree.root
    t = 0
    
    # Trace the path
    while not node.is_leaf:
        f_idx = node.feature_idx
        f_spike = sample_spikes[f_idx]
        split_t = node.split_time
        
        # Determine movement
        fired_early = f_spike < split_t
        branch = 'RIGHT (Early)' if fired_early else 'LEFT (Late/NoSpike)'
        
        path.append({
            'feature': feature_names[f_idx],
            'f_spike': f_spike,
            'split_t': split_t,
            'choice': branch,
            't_accum': t
        })
        
        if fired_early:
            t = max(t, f_spike) + 1
            node = node.right
        else:
            t = max(t, split_t) + 1
            node = node.left
            
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(path))
    
    # Plot Split Threshold vs Spike Time
    ax.barh(y_pos, [p['split_t'] for p in path], color='skyblue', alpha=0.5, label='Node Split Threshold ($t_{split}$)')
    ax.scatter([p['f_spike'] for p in path], y_pos, color='red', marker='x', s=100, label='Feature Spike Time ($t_{spike}$)')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p['feature']}\n({p['choice']})" for p in path])
    ax.set_xlabel("Time Steps")
    ax.invert_yaxis()
    ax.set_title(f"NST Visual Decision Trace (Output: Class {node.prediction})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../experiments/benchmark_run/decision_trace.png")
    print("Decision trace saved to decision_trace.png")

if __name__ == "__main__":
    # Example logic test if needed
    pass
