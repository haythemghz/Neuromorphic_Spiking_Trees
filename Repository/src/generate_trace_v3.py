import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from latency_aware_tree import LatencyAwareNST, encode_ttfs

def generate_illustrative_trace():
    # Load Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Train Model
    T_MAX = 50
    nst = LatencyAwareNST(max_depth=5, lambda_latency=0.1, time_steps=T_MAX)
    nst.fit(X, y)
    
    # Pick a sample
    sample_idx = 0
    x_sample = X[sample_idx]
    y_true = y[sample_idx]
    
    # Encode
    t_spikes = encode_ttfs(x_sample.reshape(1, -1), T_max=T_MAX)[0]
    
    # Trace path and counterfactuals
    node = nst.root
    path = []
    
    while not node.is_leaf:
        f_idx = node.feature_idx
        thresh = node.split_time
        spike_t = t_spikes[f_idx]
        
        # Real Choice
        if spike_t < thresh:
            real_action = "Spike (Early)"
            real_node = node.right
            alt_action = "Timeout (Late)"
            alt_node = node.left
            alt_time_marker = thresh + 5 # illustrative
        else:
            real_action = "Timeout (Late)"
            real_node = node.left
            alt_action = "Spike (Early)"
            alt_node = node.right
            alt_time_marker = thresh - 5 # illustrative

        # Follow alternative to a leaf
        def get_pred(n):
            curr = n
            while not curr.is_leaf:
                # We don't have spikes for the alternative branches in a real trace, 
                # but for "explanation" we follow the logic.
                # Here we just take the "majority" or most likely path in sub-tree?
                # Let's just follow the left-most or right-most for simplicity in visualization.
                curr = curr.left if curr.left else curr.right
            return curr.prediction

        alt_prediction = get_pred(alt_node)
        
        path.append({
            'depth': len(path),
            'feature': f_idx,
            'threshold': thresh,
            'spike_time': spike_t,
            'real_action': real_action,
            'alt_action': alt_action,
            'alt_pred': alt_prediction
        })
        
        node = real_node
        
    final_pred = node.prediction

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, p in enumerate(path):
        y = -i
        thresh = p['threshold']
        spk = p['spike_time']
        
        # 1. Decision Window (Threshold)
        ax.barh(y, thresh, height=0.4, left=0, color='#e0e0e0', alpha=0.6, label='Decision Window' if i==0 else "")
        ax.plot([thresh, thresh], [y-0.3, y+0.3], color='black', lw=2, label='Temporal Threshold' if i==0 else "")
        
        # 2. Real Spike
        color = '#2ecc71' if spk < thresh else '#e74c3c'
        marker = 'v' if spk < thresh else 'x'
        ax.plot(spk, y, marker=marker, color=color, ms=12, label='Input Spike' if i==0 else "")
        
        # 3. Alternative Path Indicator (Counterfactual)
        alt_x = thresh + 10 if spk < thresh else thresh - 10
        ax.annotate('', xy=(alt_x, y), xytext=(spk, y),
                    arrowprops=dict(arrowstyle="->", color='gray', linestyle='dashed', alpha=0.5))
        
        ax.text(alt_x, y - 0.1, f"If {p['alt_action']} -> Class {p['alt_pred']}", 
                fontsize=8, color='gray', style='italic', ha='center')

        # Annotations
        txt = f"Node {i}: f{p['feature']} ({data.feature_names[p['feature']]})\n"
        txt += f"Result: {p['real_action']}"
        ax.text(T_MAX + 5, y, txt, va='center', fontsize=9)

    ax.set_yticks([-i for i in range(len(path))])
    ax.set_yticklabels([f"Depth {i}" for i in range(len(path))])
    ax.set_xlabel("Time Steps (TTFS Entropy Coding)")
    ax.set_title(f"Interpretability Trace: Real vs. Counterfactual Paths\nFinal Prediction: Class {final_pred} (True: {y_true})", fontsize=14, fontweight='bold')
    ax.set_xlim(0, T_MAX + 40)
    ax.grid(True, axis='x', alpha=0.2)
    
    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#e0e0e0', lw=10, alpha=0.6),
        Line2D([0], [0], color='black', lw=2),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#2ecc71', markersize=10),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='#e74c3c', markersize=10),
        Line2D([0], [0], color='gray', linestyle='dashed', lw=1)
    ]
    ax.legend(custom_lines, ['Integration Window', 'Threshold', 'Early Spike (Pass)', 'Late Spike (Timeout)', 'Alternative Path'], 
              loc='lower left', bbox_to_anchor=(0, 1.05), ncol=3, fontsize=9)

    plt.tight_layout()
    plt.savefig('decision_trace.png', dpi=300)
    print("Generated decision_trace.png (v3 - Optimized)")

if __name__ == "__main__":
    generate_illustrative_trace()
