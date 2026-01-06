import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from latency_aware_tree import LatencyAwareNST, encode_ttfs

def generate_visual_trace():
    # Load Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Train Model
    nst = LatencyAwareNST(max_depth=5, lambda_latency=0.1, time_steps=50)
    nst.fit(X, y)
    
    # Pick a sample
    sample_idx = 0
    x_sample = X[sample_idx]
    
    # Encode
    t_spikes = encode_ttfs(x_sample.reshape(1, -1), T_max=50)[0]
    
    # Trace path
    # We need to manually simulate to capture the path
    node = nst.root
    path = []
    
    curr_t = 0
    while not node.is_leaf:
        f_idx = node.feature_idx
        thresh = node.split_time
        spike_t = t_spikes[f_idx]
        
        # Determine branch
        if spike_t < thresh:
            # Right (Early)
            action = "Right (Spike)"
            arrival = spike_t
            decision_time = spike_t
            next_node = node.right
        else:
            # Left (Late)
            action = "Left (Timeout)"
            arrival = spike_t # Might be > thresh or 50
            decision_time = thresh
            next_node = node.left
            
        path.append({
            'depth': len(path),
            'threshold': thresh,
            'spike_time': spike_t,
            'decision_time': decision_time,
            'action': action,
            'feature': f_idx
        })
        
        node = next_node
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    
    depths = [p['depth'] for p in path]
    thresholds = [p['threshold'] for p in path]
    spikes = [p['spike_time'] for p in path]
    decisions = [p['decision_time'] for p in path]
    
    # Draw horizontal bars for thresholds
    # y-axis = Node Depth
    # x-axis = Time
    
    for i, p in enumerate(path):
        y = -i
        thresh = p['threshold']
        spk = p['spike_time']
        dec = p['decision_time']
        
        # Threshold Bar
        ax.barh(y, thresh, height=0.3, left=0, color='lightgray', alpha=0.5, label='Window' if i==0 else "")
        ax.plot([thresh, thresh], [y-0.2, y+0.2], 'k|', ms=10, label='Threshold' if i==0 else "")
        
        # Spike Marker
        if spk <= 50:
            color = 'green' if spk < thresh else 'red'
            marker = 'v' if spk < thresh else 'x'
            ax.plot(spk, y, marker=marker, color=color, ms=10, label='Spike' if i==0 else "")
        
        # Decision point
        # ax.plot(dec, y, 'o', color='blue')
        
        ax.text(52, y, f"Node {i}: f{p['feature']} (t={thresh:.1f}) -> {p['action']}", va='center')

    ax.set_yticks([-i for i in range(len(path))])
    ax.set_yticklabels([f"Depth {i}" for i in range(len(path))])
    ax.set_xlabel("Time Steps (TTFS)")
    ax.set_title("Visual Decision Trace (Breast Cancer Sample)")
    ax.set_xlim(0, 55)
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_trace.png')
    print("Generated decision_trace.png")

if __name__ == "__main__":
    generate_visual_trace()
