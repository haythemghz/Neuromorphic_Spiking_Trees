"""
Generate a Loihi 2 Net-IO compatible spike-timing trace.

Produces a JSON file (loihi_netio_trace.json) containing per-sample
timing traces that record which neuron (tree node) fires at what time
step during inference, in a format compatible with Loihi 2 Net-IO
event-driven pipeline specifications.

Ported from Archive/Legacy_v1 and updated to the current codebase API.
"""

import numpy as np
import json
from dataset_loader import load_breast_cancer_preprocessed
from latency_aware_tree import encode_ttfs, LatencyAwareNST


def generate_loihi_netio_trace(n_samples=5):
    """
    Generate a spike-timing trace for a few samples in a format compatible
    with Loihi 2 neuromorphic Net-IO specifications.

    Output format per sample:
        { "sample_id": int,
          "label": int,
          "trace": [ {"time": float, "neuron_id": int, "feature": int,
                       "threshold": float, "branch": str}, ... ] }
    """
    X_tr, X_te, y_tr, y_te = load_breast_cancer_preprocessed()

    # Encode spike times
    X_tr_spikes = encode_ttfs(X_tr)
    X_te_spikes = encode_ttfs(X_te[:n_samples])

    # Train a single tree to obtain decision paths
    tree = LatencyAwareNST(max_depth=5, lambda_latency=0.1, time_steps=50)
    tree.fit(X_tr_spikes, y_tr)

    traces = []
    for i in range(n_samples):
        sample_spikes = X_te_spikes[i]
        path_info = []

        # Simulate the traversal to record timing at each node
        curr = tree.root
        t_accum = 0
        node_counter = 0

        while curr and not curr.is_leaf:
            feature_idx = curr.feature_idx
            threshold = curr.split_time
            feature_fire_time = float(sample_spikes[feature_idx])

            # Node fires at the later of: accumulated time, feature spike
            fire_time = max(t_accum, feature_fire_time)

            # Determine branch (matches current codebase: right = early, left = late)
            if feature_fire_time < threshold:
                branch = "right_early"
                next_node = curr.right
            else:
                branch = "left_late"
                next_node = curr.left

            path_info.append({
                "time": round(fire_time, 4),
                "neuron_id": node_counter,
                "feature": int(feature_idx) if np.isscalar(feature_idx) else [int(f) for f in feature_idx],
                "threshold": round(float(threshold), 4),
                "branch": branch
            })

            curr = next_node
            t_accum = fire_time + 1  # +1 step for synaptic delay
            node_counter += 1

        # Record final prediction
        prediction = int(curr.prediction) if curr else -1

        traces.append({
            "sample_id": i,
            "true_label": int(y_te[i]),
            "predicted_label": prediction,
            "n_nodes_traversed": len(path_info),
            "total_latency_steps": round(t_accum, 4),
            "trace": path_info
        })

    output_path = "loihi_netio_trace.json"
    with open(output_path, "w") as f:
        json.dump(traces, f, indent=4)
    print(f"Loihi 2 Net-IO trace saved to {output_path} ({n_samples} samples)")

    return traces


if __name__ == "__main__":
    generate_loihi_netio_trace()
