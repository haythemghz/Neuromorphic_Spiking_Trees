import numpy as np
from collections import Counter
from latency_aware_tree import LatencyAwareNST

class LatencyAwareNSF:
    """
    Neuromorphic Spiking Forest (NSF) using Bagging and Temporal Consensus.
    
    Constructs an ensemble of LatencyAwareNSTs.
    - Training: Bootstrap Aggregation (Bagging) with random feature subsets.
    - Inference: Temporal Consensus (Majority vote of early finishers).
    """
    def __init__(self, n_estimators=5, max_depth=8, lambda_latency=0.1, time_steps=50, n_classes=2, consensus_threshold=30):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lambda_latency = lambda_latency
        self.time_steps = time_steps
        self.n_classes = n_classes
        self.consensus_threshold = consensus_threshold # Time by which >50% trees must agree
        self.trees = []
    
    def fit(self, X_spikes, y):
        """
        Fit the forest using bootstrap bagging.
        """
        self.trees = []
        n_samples = X_spikes.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X_spikes[indices]
            y_sample = y[indices]
            
            # Create and train tree
            tree = LatencyAwareNST(
                max_depth=self.max_depth, 
                lambda_latency=self.lambda_latency, 
                time_steps=self.time_steps,
                n_classes=self.n_classes
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        return self

    def predict(self, X):
        """
        Predict using temporal consensus.
        """
        n_samples = X.shape[0]
        final_preds = np.zeros(n_samples, dtype=int)
        final_lats = np.zeros(n_samples)
        final_sops = np.zeros(n_samples)
        
        # Get predictions and latencies from all trees
        all_preds = np.zeros((n_samples, self.n_estimators), dtype=int)
        all_lats = np.zeros((n_samples, self.n_estimators))
        all_s_counts = np.zeros((n_samples, self.n_estimators))
        
        for i, tree in enumerate(self.trees):
            preds, lats, s_counts = tree.predict_with_latency(X)
            all_preds[:, i] = preds
            all_lats[:, i] = lats
            all_s_counts[:, i] = s_counts
            
        # Temporal Consensus Logic
        for i in range(n_samples):
            sample_lats = all_lats[i]
            sample_preds = all_preds[i]
            sample_s_counts = all_s_counts[i]
            
            # Sort indices by latency
            sorted_idx = np.argsort(sample_lats)
            
            # Check for consensus over time
            # Let's count votes that arrived before consensus_threshold
            early_indices = [idx for idx in sorted_idx if sample_lats[idx] <= self.consensus_threshold]
            
            k_threshold = (self.n_estimators // 2) + 1
            
            if len(early_indices) >= k_threshold:
                # We have enough votes for a potential majority
                early_preds = sample_preds[early_indices]
                counts = Counter(early_preds)
                winner, count = counts.most_common(1)[0]
                
                if count >= k_threshold:
                    # Clear majority reached early
                    final_preds[i] = winner
                    # Latency is the time of the k-th vote that sealed the deal or mean?
                    # Let's use the time of the vote that reached majority.
                    # To find which one: 
                    vote_count = 0
                    for idx in sorted_idx:
                        if sample_preds[idx] == winner:
                            vote_count += 1
                            if vote_count == k_threshold:
                                final_lats[i] = sample_lats[idx]
                                break
                    
                    # SOPs: Sum of SOPs for all trees that had to run up to that point
                    参与tree_idx = sorted_idx[:early_indices.index(idx)+1] if idx in early_indices else sorted_idx
                    final_sops[i] = np.sum(sample_s_counts[参与tree_idx])
                    continue

            # Fallback: Wait for all or standard majority
            final_preds[i] = Counter(sample_preds).most_common(1)[0][0]
            final_lats[i] = np.mean(sample_lats)
            final_sops[i] = np.sum(sample_s_counts)

        return final_preds, final_lats, final_sops
