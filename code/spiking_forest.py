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
        
        # Get predictions and latencies from all trees
        all_preds = np.zeros((n_samples, self.n_estimators), dtype=int)
        all_lats = np.zeros((n_samples, self.n_estimators))
        
        for i, tree in enumerate(self.trees):
            preds, lats = tree.predict_with_latency(X)
            all_preds[:, i] = preds
            all_lats[:, i] = lats
            
        # Temporal Consensus Logic
        for i in range(n_samples):
            # Sort by latency
            sample_lats = all_lats[i]
            sample_preds = all_preds[i]
            
            # Sort indices by latency
            sorted_idx = np.argsort(sample_lats)
            
            # Check for consensus over time
            votes = []
            decision_time = self.time_steps
            made_decision = False
            
            # Simple approach: Check if we have a majority at T_consensus?
            # Or accumulation?
            # User/paper says: "majority vote if >50% trees done by T_consensus"
            # Else wait?
            
            # Let's count votes that arrived before consensus_threshold
            early_indices = [idx for idx in sorted_idx if sample_lats[idx] <= self.consensus_threshold]
            
            if len(early_indices) > (self.n_estimators / 2):
                # Majority have arrived
                early_preds = sample_preds[early_indices]
                final_preds[i] = Counter(early_preds).most_common(1)[0][0]
                final_lats[i] = max(sample_lats[idx] for idx in early_indices) # Time of the last required vote? Or average? 
                # Usually max of the subset implies when the consensus is valid.
                # Actually, consensus is valid as soon as Majority is reached. 
                # Let's say we need 3/5. The time is the latency of the 3rd fastest tree.
                k = int(self.n_estimators / 2) + 1
                if len(early_indices) >= k:
                     # We take the k-th fastest latency as the decision time?
                     # Let's simplify: Avg latency of the voting trees or just the scalar stats.
                     # Let's use the mean of the participating trees for "Latency" metric.
                     final_lats[i] = np.mean([sample_lats[idx] for idx in early_indices])
            else:
                # Fallback: Wait for all or standard majority
                final_preds[i] = Counter(sample_preds).most_common(1)[0][0]
                final_lats[i] = np.mean(sample_lats)

        return final_preds, final_lats
