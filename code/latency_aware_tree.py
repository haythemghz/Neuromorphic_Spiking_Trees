"""
Latency-Aware Entropy-Based Neuromorphic Spiking Tree (NST)

This module implements a deterministic, greedy tree construction algorithm
that maximizes information gain while penalizing inference latency.
Unlike evolutionary approaches, this uses CART/ID3-style top-down induction.

Split Score: Score(f, θ) = InfoGain(f, θ) - λ × (E[t_out] / T_max)
"""

import numpy as np
from collections import Counter


class LatencyAwareNode:
    """A node in the Latency-Aware NST with support for dendritic integration."""
    
    def __init__(self, feature_idx=None, split_time=None, weights=None, prediction=None):
        self.feature_idx = feature_idx  # Single index (int) or multiple (list/array)
        self.split_time = split_time    # Temporal threshold θ
        self.weights = weights          # Synaptic weights for dendritic integration
        self.left = None                # Late arrivals (t >= θ)
        self.right = None               # Early arrivals (t < θ)
        self.is_leaf = prediction is not None
        self.prediction = prediction    # Class label if leaf
        
    def get_integrated_latency(self, x_spikes):
        """Compute the dendritic integrated latency for this node's inputs."""
        if self.weights is None:
            return x_spikes[self.feature_idx]
        return np.sum(x_spikes[self.feature_idx] * self.weights)


class LatencyAwareNST:
    """
    Latency-Aware Neuromorphic Spiking Tree using entropy-based induction
    with optional dendritic (multi-feature) splitting.
    """
    
    def __init__(self, max_depth=8, min_samples_leaf=5, time_steps=50, 
                 lambda_latency=0.1, n_classes=2, use_dendritic=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.time_steps = time_steps
        self.lambda_latency = lambda_latency
        self.n_classes = n_classes
        self.use_dendritic = use_dendritic
        self.root = None
        
    def _entropy(self, y):
        """Compute Shannon entropy H = -Σ p_c log p_c."""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))
    
    def _information_gain(self, y, y_left, y_right):
        """Compute information gain from a split."""
        n = len(y)
        if n == 0: return 0.0
        n_l, n_r = len(y_left), len(y_right)
        if n_l == 0 or n_r == 0: return 0.0
        
        return self._entropy(y) - (n_l/n)*self._entropy(y_left) - (n_r/n)*self._entropy(y_right)
    
    def _find_best_split(self, X_spikes, y_labels, depth):
        """
        Find the best split maximizing Score = InfoGain - λ × (E[t_out] / T_max).
        Includes optional search for 2-feature dendritic splits.
        """
        n_samples, n_features = X_spikes.shape
        best_score = -np.inf
        best_f, best_theta, best_w = None, None, None
        
        # 1. Single-feature split search
        for f in range(n_features):
            spike_times = X_spikes[:, f]
            unique_times = np.unique(spike_times)
            if len(unique_times) < 2: continue
            
            candidates = np.linspace(unique_times[0], unique_times[-1], min(len(unique_times), 50))
            sort_idx = np.argsort(spike_times)
            y_sorted = y_labels[sort_idx]
            spikes_sorted = spike_times[sort_idx]
            split_pts = np.searchsorted(spikes_sorted, candidates)
            
            for i, theta in enumerate(candidates):
                pos = split_pts[i]
                if pos < self.min_samples_leaf or (n_samples - pos) < self.min_samples_leaf:
                    continue
                
                ig = self._information_gain(y_labels, y_sorted[pos:], y_sorted[:pos])
                e_lat = (pos/n_samples)*np.mean(spikes_sorted[:pos]) + ((n_samples-pos)/n_samples)*theta
                score = ig - self.lambda_latency * (e_lat / self.time_steps)
                
                if score > best_score:
                    best_score, best_f, best_theta, best_w = score, f, theta, None

        # 2. Dendritic (Multi-feature) search (if enabled and fruitful)
        if self.use_dendritic and best_f is not None and n_features > 1:
            orig_best_f = best_f
            # Try combining best feature with top correlates or random partners
            partner_f = (orig_best_f + 1) % n_features
            for w2 in [-1.0, -0.5, 0.5, 1.0]:
                f_pair = [orig_best_f, partner_f]
                w_pair = np.array([1.0, w2])
                mixed_latencies = np.sum(X_spikes[:, f_pair] * w_pair, axis=1)
                
                # Evaluate mixed split
                unique_mixed = np.unique(mixed_latencies)
                candidates = np.linspace(unique_mixed[0], unique_mixed[-1], min(len(unique_mixed), 50))
                sort_idx = np.argsort(mixed_latencies)
                y_sorted = y_labels[sort_idx]
                spikes_sorted = mixed_latencies[sort_idx]
                split_pts = np.searchsorted(spikes_sorted, candidates)
                
                for i, theta in enumerate(candidates):
                    pos = split_pts[i]
                    if pos < self.min_samples_leaf or (n_samples - pos) < self.min_samples_leaf: continue
                    ig = self._information_gain(y_labels, y_sorted[pos:], y_sorted[:pos])
                    e_lat = (pos/n_samples)*np.mean(spikes_sorted[:pos]) + ((n_samples-pos)/n_samples)*theta
                    score = ig - self.lambda_latency * (e_lat / self.time_steps)
                    
                    if score > best_score:
                        best_score, best_f, best_theta, best_w = score, f_pair, theta, w_pair
                        
        return best_f, best_theta, best_w, best_score

    def _build_tree(self, X_spikes, y, depth):
        n_samples = len(y)
        if (depth >= self.max_depth or n_samples < self.min_samples_leaf * 2 or len(np.unique(y)) == 1):
            return LatencyAwareNode(prediction=Counter(y).most_common(1)[0][0])
        
        best_f, best_theta, best_w, _ = self._find_best_split(X_spikes, y, depth)
        if best_f is None:
            return LatencyAwareNode(prediction=Counter(y).most_common(1)[0][0])
        
        node = LatencyAwareNode(feature_idx=best_f, split_time=best_theta, weights=best_w)
        
        # Calculate routing latencies (single or integrated)
        if best_w is None:
            node_latencies = X_spikes[:, best_f]
        else:
            node_latencies = np.sum(X_spikes[:, best_f] * best_w, axis=1)
            
        right_mask = node_latencies < best_theta
        node.right = self._build_tree(X_spikes[right_mask], y[right_mask], depth + 1)
        node.left = self._build_tree(X_spikes[~right_mask], y[~right_mask], depth + 1)
        return node

    def get_depth(self):
        def _get_depth(node):
            if node.is_leaf: return 0
            return 1 + max(_get_depth(node.left), _get_depth(node.right))
        return _get_depth(self.root) if self.root else 0

    def fit(self, X_spikes, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X_spikes, y, depth=0)
        return self

    def estimate_energy(self, X_spikes, e_spike=1.0, e_static=0.1):
        """
        Estimate energy consumption in μJ.
        E = Σ (path_length * e_spike) + e_static
        """
        _, lats = self.predict_with_latency(X_spikes)
        # In NST, spikes occur at each node decision.
        # Energy is roughly proportional to depth for each sample.
        # We can approximate spike count by looking at depth in predict_single if we track it.
        # For now, use average latency as a proxy for depth/activity.
        return np.mean(lats) * e_spike + e_static

    def _predict_single(self, spike_times):
        node = self.root
        latency = 0
        while not node.is_leaf:
            # Latency at this node
            if node.weights is None:
                t = spike_times[node.feature_idx]
            else:
                t = np.sum(spike_times[node.feature_idx] * node.weights)
            
            if t < node.split_time:
                latency = max(latency, t) + 1
                node = node.right
            else:
                latency = max(latency, node.split_time) + 1
                node = node.left
        return node.prediction, latency

    def predict(self, X_spikes):
        return np.array([self._predict_single(s)[0] for s in X_spikes])

    def predict_with_latency(self, X_spikes):
        preds = []
        lats = []
        for s in X_spikes:
            p, l = self._predict_single(s)
            preds.append(p)
            lats.append(l)
        return np.array(preds), np.array(lats)

    def score(self, X_spikes, y):
        return np.mean(self.predict(X_spikes) == y)


class LatencyAwareNSF:
    def __init__(self, n_estimators=3, max_depth=8, min_samples_leaf=5, time_steps=50, 
                 lambda_latency=0.1, max_features='sqrt', use_dendritic=True, use_consensus=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.time_steps = time_steps
        self.lambda_latency = lambda_latency
        self.max_features = max_features
        self.use_dendritic = use_dendritic
        self.use_consensus = use_consensus
        self.trees = []
        self.feature_indices = []

    def fit(self, X_spikes, y):
        n_samples, n_features = X_spikes.shape
        n_classes = len(np.unique(y))
        if self.max_features == 'sqrt': k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2': k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int): k = min(self.max_features, n_features)
        else: k = n_features

        self.trees, self.feature_indices = [], []
        for _ in range(self.n_estimators):
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            feat_idx = np.random.choice(n_features, k, replace=False)
            tree = LatencyAwareNST(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, 
                                   time_steps=self.time_steps, lambda_latency=self.lambda_latency, 
                                   n_classes=n_classes, use_dendritic=self.use_dendritic)
            tree.fit(X_spikes[boot_idx][:, feat_idx], y[boot_idx])
            self.trees.append(tree)
            self.feature_indices.append(feat_idx)
        return self

    def predict(self, X_spikes):
        preds = np.array([t.predict(X_spikes[:, f]) for t, f in zip(self.trees, self.feature_indices)])
        return np.array([Counter(preds[:, i]).most_common(1)[0][0] for i in range(X_spikes.shape[0])])

    def predict_with_latency(self, X_spikes):
        res = [t.predict_with_latency(X_spikes[:, f]) for t, f in zip(self.trees, self.feature_indices)]
        all_preds = np.array([r[0] for r in res])
        all_lats = np.array([r[1] for r in res])
        
        if not self.use_consensus:
            # Fallback to simple mean latency and majority vote
            final_preds = np.array([Counter(all_preds[:, i]).most_common(1)[0][0] for i in range(X_spikes.shape[0])])
            final_lats = np.mean(all_lats, axis=0)
            return final_preds, final_lats
            
        final_preds, final_lats = [], []
        for i in range(X_spikes.shape[0]):
            votes, lats = all_preds[:, i], all_lats[:, i]
            s_idx = np.argsort(lats)
            needed = (self.n_estimators // 2) + 1
            for k in range(needed, self.n_estimators + 1):
                counts = Counter(votes[s_idx[:k]])
                winner, count = counts.most_common(1)[0]
                if count >= needed:
                    final_preds.append(winner)
                    final_lats.append(lats[s_idx[k-1]])
                    break
            else:
                final_preds.append(Counter(votes).most_common(1)[0][0])
                final_lats.append(np.mean(lats))
        return np.array(final_preds), np.array(final_lats)


def encode_ttfs(X, T_max=50, noise_std=0.0):
    """
    Time-to-First-Spike encoding with optional jitter noise.
    t = T_max * (1 - normalized_x) + N(0, noise_std)
    """
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    range_ = X_max - X_min
    range_[range_ == 0] = 1
    
    # Normalize and scale
    latencies = (T_max * (1 - (X - X_min) / range_))
    
    # Add jitter
    if noise_std > 0:
        latencies += np.random.normal(0, noise_std, size=latencies.shape)
        
    return np.clip(latencies, 0, T_max).astype(np.float32)
