import numpy as np

def latency_encode(data, time_steps=20, tau=1.0, threshold=0.0):
    """
    Encodes continuous data into spike latencies.
    Higher values -> Earlier spikes.
    
    Args:
        data (np.ndarray): Input data (n_samples, n_features), normalized [0, 1].
        time_steps (int): Maximum simulation time window.
    
    Returns:
        np.ndarray: Spike times array (n_samples, n_features). 
                    Values are integers in [0, time_steps].
                    Val = time_steps means "no spike".
    """
    # Inverse mapping: High value = Low latency
    # latency = (1 - val) * time_steps
    # Ensure data is 0-1
    data = np.clip(data, 0.0, 1.0)
    
    # Calculate float latencies
    latencies = (1.0 - data) * (time_steps - 1)
    
    # Round to nearest time step integer
    spike_times = np.round(latencies).astype(int)
    
    return spike_times

def rank_order_encode(data):
    """
    Encodes data based on the rank of feature values.
    """
    pass # Todo if needed
