import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from train_spiking_cifar import SpikingCNNFrontEnd
from spiking_forest import LatencyAwareNSF
from latency_aware_tree import encode_ttfs

def evaluate_cifar_nsf():
    print("Evaluating CIFAR-10 with Spiking CNN Front-End + NSF...")
    
    # 1. Load trained Front-End
    fe = SpikingCNNFrontEnd()
    fe.load_state_dict(torch.load('spiking_cifar_fe.pth'))
    fe.eval()
    
    # 2. Prepare Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Subset for speed
    indices = np.random.choice(len(testset), 500, replace=False)
    testloader = DataLoader(torch.utils.data.Subset(testset, indices), batch_size=500, shuffle=False)
    
    # 3. Extract Features
    with torch.no_grad():
        inputs, labels = next(iter(testloader))
        features = fe(inputs) # (500, 32, 8, 8)
        # Flatten and normalize for TTFS
        features = features.view(500, -1).numpy()
        # Scale to [0, 1] for encode_ttfs
        features = (features - features.min()) / (features.max() - features.min() + 1e-6)
        
    # 4. Use a dummy NSF to estimate accuracy (or train a quick one)
    # Since we can't train a full NSF on CIFAR features in seconds, 
    # we'll use the learned features as a metric.
    # Actually, let's train a very small forest.
    X_spikes = encode_ttfs(features)
    y = labels.numpy()
    
    nsf = LatencyAwareNSF(n_estimators=3, max_depth=10, lambda_latency=0.1, n_classes=10)
    # Quick train on a small subset
    nsf.fit(X_spikes[:400], y[:400])
    
    # Predict
    y_pred, lats = nsf.predict(X_spikes[400:])
    accuracy = np.mean(y_pred == y[400:])
    avg_lat = np.mean(lats)
    
    print(f"CIFAR-10 NSF Accuracy: {accuracy:.4f} @ Latency {avg_lat:.1f}")
    
    with open("cifar_result.txt", "w") as f:
        f.write(f"{accuracy:.4f},{avg_lat:.1f}")

if __name__ == "__main__":
    evaluate_cifar_nsf()
