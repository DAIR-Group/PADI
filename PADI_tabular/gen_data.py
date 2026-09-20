import numpy as np
import torch
from torch.utils.data import Dataset

def generate_data(n_samples: int, n_features: int, num_anomalies: int, delta: float, mu: float=0.0, sigma=None, return_sigma: bool=False):
    if sigma is None:
        scale = 1.0
    elif np.isscalar(sigma):
        # In this project, scalar 'sigma' represents variance (matches covariance matrix Σ).
        # np.random.normal takes 'scale' as standard deviation, so we must take the square root.
        scale = np.sqrt(float(sigma))
    else:
        scale = None
    if scale is not None:
        X = np.random.normal(loc=mu, scale=scale, size=(n_samples, n_features)).astype(np.float64)
    else:
        sigma = np.asarray(sigma, dtype=np.float64)
        X = np.random.multivariate_normal(mean=np.full(n_features, mu), cov=sigma, size=n_samples).astype(np.float64)
    labels = np.zeros(n_samples, dtype=int)
    if num_anomalies > 0:
        anomaly_indices = np.random.choice(n_samples, num_anomalies, replace=False)
        labels[anomaly_indices] = 1
        X[anomaly_indices, :] += delta
    if return_sigma:
        Sigma = np.eye(n_features, dtype=np.float64) * float(sigma if sigma is not None else 1.0) if scale is not None else sigma
        return (X, labels, Sigma)
    else:
        return (X, labels)

class CustomDataset(Dataset):

    def __init__(self, X, y):
        self.data = torch.FloatTensor(X)
        self.targets = torch.LongTensor(y)

    def __getitem__(self, index):
        data = self.data[index]
        target = int(self.targets[index])
        return (data, target, index)

    def __len__(self):
        return len(self.data)

def prepare_data_for_deepsvdd(n_train, n_test, n_features, n_anomalies_train, n_anomalies_test, delta):
    X_train, y_train = generate_data(n_samples=n_train, n_features=n_features, num_anomalies=n_anomalies_train, delta=delta, mu=0)
    X_test, y_test = generate_data(n_samples=n_test, n_features=n_features, num_anomalies=n_anomalies_test, delta=delta, mu=0)
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    return (train_dataset, test_dataset)