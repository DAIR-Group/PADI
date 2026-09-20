import torch
from torch.utils.data import Dataset

class SyntheticFullImageDataset(Dataset):

    def __init__(self, n_samples, img_size=(192, 192), mode='normal', channels=1, delta=20.0):
        self.img_size = img_size
        H, W = img_size
        mean = 127.5 if mode == 'normal' else 127.5 + delta
        raw = torch.randn(n_samples, channels, H, W) + mean
        raw = torch.clamp(raw, 0, 255)
        self.data = raw / 255.0
        self.pixel_std_normalized = 1.0 / 255.0
        self.labels = torch.zeros(n_samples, dtype=torch.long) if mode == 'normal' else torch.ones(n_samples, dtype=torch.long)
        self.semi_targets = self.labels.clone()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.semi_targets[idx], idx)

def prepare_synthetic_global_anomaly(img_size=(192, 192), n_train=2000, n_ref=50, n_test_normal=100, n_test_anomaly=100, delta=20.0):
    train_full_ds = SyntheticFullImageDataset(n_samples=n_train, img_size=img_size, mode='normal')
    ref_full_ds = SyntheticFullImageDataset(n_samples=n_ref, img_size=img_size, mode='normal')
    test_normal = SyntheticFullImageDataset(n_samples=n_test_normal, img_size=img_size, mode='normal')
    test_anomaly = SyntheticFullImageDataset(n_samples=n_test_anomaly, img_size=img_size, mode='anomaly', delta=delta)
    test_full_ds = torch.utils.data.ConcatDataset([test_normal, test_anomaly])
    return (train_full_ds, test_full_ds, ref_full_ds)