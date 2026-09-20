import os
import time
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .network import Network, Autoencoder

class AETrainer:

    def __init__(self, lr=0.0001, n_epochs=150, lr_milestones=None, batch_size=128, weight_decay=1e-06, device='cpu'):
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones or []
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

    def train(self, train_set, ae_net):
        print(f'\nPretraining autoencoder for {self.n_epochs} epochs...')
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        ae_net = ae_net.to(self.device).float()
        ae_net.train()
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        start_time = time.time()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device).float()
                optimizer.zero_grad()
                outputs = ae_net(inputs)
                loss = nn.MSELoss()(outputs, inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if (epoch + 1) % 50 == 0:
                print(f'  AE Epoch {epoch + 1}/{self.n_epochs} | Loss: {epoch_loss / n_batches:.6f}')
        train_time = time.time() - start_time
        print(f'Pretraining completed in {train_time:.2f}s')
        return ae_net

    def test(self, test_set, ae_net):
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)
        ae_net = ae_net.to(self.device).float()
        ae_net.eval()
        loss_total = 0.0
        n_batches = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device).float()
                outputs = ae_net(inputs)
                loss = nn.MSELoss()(outputs, inputs)
                loss_total += loss.item()
                n_batches += 1
        print(f'AE Test Loss: {loss_total / n_batches:.6f}')

class DeepSVDDTrainer:

    def __init__(self, c=None, R=0.0, objective='soft-boundary', nu=0.1, warm_up_n_epochs=10, lr=0.0001, n_epochs=150, lr_milestones=None, batch_size=128, weight_decay=1e-06, device='cpu'):
        self.c = c
        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.objective = objective
        self.nu = nu
        self.warm_up_n_epochs = warm_up_n_epochs
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones or []
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.R_squared = None
        self.train_time = None
        self.test_auc = None
        self.test_scores = None

    def init_center_c(self, train_loader, net, eps=0.1):
        n_samples = 0
        c = torch.zeros(net.repdim, device=self.device)
        net = net.to(self.device)
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device).float()
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def get_radius(self, dist, nu):
        return np.quantile(dist.cpu().numpy(), 1 - nu)

    def train(self, train_set, net):
        print(f'\nTraining Deep SVDD ({self.objective}) for {self.n_epochs} epochs...')
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        net = net.to(self.device).float()
        net.train()
        if self.c is None:
            self.c = self.init_center_c(train_loader, net)
        else:
            self.c = torch.tensor(self.c, device=self.device, dtype=torch.float32) if not isinstance(self.c, torch.Tensor) else self.c.to(self.device)
        self.R = self.R.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        start_time = time.time()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_dists = []
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device).float()
                optimizer.zero_grad()
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + 1.0 / self.nu * torch.mean(torch.clamp(scores, min=0))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                epoch_dists.append(dist.detach())
            if self.objective == 'soft-boundary' and epoch >= self.warm_up_n_epochs:
                all_dist = torch.cat(epoch_dists)
                new_R = self.get_radius(all_dist, self.nu)
                self.R.data = torch.tensor(new_R, device=self.device, dtype=torch.float32)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch + 1}/{self.n_epochs} | Loss: {epoch_loss / n_batches:.6f}')
        self.train_time = time.time() - start_time
        self.R_squared = float(self.R.item() ** 2)
        print(f'Training completed in {self.train_time:.2f}s')
        return net

    def test(self, test_set, net):
        print('\nTesting Deep SVDD model...')
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)
        net = net.to(self.device).float()
        net.eval()
        idx_label_score = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device).float()
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                idx_label_score += list(zip(idx.cpu().numpy().tolist(), labels.cpu().numpy().tolist(), dist.cpu().numpy().tolist()))
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_scores = scores
        try:
            from sklearn.metrics import roc_auc_score
            self.test_auc = roc_auc_score(labels, scores)
            print(f'Deep SVDD Test AUC: {100.0 * self.test_auc:.2f}%')
        except Exception:
            self.test_auc = None
            print('Could not compute AUC (sklearn issue or single-class labels)')
        return self.test_auc

class DeepSVDD:

    def __init__(self, objective='soft-boundary', nu=0.1, warm_up_n_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.objective = objective
        self.nu = nu
        self.warm_up_n_epochs = warm_up_n_epochs
        self.repdim = None
        self.c = None
        self.R = 0.0
        self.R_squared = None
        self.net = None
        self.ae_net = None
        self.trainer = None
        self.ae_trainer = None
        self.in_features = None
        self.hidden_dims = None

    def set_network(self, in_features: int=50, repdim: int=16, hidden_dims: Tuple[int, ...]=(32, 16, 8)):
        self.in_features = in_features
        self.repdim = repdim
        self.hidden_dims = hidden_dims
        self.net = Network(in_features, repdim, hidden_dims)
        self.ae_net = Autoencoder(in_features, repdim, hidden_dims)
        print(f'Network created: in={in_features}, hidden={hidden_dims}, rep={repdim}')

    def pretrain(self, train_set, test_set=None, lr=0.0001, n_epochs=150, lr_milestones=None, batch_size=128, weight_decay=1e-06, device=None):
        device = device if device else self.device
        self.ae_trainer = AETrainer(lr, n_epochs, lr_milestones, batch_size, weight_decay, device)
        self.ae_net = self.ae_trainer.train(train_set, self.ae_net)
        if test_set is not None:
            self.ae_trainer.test(test_set, self.ae_net)

    def init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.net.load_state_dict(net_dict)
        print('Network weights initialized from pretraining')

    def train(self, train_set, lr=0.0001, n_epochs=150, lr_milestones=None, batch_size=128, weight_decay=1e-06, device=None):
        device = device if device else self.device
        self.trainer = DeepSVDDTrainer(c=self.c, R=self.R, objective=self.objective, nu=self.nu, warm_up_n_epochs=self.warm_up_n_epochs, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay, device=device)
        self.net = self.trainer.train(train_set, self.net)
        self.c = self.trainer.c
        self.R = float(self.trainer.R.item())
        self.R_squared = self.trainer.R_squared

    def test(self, test_set, device=None):
        device = device if device else self.device
        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.c, device=device)
        return self.trainer.test(test_set, self.net)

    def compute_radius(self, train_set, quantile=1.0, device=None):
        device = device if device else self.device
        self.net = self.net.to(device).float()
        self.net.eval()
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        all_dist = []
        with torch.no_grad():
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(device).float()
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                all_dist.append(dist.cpu().numpy())
        all_dist = np.concatenate(all_dist)
        self.R_squared = float(np.quantile(all_dist, quantile))
        self.R = np.sqrt(self.R_squared)
        print(f'R² = {self.R_squared:.6f} (R = {self.R:.6f}) at quantile {quantile}')
        return self.R_squared

    def predict_score(self, data_batch: torch.Tensor):
        self.net = self.net.to(self.device).float()
        self.net.eval()
        c = self.c if isinstance(self.c, torch.Tensor) else torch.tensor(self.c, dtype=torch.float32)
        c = c.to(self.device)
        with torch.no_grad():
            data_batch = data_batch.to(self.device).float()
            outputs = self.net(data_batch)
            scores = torch.sum((outputs - c) ** 2, dim=1)
        return scores.cpu().numpy()

    def anomaly_detection_fixed_radius(self, data_batch: torch.Tensor, R_squared: float=None):
        if R_squared is None:
            R_squared = self.R_squared
        scores = self.predict_score(data_batch)
        is_anomaly = scores > R_squared
        anomaly_indices = np.where(is_anomaly)[0]
        return (anomaly_indices.tolist(), scores, is_anomaly)

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        state = {'net_state_dict': self.net.state_dict(), 'c': self.c.cpu().numpy() if isinstance(self.c, torch.Tensor) else self.c, 'R': self.R, 'R_squared': self.R_squared, 'objective': self.objective, 'nu': self.nu, 'in_features': self.in_features, 'repdim': self.repdim, 'hidden_dims': self.hidden_dims}
        torch.save(state, filepath)
        print(f'Model saved to {filepath}')

    def load_model(self, filepath, device=None):
        device = device or self.device
        state = torch.load(filepath, map_location=device, weights_only=False)
        self.objective = state['objective']
        self.nu = state['nu']
        self.in_features = state['in_features']
        self.repdim = state['repdim']
        self.hidden_dims = state['hidden_dims']
        self.set_network(self.in_features, self.repdim, self.hidden_dims)
        self.net.load_state_dict(state['net_state_dict'])
        self.net = self.net.to(device).float()
        self.net.eval()
        c_val = state['c']
        self.c = torch.tensor(c_val, device=device, dtype=torch.float32) if not isinstance(c_val, torch.Tensor) else c_val.to(device)
        self.R = state['R']
        self.R_squared = state['R_squared']
        self.device = device
        print(f'Model loaded from {filepath}')