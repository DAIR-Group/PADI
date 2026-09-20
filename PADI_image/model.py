import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .network import PatchNetwork, PatchAutoencoder
from ..util import resolve_device

class AETrainer:

    def __init__(self, lr=0.0001, n_epochs=150, lr_milestones=None, batch_size=128, weight_decay=1e-06, device=None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones or []
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = resolve_device(device)
        self.train_time = None
        self.test_auc = None

    def train(self, train_set, ae_net):
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        ae_net = ae_net.to(self.device).float()
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):
            epoch_loss, n_batches = (0.0, 0)
            for data in loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device).float()
                optimizer.zero_grad()
                loss = torch.mean(criterion(ae_net(inputs), inputs))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f'AE Epoch {epoch + 1}/{self.n_epochs} | Loss: {epoch_loss / n_batches:.6f}')
        self.train_time = time.time() - start_time
        print(f'Pretraining completed in {self.train_time:.2f}s')
        return ae_net

    def test(self, test_set, ae_net):
        loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)
        ae_net = ae_net.to(self.device).float()
        criterion = nn.MSELoss(reduction='none')
        ae_net.eval()
        idx_label_score = []
        with torch.no_grad():
            for data in loader:
                inputs, labels, _, idx = data
                inputs = inputs.to(self.device).float()
                rec = ae_net(inputs)
                scores = torch.mean(criterion(rec, inputs), dim=tuple(range(1, rec.dim())))
                idx_label_score += list(zip(idx.cpu().numpy().tolist(), labels.cpu().numpy().tolist(), scores.cpu().numpy().tolist()))
        _, labels, scores = zip(*idx_label_score)
        labels, scores = (np.array(labels), np.array(scores))
        if len(np.unique(labels)) < 2:
            self.test_auc = 0.5
        else:
            from sklearn.metrics import roc_auc_score
            self.test_auc = roc_auc_score(labels, scores)
            print(f'Autoencoder Test AUC: {100.0 * self.test_auc:.2f}%')
        return self.test_auc


class DeepSVDDOneClassTrainer:

    def __init__(self, c=None, R_squared=None, nu=0.01, lr=0.0001, n_epochs=150, batch_size=128, weight_decay=1e-06, device=None, R_update_interval=1):
        self.device = resolve_device(device)
        self.nu = float(nu)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.R_update_interval = max(1, int(R_update_interval))
        self.c = None if c is None else c.to(self.device).float() if isinstance(c, torch.Tensor) else torch.tensor(c, device=self.device, dtype=torch.float32)
        self.R_squared = torch.tensor(float(R_squared or 0.0), device=self.device, dtype=torch.float32)
        self.train_time = None
        self.test_auc = None
        self.test_scores = None

    def init_center_c(self, train_loader, net, eps=0.1):
        n_samples = 0
        c = torch.zeros(net.repdim, device=self.device, dtype=torch.float32)
        net = net.float().to(self.device)
        net.eval()
        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[0].to(self.device).float()
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        c[(c > -eps) & (c < eps)] = eps
        self.c = c
        return self.c

    def train(self, train_set, net):
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        net = net.to(self.device).float()
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.c is None:
            self.init_center_c(loader, net)
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):
            epoch_loss, n_batches = (0.0, 0)
            for batch in loader:
                inputs = batch[0].to(self.device).float()
                optimizer.zero_grad()
                loss = torch.mean(torch.sum((net(inputs) - self.c) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 10 == 0:
                print(f'SVDD Epoch {epoch + 1}/{self.n_epochs} | Loss: {epoch_loss / max(n_batches, 1):.6f}')
        self.train_time = time.time() - start_time
        return net

    def test(self, test_set, net):
        loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)
        net = net.to(self.device).float()
        net.eval()
        idx_label_score = []
        default_idx = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device).float()
                labels = batch[1] if len(batch) > 1 else None
                scores = torch.sum((net(inputs) - self.c) ** 2, dim=1)
                if len(batch) > 3:
                    indices = batch[3].cpu().numpy().tolist()
                else:
                    indices = list(range(default_idx, default_idx + inputs.shape[0]))
                default_idx += inputs.shape[0]
                labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.zeros(inputs.shape[0])
                idx_label_score += list(zip(indices, labels_np.tolist(), scores.cpu().numpy().tolist()))
        self.test_scores = idx_label_score
        if idx_label_score:
            _, labels, scores = zip(*idx_label_score)
            labels, scores = (np.array(labels), np.array(scores))
            if len(np.unique(labels)) < 2:
                self.test_auc = 0.5
            else:
                from sklearn.metrics import roc_auc_score
                self.test_auc = roc_auc_score(labels, scores)
        return self.test_auc

class DeepSVDDOneClass:

    def __init__(self, nu=0.01, weight_decay=1e-06, R_update_interval=1, device=None):
        self.nu = float(nu)
        self.weight_decay = weight_decay
        self.R_update_interval = max(1, int(R_update_interval))
        self.device = resolve_device(device)
        self.repdim = None
        self.c = None
        self.R = None
        self.R_squared = None
        self.net = None
        self.ae_net = None
        self.trainer = None
        self.in_channels = None
        self.img_size = None
        self.channels = None

    def set_network(self, in_channels=1, img_size=(30, 30), repdim=32, channels=(16, 32, 64)):
        self.in_channels = in_channels
        self.img_size = img_size
        self.channels = channels
        self.repdim = repdim
        self.net = PatchNetwork(in_channels=in_channels, repdim=repdim, img_size=img_size, channels=channels, n_pool_blocks=2).float()
        self.ae_net = PatchAutoencoder(in_channels=in_channels, img_size=img_size, repdim=repdim, channels=channels, n_pool_blocks=2).float()

    def pretrain(self, train_set, test_set, **kwargs):
        device = resolve_device(kwargs.get('device'))
        self.device = device
        ae_trainer = AETrainer(device=device, **{k: v for k, v in kwargs.items() if k != 'device'})
        self.ae_net = ae_trainer.train(train_set, self.ae_net)
        ae_trainer.test(test_set, self.ae_net)
        net_dict = self.net.state_dict()
        ae_dict = {k: v for k, v in self.ae_net.encoder.state_dict().items() if k in net_dict}
        net_dict.update(ae_dict)
        self.net.load_state_dict(net_dict)

    def train(self, train_set, lr=0.0001, n_epochs=150, batch_size=128, weight_decay=None, device=None, **kwargs):
        device = resolve_device(device)
        self.device = device
        wd = self.weight_decay if weight_decay is None else weight_decay
        self.trainer = DeepSVDDOneClassTrainer(c=self.c, R_squared=self.R_squared, nu=self.nu, lr=lr, n_epochs=n_epochs, batch_size=batch_size, weight_decay=wd, device=device, R_update_interval=self.R_update_interval)
        self.net = self.trainer.train(train_set, self.net)
        self.c = self.trainer.c.detach().cpu().numpy().tolist()
        self.R_squared = None
        self.R = None
        return self.net

    def test(self, test_set, device=None):
        device = resolve_device(device)
        if self.trainer is None:
            self.trainer = DeepSVDDOneClassTrainer(c=self.c, R_squared=self.R_squared, nu=self.nu, weight_decay=self.weight_decay, device=device)
        self.trainer.test(test_set, self.net)

    def compute_radius(self, train_set, quantile=None, device=None):
        device = resolve_device(device)
        loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0, drop_last=False)
        self.net = self.net.to(device).float()
        self.net.eval()
        c_tensor = self.c if torch.is_tensor(self.c) else torch.tensor(self.c, device=device, dtype=torch.float32)
        if torch.is_tensor(c_tensor):
            c_tensor = c_tensor.to(device).float()
        quantile = 1.0 - self.nu if quantile is None else quantile
        scores = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device).float()
                dist = torch.sum((self.net(inputs) - c_tensor) ** 2, dim=1)
                scores.extend(dist.cpu().numpy().tolist())
        scores = np.array(scores)
        self.R_squared = float(np.max(scores)) if quantile >= 1.0 else float(np.quantile(scores, quantile))
        self.R = float(np.sqrt(max(self.R_squared, 0.0)))
        print(f'DeepSVDD radius: R={self.R:.6f}, R²={self.R_squared:.6f}')
        return self.R

    def predict_score(self, data_batch):
        self.net = self.net.to(self.device).float()
        data_batch = data_batch.to(self.device).float()
        c_tensor = self.c if torch.is_tensor(self.c) else torch.tensor(self.c, device=self.device, dtype=torch.float32)
        if torch.is_tensor(c_tensor):
            c_tensor = c_tensor.to(self.device).float()
        self.net.eval()
        with torch.no_grad():
            return torch.sum((self.net(data_batch) - c_tensor) ** 2, dim=1).cpu().numpy()

    def anomaly_detection_fixed_radius(self, data_batch, R_squared=None):
        if R_squared is None:
            R_squared = self.R_squared
        scores = self.predict_score(data_batch)
        O = np.where(scores > R_squared)[0]
        return (np.sort(O), scores, R_squared)

    def save_model(self, model_path='deep_svdd_model.pth'):
        d = os.path.dirname(model_path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save({'model_type': 'deep_svdd_one_class', 'nu': self.nu, 'weight_decay': self.weight_decay, 'R_update_interval': self.R_update_interval, 'repdim': self.repdim, 'c': self.c, 'R': self.R, 'R_squared': self.R_squared, 'in_channels': self.in_channels, 'img_size': self.img_size, 'channels': self.channels, 'net_state_dict': self.net.state_dict(), 'ae_state_dict': self.ae_net.state_dict() if self.ae_net else None}, model_path)

    @classmethod
    def load_model(cls, model_path, device=None):
        device = resolve_device(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = cls(nu=ckpt.get('nu', 0.01), weight_decay=ckpt.get('weight_decay', 1e-06), R_update_interval=ckpt.get('R_update_interval', 1), device=device)
        model.set_network(in_channels=ckpt.get('in_channels', 1), img_size=tuple(ckpt.get('img_size', (28, 28))), repdim=ckpt.get('repdim', 32), channels=tuple(ckpt.get('channels', (8, 4))))
        model.net.load_state_dict(ckpt['net_state_dict'])
        if ckpt.get('ae_state_dict') and model.ae_net:
            model.ae_net.load_state_dict(ckpt['ae_state_dict'])
        model.c = ckpt.get('c')
        if isinstance(model.c, torch.Tensor):
            model.c = model.c.float().to(device)
        elif model.c is not None:
            model.c = torch.tensor(model.c, device=device, dtype=torch.float32)
        model.R = ckpt.get('R')
        model.R_squared = ckpt.get('R_squared')
        model.net = model.net.float().to(device)
        model.net.eval()
        return model