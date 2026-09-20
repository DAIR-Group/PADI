import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchNetwork(nn.Module):

    def __init__(self, in_channels=1, repdim=32, img_size=(30, 30), channels=(16, 32, 64), n_pool_blocks: int=2):
        super().__init__()
        self.repdim = repdim
        self.img_size = img_size
        self.channels = channels
        self.n_pool_blocks = int(n_pool_blocks)
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c_out, eps=0.0001, affine=False))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            if i < self.n_pool_blocks:
                layers.append(nn.MaxPool2d(2, 2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        num_pools = min(self.n_pool_blocks, len(channels))
        final_h = img_size[0] // 2 ** num_pools
        final_w = img_size[1] // 2 ** num_pools
        if final_h < 1 or final_w < 1:
            raise ValueError(f'Patch too small. Input: {img_size}, Pools: {num_pools}.')
        self.fc = nn.Linear(c_in * final_h * final_w, repdim, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PatchNetworkDecoder(nn.Module):

    def __init__(self, out_channels=1, repdim=32, img_size=(30, 30), channels=(16, 32, 64), n_pool_blocks: int=2):
        super().__init__()
        self.repdim = repdim
        self.channels = channels
        self.target_size = img_size
        self.n_pool_blocks = int(n_pool_blocks)
        num_pools = min(self.n_pool_blocks, len(channels))
        current_h, current_w = (img_size[0], img_size[1])
        encoder_input_sizes = []
        for i in range(len(channels)):
            encoder_input_sizes.append((current_h, current_w))
            if i < num_pools:
                current_h = current_h // 2
                current_w = current_w // 2
        self.init_h = current_h
        self.init_w = current_w
        self.target_sizes = list(reversed(encoder_input_sizes))
        c_last = channels[-1]
        self.fc = nn.Linear(repdim, c_last * self.init_h * self.init_w, bias=False)
        rev_channels = list(channels[::-1])
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.is_last = []
        for i, c_in in enumerate(rev_channels):
            c_out = rev_channels[i + 1] if i < len(rev_channels) - 1 else out_channels
            is_last = i == len(rev_channels) - 1
            self.is_last.append(is_last)
            self.conv_layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            if not is_last:
                self.bn_layers.append(nn.BatchNorm2d(c_out, eps=0.0001, affine=False))
            else:
                self.bn_layers.append(nn.Identity())

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.channels[-1], self.init_h, self.init_w)
        for i, (conv, bn, is_last) in enumerate(zip(self.conv_layers, self.bn_layers, self.is_last)):
            target_h, target_w = self.target_sizes[i]
            if x.size(2) != target_h or x.size(3) != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode='nearest')
            x = conv(x)
            x = bn(x)
            if is_last:
                x = torch.sigmoid(x)
            else:
                x = F.leaky_relu(x, 0.01, inplace=True)
        return x

class PatchAutoencoder(nn.Module):

    def __init__(self, in_channels=1, img_size=(30, 30), repdim=32, channels=(16, 32, 64), n_pool_blocks: int=2):
        super().__init__()
        self.repdim = repdim
        self.img_size = img_size
        self.n_pool_blocks = int(n_pool_blocks)
        self.encoder = PatchNetwork(in_channels, repdim, img_size, channels, n_pool_blocks=self.n_pool_blocks)
        self.decoder = PatchNetworkDecoder(out_channels=in_channels, repdim=repdim, img_size=img_size, channels=channels, n_pool_blocks=self.n_pool_blocks)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)