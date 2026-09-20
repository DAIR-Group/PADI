import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, in_features: int=50, repdim: int=16, hidden_dims: tuple=(32, 16, 8)):
        super().__init__()
        self.in_features = in_features
        self.repdim = repdim
        self.hidden_dims = hidden_dims
        layers = []
        dim_in = in_features
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out, bias=False))
            layers.append(nn.BatchNorm1d(dim_out, eps=0.0001, affine=False))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            dim_in = dim_out
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(dim_in, repdim, bias=False)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

class NetworkDecoder(nn.Module):

    def __init__(self, out_features: int=50, repdim: int=16, hidden_dims: tuple=(8, 16, 32)):
        super().__init__()
        self.out_features = out_features
        self.repdim = repdim
        layers = []
        dim_in = repdim
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out, bias=False))
            layers.append(nn.BatchNorm1d(dim_out, eps=0.0001, affine=False))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            dim_in = dim_out
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(dim_in, out_features, bias=False)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

class Autoencoder(nn.Module):

    def __init__(self, in_features: int=50, repdim: int=16, hidden_dims: tuple=(32, 16, 8)):
        super().__init__()
        self.repdim = repdim
        self.encoder = Network(in_features, repdim, hidden_dims)
        decoder_hidden = tuple(reversed(hidden_dims))
        self.decoder = NetworkDecoder(in_features, repdim, decoder_hidden)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)