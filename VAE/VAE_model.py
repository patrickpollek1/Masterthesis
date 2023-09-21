import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, x_dim, z_dim,zz_dim,encoder_z_layers,encoder_zz_layers,decoder_layers,fc_z_to_y_layers,z_and_zz_to_y=True):
        super(VAE, self).__init__()

        self.z_and_zz_to_y=z_and_zz_to_y
        
        # Define layers for the z encoder
        layers = []
        input_dim = x_dim
        for layer_dim in encoder_z_layers:
            layers.extend([
                nn.Linear(input_dim, layer_dim),
                nn.ReLU(),
            ])
            input_dim = layer_dim
        self.encoder_z_seq = nn.Sequential(*layers)
        self.fc21 = nn.Linear(layer_dim, z_dim)
        self.fc22 = nn.Linear(layer_dim, z_dim)
        
        # Define layers for the zz encoder
        layers = []
        input_dim_zz = z_dim
        for layer_dim in encoder_zz_layers:
            layers.extend([
                nn.Linear(input_dim_zz, layer_dim),
                nn.ReLU(),
            ])
            input_dim_zz = layer_dim
        self.encoder_zz_seq = nn.Sequential(*layers)
        self.fcz21 = nn.Linear(input_dim_zz, zz_dim)
        self.fcz22 = nn.Linear(input_dim_zz, zz_dim)
        
        # Define layers for the decoder
        if decoder_layers[-1] != x_dim:
            print("Wrong output dim decoder_layers[-1]!=x_dim ")
        layers = []
        input_dim_de = z_dim + zz_dim
        for i, layer_dim in enumerate(decoder_layers):
            layers.extend([
                nn.Linear(input_dim_de, layer_dim)
            ])
            if i < len(decoder_layers)-1:
                layers.extend([nn.ReLU(),])
            input_dim_de = layer_dim
        self.decoder = nn.Sequential(*layers)
        
        if fc_z_to_y_layers[-1] != 1:
            print("Wrong output dim fc_z_to_y_layers[-1]!=1")
        layers = []
        if z_and_zz_to_y:
            input_dim = zz_dim+z_dim
        else:
            input_dim = zz_dim
        for i, layer_dim in enumerate(fc_z_to_y_layers):
            layers.extend([
                nn.Linear(input_dim, layer_dim),
            ])
            if i < len(fc_z_to_y_layers)-1:
                layers.extend([nn.ReLU(),])
            input_dim = layer_dim
        self.zz_to_y = nn.Sequential(*layers)

    def encode_z(self, x):
        x = self.encoder_z_seq(x)
        return self.fc21(x), self.fc22(x)
    
    def encode_zz(self, z):
        z = self.encoder_zz_seq(z)
        return self.fcz21(z), self.fcz22(z)
    
    def reparameterize(self, mu, logvar, n):
        samples = []
        for _ in range(n):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            sample = mu + eps*std
            samples.append(sample)
        return torch.mean(torch.stack(samples), dim=0)
    
    def forward(self, x):
        mu_z, logvar_z = self.encode_z(x)
        z = self.reparameterize(mu_z, logvar_z,1)
        mu_zz, logvar_zz = self.encode_zz(z)
        zz = self.reparameterize(mu_zz, logvar_zz,1)
        if self.z_and_zz_to_y:
            y_pred = self.zz_to_y(torch.cat((z, zz), 1))
        else:
            y_pred = self.zz_to_y(zz)
        latent = torch.cat((z, zz), 1)
        return self.decoder(latent), mu_z, logvar_z, mu_zz, logvar_zz, y_pred
    


