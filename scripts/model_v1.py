import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hyperparams):
        super(Encoder, self).__init__()

        # dropout = hyperparams["dropout_rate"]
        channels = hyperparams["position_channels"]
        n_embed = hyperparams["n_embed"]
        filters = hyperparams["filters"]
        fc_size = hyperparams["fc_size"]
        
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters, filters * 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters * 2, filters * 4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters * 4 * 1 * 1, fc_size)
        self.fc2 = nn.Linear(fc_size, n_embed)  # Compressed representation
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.dropout(x)
        # x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # x = self.dropout(x)
        x = F.relu(self.conv4(x))  
        x = self.pool(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hyperparams):        
        super(Decoder, self).__init__()

        # dropout = hyperparams["dropout_rate"]
        channels = hyperparams["position_channels"]
        n_embed = hyperparams["n_embed"]
        filters = hyperparams["filters"]
        fc_size = hyperparams["fc_size"]

        
        self.fc1 = nn.Linear(n_embed, fc_size)
        self.fc2 = nn.Linear(fc_size, filters * 4 * 1 * 1)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(filters * 4 , 1, 1))
        self.deconv1 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv3 = nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(filters, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.unflatten(x)
        x = F.relu(self.deconv1(x))
        # x = self.dropout(x)
        x = F.relu(self.deconv2(x))
        # x = self.dropout(x)
        # x = self.deconv3(x)
        x = self.deconv4(x)
        return x

class PositionAutoEncoder(nn.Module):
    def __init__(self, hyperparams):
        super(PositionAutoEncoder, self).__init__()
        self.encoder = Encoder(hyperparams)
        self.decoder = Decoder(hyperparams)
        # self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        x = self.encoder(x)
        # x = self.dropout(x)
        x = self.decoder(x)

        return x

    @torch.no_grad()
    def embed(self, x):
        code = self.encoder(x)
        return code