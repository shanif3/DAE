import os
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class DAE:
    def __init__(self,
                 input_size=784,
                 latent_dim=None,
                 encoder_units=(128, 64),  # encoder -> latent_dim -> decoder (1 hidden layer each)
                 decoder_units=(64, 128),
                 activation_name=nn.ReLU(),
                 dropout_rate=0.0,
                 learning_rate=1e-3,column_types=None):

        # input size
        self.input_size = input_size

        # latent dimension
        self.latent_dim = latent_dim

        # device: whether gpu or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize the network and put into device
        self.network = DAE_Network(input_size=input_size,
                                   latent_dim=latent_dim,
                                   encoder_units=encoder_units,
                                   decoder_units=decoder_units,
                                   activation_name=activation_name,
                                   dropout_rate=dropout_rate, column_types=column_types).to(self.device)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # define loss functions
        self.bce_criterion = nn.BCELoss()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, n_epochs=1000, batch_size=32, show_progress=True):
        """
        Train the model and in each epoch generate a bernouli mask
        :param x:
        :param y:
        """
        self.network.train()

        x = torch.from_numpy(x.to_numpy()).float()
        y = torch.from_numpy(y.to_numpy()).float()

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train the model
        for epoch in range(n_epochs):
            # make a mask to the specific epoch
            for i, (x, y) in enumerate(dataloader):
                mask = self.create_mask(x).to(self.device)
                # x = np.nan_to_num(x, nan=0.0)
                # move batch to device
                x, y = x.float().to(self.device), y.float().unsqueeze(1).to(
                    self.device)  # y is transformed to (batch_size, 1)
                # forward step
                constructed, p = self.network(x, mask)
                # backward step
                self.optimizer.zero_grad()
                loss = self.mse_mask(constructed, x, mask) + self.bce_criterion(p, y)
                loss.backward()
                self.optimizer.step()

                if show_progress and epoch % 10 == 0 and i == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

    def mse_mask(self, constructed, x, mask):
        # sum of each row (mask) (dim=1), so we will get the amount of present features we are doing the loss on
        mse_for_each_sample = 1 / torch.sum(mask, dim=1) * torch.sum(torch.pow((x - constructed), 2))
        return torch.mean(mse_for_each_sample)

    def predict(self, x: Union[pd.DataFrame, np.ndarray],
                mask_vector: np.ndarray = None):
        self.network.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            # if mask_vector is not provided, use a mask of ones to include all features
            if mask_vector is None:
                mask_vector = np.ones(x.shape[1])

            mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1).to(self.device)
            constructed, p = self.network(x, mask)
            predicted = p > 0.5
            predicted = predicted.cpu().detach().numpy().astype(int)
            # print(f"Predicted: {predicted}")
        return predicted

    def predict_proba(self,
                      x: Union[pd.DataFrame, np.ndarray],
                      mask_vector: np.ndarray):
        self.network.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)
            mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1).to(self.device)
            constructed, p = self.network(x, mask)
            # TODO: check dimensions of p, check concatenation of 1-p and p
            # print(f"p shape: {p.shape}")
            p = p.cpu().detach().numpy()
            # print(f"p: {p}")
            proba = np.concatenate([1 - p, p], axis=1)

            # print(f"proba: {proba}")
            # print(f"proba shape: {proba.shape}")
            # print(f"Predicted: {predicted}")
        return proba

    def create_mask(self, x: torch.Tensor):
        """
        Create a mask for the input x, the mask is a tensor
        :return: mask tensor of 0 and 1 of the same size as x (batch_size, input_size), where 1 means the feature is present and 0 means the feature is missing
        """
        mask = ~x.isnan()
        # convert bool to int
        mask = mask.int()
        return mask

    def save_checkpoint(self, checkpoint_dir):
        state = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, 'model' + ".pth"))

    def load_checkpoint(self, checkpoint_dir):
        loaded = False
        if os.path.exists(checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
            if self.device.type == 'cpu':
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint_path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loaded = True
        # n_training_seqs = checkpoint['n_training_seqs']
        # loss_history = checkpoint['loss_history']

        return loaded


class DAE_Network(nn.Module):
    def __init__(self,
                 input_size=784,
                 latent_dim=None,
                 encoder_units=(128, 64),
                 decoder_units=(64, 128),
                 activation_name=nn.ReLU,
                 dropout_rate=0.0,
                 column_types=None):

        super(DAE_Network, self).__init__()

        if latent_dim is None:
            self.latent_dim = 3
        else:
            self.latent_dim = latent_dim

        # Store column types
        self.column_types = column_types if column_types is not None else ['continuous'] * input_size

        # Encoder
        encoder_layers = nn.ModuleList()
        current_size = input_size * 2
        for units in encoder_units:
            encoder_layers.append(nn.Linear(current_size, units))
            encoder_layers.append(activation_name())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            current_size = units
        encoder_layers.append(nn.Linear(current_size, latent_dim))
        encoder_layers.append(activation_name())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = nn.ModuleList()
        current_size = latent_dim
        for units in decoder_units:
            decoder_layers.append(nn.Linear(current_size, units))
            decoder_layers.append(activation_name())
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            current_size = units
        decoder_layers.append(nn.Linear(current_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

        # Separate layers for applying column-specific activations
        self.column_activations = nn.ModuleList()
        for col_type in self.column_types:
            match col_type:
                case 'categorical':
                    # Apply softmax to categorical columns
                    self.column_activations.append(nn.Softmax(dim=1))
                case 'bool':
                    # Use sigmoid for boolean columns
                    self.column_activations.append(nn.Sigmoid())
                case 'positive':
                    # Use sigmoid for boolean columns
                    self.column_activations.append(nn.ReLU())
                case 'continuous':
                    # Identity activation for continuous columns
                    self.column_activations.append(nn.Identity())

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):  # assume x is a tensor of size (batch_size, 784), mask is (784)
        # apply mask
        x = x * mask

        # concatenate the masked input with the mask
        # print(f"x: {x.shape}, mask: {mask.shape}")
        concat = torch.cat([x, mask], 1)
        # print(f"concat: {concat.shape}")
        latent = self.encoder(concat)
        constructed = self.decoder(latent)

        # Apply column-specific activations
        split_outputs = torch.split(constructed, [1] * constructed.size(1), dim=1)
        activated_outputs = [activation(col) for activation, col in zip(self.column_activations, split_outputs)]
        constructed = torch.cat(activated_outputs, dim=1)

        # mlp
        p = self.mlp(latent)
        return constructed,p
