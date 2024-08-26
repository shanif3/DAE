# Denoising Autoencoder (DAE)

This project implements a Denoising Autoencoder (DAE) neural network using PyTorch and PyTorch Lightning.

## Setup
1. Clone the repository
2. Install required packages:   
`pip install -r requirements.txt`

## Usage

1. Prepare your dataset in CSV format.
2. Update the file path in `main()` function to point to your dataset.
3. Run the main script:  
`python main.py`

## Model Architecture

The DAE architecture consists of three primary components:

1. **Encoder**: Compresses input data into a latent space representation.
2. **Decoder**: Reconstructs the original input from the latent representation.
3. **Classifier**: Predicts target variables based on the learned latent representation.

The model employs type-specific activation functions to handle various column types effectively.

## Hyperparameters

Key hyperparameters include:

- `latent_dim`: Dimensionality of the latent space
- `hidden_dims`: Dimensions of hidden layers in encoder/decoder
- `dropout_rate`: Probability of neuron deactivation for regularization
- `learning_rate`: Step size for optimizer during training

These parameters can be adjusted in the `main()` function for performance optimization. Future updates will include Optuna for automated hyperparameter tuning.

## Results and Visualization

After training:

1. Console output will display key performance metrics.
2. Generated plots will be saved in the `plots` directory.
3. To visualize training progress with TensorBoard:  
`tensorboard --logdir=tb_logs`