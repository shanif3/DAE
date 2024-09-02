import os
import json
import pandas as pd
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from pytorch_lightning.loggers import WandbLogger
from main import DAELightning, prepare_data


def objective(trial, X_train, y_train, X_val, y_val, column_types,categorical_dims):
    # Hyperparameters to optimize
    latent_dim = trial.suggest_int('latent_dim', 10, 200)
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 3)

    hidden_dims = []
    for i in range(n_hidden_layers):
        hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 32, 256))

    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val.values))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the wandb logger and add hyperparameters to the config

    name= f"optuna_trial_number_{trial.number}"
    print(f"{name}")
    wandb_logger = WandbLogger(project='name')

    # Log all hyperparameters
    wandb_logger.experiment.config.update({
        "latent_dim": latent_dim,
        "n_hidden_layers": n_hidden_layers,
        **{f"hidden_dim_{i}": hidden_dims[i] for i in range(n_hidden_layers)},
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }, allow_val_change=True)


    # Create model
    model = DAELightning(
        input_size=X_train.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        column_types=column_types,
        categorical_dims= categorical_dims,
        learning_rate=learning_rate
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )

    # Initialize TensorBoard logger
    logger = TensorBoardLogger("calude/tb_logs", name=name)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        logger=[wandb_logger, logger],  # Use both loggers
        callbacks=[early_stop_callback],
        enable_progress_bar=True
    )

    # Train and validate
    trainer.fit(model, train_loader, val_loader)

    # Return the best validation loss
    return trainer.callback_metrics['val_loss'].item()


def optimize_model_for_dataset(dataset_path: str, target_column: str):
    # Prepare data
    data = prepare_data(dataset_path, target_column)

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    column_types = data['column_types']
    categorical_dims= data['categorical_dims']

    # Create a partial function with data pre-filled
    objective_with_data = lambda trial: objective(trial, X_train, y_train, X_val, y_val, column_types,categorical_dims)

    # Run the optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_with_data, n_trials=3)  # Adjust number of trials as needed

    best_params = study.best_params

    # Save the best hyperparameters
    os.makedirs('optuna_results', exist_ok=True)
    with open('optuna_results/best_params.json', 'w') as f:
        json.dump(best_params, f)

    # Create and save visualization plots
    # fig_slice = optuna.visualization.plot_slice(study)
    # fig_slice.write_image("optuna_results/slice_plot.png")

    # fig_param_importances = optuna.visualization.plot_param_importances(study)
    # fig_param_importances.write_image("optuna_results/param_importances_plot.png")

    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {study.best_value}")

    return best_params


if __name__ == "__main__":
    dataset_path = "Data/heart_statlog_cleveland_hungary_final.csv"
    target_column = 'target'
    best_params = optimize_model_for_dataset(dataset_path, target_column)
