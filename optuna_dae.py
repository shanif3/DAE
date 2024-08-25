import os
import json
import pandas as pd

import optuna
from functools import partial
import torch.nn as nn
from see import DAE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

BATCH_SIZE = 32


def objective(trial, X_train, y_train, X_val, y_val):
    # Determine the columns for preprocessing
    non_categorical_columns = [col for col in X_train.columns if X_train[col].nunique() > 2]
    categorical_columns = [col for col in X_train.columns if col not in non_categorical_columns]
    original_columns = X_train.columns

    scaler_pipeline = ColumnTransformer([
        ('scaler', StandardScaler(), non_categorical_columns)
    ], remainder='passthrough')

    X_train = pd.DataFrame(scaler_pipeline.fit_transform(X_train),
                           columns=non_categorical_columns + categorical_columns)
    # keep the order of columns
    X_train = X_train[original_columns]

    X_val = pd.DataFrame(scaler_pipeline.transform(X_val),
                         columns=non_categorical_columns + categorical_columns)
    # keep the order of columns
    X_val = X_val[original_columns]

    # Hyperparameters to optimize
    latent_dim = trial.suggest_int('latent_dim', 10, 50)

    encoder_units = []
    previous_size = 256
    for i in range(2):
        current_size = trial.suggest_int(f'encoder_units_{i}', 64, previous_size)
        encoder_units.append(current_size)
        previous_size = current_size

    # Enforcing increasing sizes for decoder units
    decoder_units = []
    previous_size = 64
    for i in range(2):
        current_size = trial.suggest_int(f'decoder_units_{i}', previous_size, 256)
        decoder_units.append(current_size)
        previous_size = current_size

    # Convert to tuple
    encoder_units = tuple(encoder_units)
    decoder_units = tuple(decoder_units)

    activation_name = trial.suggest_categorical('activation_name',
                                                [nn.ReLU, nn.LeakyReLU,
                                                 nn.Tanh])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = BATCH_SIZE
    n_epochs = trial.suggest_int('n_epochs', 200, 1500)

    dae = DAE(
        input_size=X_train.shape[1],
        latent_dim=latent_dim,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        activation_name=activation_name,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    dae.fit(X_train, y_train, n_epochs=n_epochs, batch_size=batch_size, show_progress=False)

    y_pred = dae.predict(X_val)
    accuracy = (y_pred.flatten() == y_val).mean()
    print(accuracy)

    return accuracy


def optimize_model_for_dataset(dataset_name: str):
    # Load data
    data_path = "../data"
    df = pd.read_csv(f"{dataset_name}")
    x = df.copy()
    y = df['target']
    x = x.drop(['target'], axis=1)
    # x= stats.zscore(x,axis=1)
    column_types = {'age': 'positive', 'sex': 'bool', 'chest pain type': 'categorical', 'resting bp s': 'positive',
                    'cholesterol': 'positive', 'fasting blood sugar': 'bool', 'resting ecg': 'categorical',
                    'max heart rate': 'positive', 'exercise angina': 'bool', 'oldpeak': 'continuous',
                    'ST slope': 'categorical'}

    # One-hot encode the categorical columns
    categorical_columns = [col for col, col_type in column_types.items() if col_type == 'categorical']
    X = pd.get_dummies(x, columns=categorical_columns)
    X = X.astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Create a new function with X_train and y_train pre-filled
    objective_with_data = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=2)

    _best_params = study.best_params

    # Create directories
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    if not os.path.exists(f"{dataset_name}/{model_name}"):
        os.makedirs(f"{dataset_name}/{model_name}")

    # Save the best hyperparameters
    with open(f"{dataset_name}/{model_name}/best_params.json", 'w') as f:
        json.dump(_best_params, f)

    # Save the study

    fig_slice = optuna.visualization.plot_slice(study,
                                                params=['latent_dim',
                                                        'encoder_units_0', 'encoder_units_1',
                                                        'decoder_units_0', 'decoder_units_1',
                                                        'activation_name',
                                                        'dropout_rate', 'learning_rate',
                                                        'n_epochs'])
    fig_slice.write_image(f"{dataset_name}/{model_name}/slice_plot.png")

    fig_param_importances = optuna.visualization.plot_param_importances(study)
    fig_param_importances.write_image(f"{dataset_name}/{model_name}/param_importances_plot.png")

    best_encoder_units = (_best_params['encoder_units_0'], _best_params['encoder_units_1'])
    best_decoder_units = (_best_params['decoder_units_0'], _best_params['decoder_units_1'])

    print(
        f"best param: input_size={X_train.shape[1]},latent_dim={_best_params['latent_dim']},encoder_units={best_encoder_units},decoder_units={best_decoder_units},activation_name={_best_params['activation_name']},dropout_rate={_best_params['dropout_rate']},learning_rate={_best_params['learning_rate']}")

    non_categorical_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2]
    categorical_columns = [col for col in train.columns[:-1] if col not in non_categorical_columns]
    target_column = [train.columns[-1]]
    original_columns = train.columns
    scaler_pipeline = ColumnTransformer([
        ('scaler', StandardScaler(), non_categorical_columns)
    ], remainder='passthrough')

    train = pd.DataFrame(scaler_pipeline.fit_transform(train),
                         columns=non_categorical_columns + categorical_columns + target_column)
    # keep the order of columns
    train = train[original_columns]

    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]

    n_epochs = _best_params['n_epochs']
    batch_size = BATCH_SIZE

    dae.fit(X_train, y_train, n_epochs=n_epochs, batch_size=batch_size)

    # Save the model
    dae.save_checkpoint(f"{dataset_name}/{model_name}")


if __name__ == "__main__":
    # config_path = "../datasets/config.json"
    # # get datasets config
    # with open(config_path) as f:
    #     config = json.load(f)
    #
    # model_name = ModelFactory.DAE_NAME
    #
    # # get datasets from config
    # datasets: list = config['datasets']
    # # run experiment for each dataset
    # for dataset in tqdm(datasets):
    #     dataset_name_ = dataset['name']
    #     relative_path_ = dataset['relative_path']
    #     label_position_ = dataset['label_position']
    #     has_header_ = dataset['has_header']
    #     has_id_ = dataset['has_id']
    #     is_multy_class_ = dataset['is_multy_class']

        optimize_model_for_dataset(dataset_name='heart_statlog_cleveland_hungary_final.csv')
