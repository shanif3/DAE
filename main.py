import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import BinaryAUROC

from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from pytorch_lightning.loggers import WandbLogger
class DAENetwork(nn.Module):
    def __init__(self, input_size: int, latent_dim: int, hidden_dims: List[int],
                 dropout_rate: float, column_types: List[str], categorical_dims: Dict[str, int]):
        super(DAENetwork, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.column_types = column_types
        self.categorical_dims = categorical_dims


        # Encoder
        encoder_layers = []
        current_dim = input_size * 2  # *2 because we concatenate input with mask
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers, nn.Linear(current_dim, latent_dim))

        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers, nn.Linear(current_dim, input_size))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def encode(self, x, mask):
        x= x*mask
        concat = torch.cat([x, mask], 1)
        latent = self.encoder(concat)
        return latent

    def decode(self, z):
        reconstructed = self.decoder(z)

        # Split the reconstructed output into groups based on column types
        start = 0
        activated_outputs = []

        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                end = start + self.categorical_dims[col]
                cat_output = reconstructed[:, start:end]
                if cat_output.shape[1] != 0: # meaning that this column is starting the one hot category
                    activated_outputs.append(F.softmax(cat_output, dim=1))
                    start = end
            elif col_type == 'boolean':
                activated_outputs.append(torch.sigmoid(reconstructed[:, start:start + 1]))
                start += 1
            elif col_type == 'real_positive':
                activated_outputs.append(F.leaky_relu(reconstructed[:, start:start + 1]))
                start += 1
            else:  # getting identity() for col_type == 'real' and else
                activated_outputs.append(reconstructed[:, start:start + 1])
                start += 1

        return torch.cat(activated_outputs, dim=1)

    def forward(self, x, mask):
        latent = self.encode(x, mask)
        reconstructed = self.decode(latent)
        classification = self.classifier(latent)
        return reconstructed, classification

class DAELightning(pl.LightningModule):
    def __init__(self, input_size: int, latent_dim: int, hidden_dims: List[int],
                 dropout_rate: float, column_types: List[str],  categorical_dims: Dict[str,int], learning_rate: float):
        super(DAELightning, self).__init__()
        self.save_hyperparameters()
        self.model = DAENetwork(input_size, latent_dim, hidden_dims, dropout_rate, column_types, categorical_dims)
        self.learning_rate = learning_rate
        self.column_types= column_types
        self.categorical_dims = categorical_dims

        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    def training_step(self, batch, batch_idx):
        loss,y_true,y_pred= self._common_step(batch, batch_idx, "train")

        self.train_auroc.update(y_pred, y_true)
        return loss

    def validation_step(self, batch, batch_idx):
        loss,y_true,y_pred=self._common_step(batch, batch_idx, "val")
        self.val_auroc.update(y_pred, y_true)

    def test_step(self, batch, batch_idx):
        loss,y_true,y_pred= self._common_step(batch, batch_idx, "test")
        self.test_auroc.update(y_pred, y_true)

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        mask = self.generate_mask(x.shape)
        nan_mask= self.create_nan_mask(x)
        mask= self.combine_mask(nan_mask,mask)
        x= torch.nan_to_num(x,nan=0.0)
        reconstructed, classification = self.model(x, mask)

        loss = self.loss_function(reconstructed, x, classification, y)

        # Calculate AUC
        y_pred = classification.squeeze().cpu().detach().numpy()
        y_true = y.cpu().numpy()



        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, y,classification.squeeze()

    def on_train_epoch_end(self):
        train_auc = self.train_auroc.compute()
        self.log("train_auc", train_auc, on_step=False, on_epoch=True)
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        val_auc = self.val_auroc.compute()
        self.log("val_auc", val_auc, on_step=False, on_epoch=True)
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        test_auc = self.test_auroc.compute()
        self.log("test_auc", test_auc, on_step=False, on_epoch=True)
        self.test_auroc.reset()
    def loss_function(self, reconstructed, x, classification, y):
        start=0
        activated_outputs=[]
        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                end = start + self.categorical_dims[col]
                cat_output = reconstructed[:, start:end]

                if cat_output.shape[1] != 0:  # meaning that this column is starting the one hot category
                    activated_outputs.append(nn.CrossEntropyLoss()(cat_output, x[:,start:end]))
                    start = end
            elif col_type == 'boolean':
                activated_outputs.append(F.binary_cross_entropy(reconstructed[:, start:start + 1],x[:,start:start + 1]))
                start += 1
            else:
                activated_outputs.append(F.mse_loss(reconstructed[:, start:start + 1], x[:,start:start + 1]))
                start+=1


        reconstruction_loss= torch.sum(torch.stack(activated_outputs))
        # reconstruction_loss = F.mse_loss(reconstructed, x)
        classification_loss = F.binary_cross_entropy(classification, y.float().unsqueeze(1))
        return reconstruction_loss + classification_loss

    def generate_mask(self, shape):
        all_rows=[]
        for i in range(shape[0]):
            p= np.random.uniform(0.5,0.9)
            p=torch.full((shape[1],),p)
            mask=torch.bernoulli(p)
            all_rows.append(mask)
        mask= torch.stack(all_rows).to(self.device)
        # p = np.random.uniform(0.1, 0.9)
        # p = torch.full((shape[1],), p)
        # mask = torch.bernoulli(p).expand(shape[0], -1).to(self.device)
        return mask
    def create_nan_mask(self, x):
        mask= ~torch.isnan(x)
        mask= mask.int()
        return mask
    def combine_mask(self, nan_mask, mask):
        combined_mask=torch.where(nan_mask==1,mask,nan_mask)
        return combined_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def save_tensorboard_plots(log_dir: str, output_dir: str):
    """Reads TensorBoard logs and saves loss and AUC plots as images."""

    # Check if the output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get scalars for loss and AUC
    train_loss_values = event_acc.Scalars('train_loss')
    val_loss_values = event_acc.Scalars('val_loss')
    train_auc_values = event_acc.Scalars('train_auc')
    val_auc_values = event_acc.Scalars('val_auc')
    test_loss_values= event_acc.Scalars('test_loss')
    test_auc_values= event_acc.Scalars('test_auc')

    # Extract steps and values for losses
    train_steps = [x.step for x in train_loss_values]
    train_loss = [x.value for x in train_loss_values]
    val_steps = [x.step for x in val_loss_values]
    val_loss = [x.value for x in val_loss_values]
    test_loss= [x.value for x in test_loss_values]

    # Extract steps and values for AUC
    train_auc_steps = [x.step for x in train_auc_values]
    train_auc = [x.value for x in train_auc_values]
    val_auc_steps = [x.step for x in val_auc_values]
    val_auc = [x.value for x in val_auc_values]
    test_auc = [x.value for x in test_auc_values]


    # Plot loss
    plt.figure()
    plt.plot(train_steps, train_loss, label='Training Loss')
    plt.plot(val_steps, val_loss, label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))

    # Plot AUC
    plt.figure()
    plt.plot(train_auc_steps, train_auc, label='Training AUC')
    plt.plot(val_auc_steps, val_auc, label='Validation AUC')
    plt.axhline(y=test_auc, color='r', linestyle='--', label='Test AUC')
    plt.xlabel('Steps')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'auc.png'))

def prepare_data(file_path: str, target_column: str) -> Dict:
    df = pd.read_csv(file_path)
    # column_types = {
    #     'age': 'positive', 'sex': 'bool', 'chest pain type': 'categorical',
    #     'resting bp s': 'positive', 'cholesterol': 'positive', 'fasting blood sugar': 'bool',
    #     'resting ecg': 'categorical', 'max heart rate': 'positive', 'exercise angina': 'bool',
    #     'oldpeak': 'continuous', 'ST slope': 'categorical'
    # }
    y = df[target_column]
    X = df.drop([target_column], axis=1)

    # split into train valid and test
    X_TRAIN, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_TRAIN, y_train, test_size=0.25,
                                                      stratify=y_train)  # 0.25 x 0.8 = 0.2

    # Checking columns types based on train and valid data so we can catch most of the values
    bool_columns = [col for col in X_TRAIN.columns if X_TRAIN[col].nunique() == 2 and set(X_TRAIN[col].unique()) == {0, 1}]
    real_positive_columns = [col for col in X_TRAIN.columns if X_TRAIN[col].nunique() > 20 and (X_TRAIN[col] > 0).all()]
    real_columns = [col for col in X_TRAIN.columns if X_TRAIN[col].nunique() > 20 and not (X_TRAIN[col] > 0).all()]
    non_categorical_columns = bool_columns + real_columns + real_positive_columns
    # where the categorical has at least 2 unique values and maximum 20
    categorical_columns = [col for col in X_train.columns if col not in non_categorical_columns]

    scaler_pipeline = ColumnTransformer(
        transformers=[
            ('minmax_scaler', MinMaxScaler(), real_positive_columns),
            ('standard_scaler', StandardScaler(), real_columns),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough'  # Keep the rest of the columns (like boolean) as is
    )

    # Fit and transform the training data, and transform the validation and test data
    X_train_transformed = scaler_pipeline.fit_transform(X_train)
    X_val_transformed = scaler_pipeline.transform(X_val)
    X_test_transformed = scaler_pipeline.transform(X_test)

    # Get the correct feature names after transformations
    feature_names = scaler_pipeline.get_feature_names_out()

    # Create DataFrames with the correct column names
    X_train = pd.DataFrame(X_train_transformed.toarray(), columns=feature_names)
    X_val = pd.DataFrame(X_val_transformed.toarray(), columns=feature_names)
    X_test = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)

    # Ensure that all sets have the same columns after one-hot encoding
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Create a dictionary to store the type of each column
    column_types = {}
    for col in X_train.columns:
        if any(orig_col in col for orig_col in bool_columns):
            column_types[col] = 'boolean'
        elif any(orig_col in col for orig_col in real_positive_columns):
            column_types[col] = 'real_positive'
        elif any(orig_col in col for orig_col in real_columns):
            column_types[col] = 'real'
        elif any(orig_col in col for orig_col in categorical_columns):
            column_types[col] = 'categorical'

    # Calculate categorical_dims
    categorical_dims = {}
    onehot_encoder = scaler_pipeline.named_transformers_['onehot']
    for orig_col, encoded_cols in zip(categorical_columns, onehot_encoder.categories_):
        n_categories = len(encoded_cols)
        if n_categories>=2:
            for i in range(n_categories):
                item_value = encoded_cols[i]
                col_name = f"onehot__{orig_col}_{item_value}"
                categorical_dims[col_name]= n_categories if i==0 else 0 # putting 0 if the one hot column is not the column that start the one hot, if it is I will put the number of columns that encoded in the same category.

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'column_types': column_types,
        'categorical_dims': categorical_dims,
        'input_size': X_train.shape[1]
    }


def main():
    # Prepare data
    # data = prepare_data(r"Data/heart_statlog_cleveland_hungary_final.csv",
    #                     target_column='target')

    data = prepare_data(r"see.csv",
                        target_column='target')



    num_workers = 4
    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(data['X_train'].values), torch.FloatTensor(data['y_train'].values))
    val_dataset = TensorDataset(torch.FloatTensor(data['X_val'].values), torch.FloatTensor(data['y_val'].values))
    test_dataset = TensorDataset(torch.FloatTensor(data['X_test'].values), torch.FloatTensor(data['y_test'].values))

    BATCH_SIZE= 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model hyperparameters
    input_size = data['input_size']
    latent_dim = 100
    hidden_dims = [128, 64]
    # hidden_dims = [265, 128, 64]
    dropout_rate = 0.5 # changed from 0.3 bc overfit
    learning_rate = 1e-3

    # Create model
    model = DAELightning(
        input_size=input_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        column_types=data['column_types'],
        categorical_dims=data['categorical_dims'],
        learning_rate=learning_rate
    )
    # Initialize TensorBoard logger
    logger = TensorBoardLogger("calude/tb_logs", name="DAE_model")
    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='DAE-heart_statlog')

    # add your batch size to the wandb config
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.000002,
        patience=20,
        verbose=True,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=200,
        logger=[logger,wandb_logger],
        callbacks=[early_stop_callback, checkpoint_callback],
        # callbacks=[ checkpoint_callback],
        devices=1,
        enable_progress_bar=True,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, dataloaders=test_loader)


    # Load best model and make predictions
    best_model = DAELightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()

    with torch.no_grad():
        x_test = torch.FloatTensor(data['X_test'].values).to(best_model.device)
        x_test= torch.nan_to_num(x_test,nan=0.0)
        nan_mask= best_model.create_nan_mask(x_test)
        # mask = best_model.generate_mask(x_test.shape)
        # mask= best_model.combine_mask(nan_mask,mask)
        reconstructed, predictions = best_model.model(x_test, nan_mask)
        predictions = (predictions > 0.5).squeeze().cpu().numpy()

    print(f"Accuracy: {accuracy_score(data['y_test'], predictions):.4f}")
    print(f"F1 Score: {f1_score(data['y_test'], predictions):.4f}")
    print(f"Precision: {precision_score(data['y_test'], predictions):.4f}")
    print(f"Recall: {recall_score(data['y_test'], predictions):.4f}")
    if len(set(data['y_test'])) > 1:  # Ensure both classes are present to calculate AUC
        auc_score = roc_auc_score(data['y_test'], predictions)
        print(f"AUC: {auc_score:.4f}")
    reconstructed=reconstructed.cpu().numpy()
    reconstructed= pd.DataFrame(reconstructed, columns=data['X_test'].columns)
    reconstructed.to_csv('reconstructed.csv')
    data['y_test'].to_csv('y_test.csv')
    # Save plots after training
    save_tensorboard_plots(log_dir=logger.log_dir, output_dir='calude/plots')

if __name__ == "__main__":
    main()
