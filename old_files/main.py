from  see import *
from sklearn.preprocessing import StandardScaler
import pandas

def main():
    # Example data: Random data for demonstration purposes
    df= pd.read_csv("../Data/heart_statlog_cleveland_hungary_final.csv")
    x = df.copy()
    y = df['target']
    x=x.drop(['target'],axis=1)
    # x= stats.zscore(x,axis=1)
    column_types= {'age':'positive','sex':'bool','chest pain type':'categorical','resting bp s':'positive','cholesterol':'positive','fasting blood sugar':'bool','resting ecg':'categorical','max heart rate':'positive','exercise angina':'bool','oldpeak':'continuous','ST slope':'categorical'}

    # One-hot encode the categorical columns
    categorical_columns = [col for col, col_type in column_types.items() if col_type == 'categorical']
    numerical_columns= [col for col, col_type in column_types.items()  if col_type =='positive'   ]
    x = pd.get_dummies(x, columns=categorical_columns)
    x = x.astype(int)

    # Apply Z-score normalization to numerical columns
    scaler = StandardScaler()
    x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

    column_types= list(column_types.values())

    # Define model parameters
    input_size = x.shape[1]
    latent_dim = 100
    encoder_units = (128, 64)
    decoder_units = (64, 128)
    activation_name = nn.ReLU
    dropout_rate = 0.3
    learning_rate = 1e-3
    n_epochs = 1000
    batch_size = 32

    # Instantiate the model
    dae = DAE(input_size=input_size,
              latent_dim=latent_dim,
              encoder_units=encoder_units,
              decoder_units=decoder_units,
              activation_name=activation_name,
              dropout_rate=dropout_rate,
              learning_rate=learning_rate,column_types=column_types)


    # Train the model
    dae.fit(x, y, n_epochs=n_epochs, batch_size=batch_size)

    # Example of prediction
    predictions, reconstructed_df = dae.predict(x)
    probabilities = dae.predict_proba(x, mask_vector=np.ones(input_size))

    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")

    # Save the model checkpoint
    dae.save_checkpoint('checkpoints')

    # Load the model checkpoint
    loaded = dae.load_checkpoint('checkpoints')
    if loaded:
        print("Model loaded successfully.")

if __name__ == "__main__":
    main()
