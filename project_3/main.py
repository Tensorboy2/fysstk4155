import torch
import torch.nn as nn
import torch.optim as optim
from src.models.RNN import RNN
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

data_pth = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data_time.csv'

def main_rnn():
    sequence_length = 10  # Number of time steps in each sequence
    train_split = 0.8     # 20% for training

    # Load the scaled dataset
    data = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data_time.csv')
    data_values = data.iloc[:, 1:].values  # Exclude the 'Date' column

    # Split into training and test sets
    n_train = int(len(data_values) * train_split)
    train_data = data_values[:n_train]
    test_data = data_values[n_train:]

    # Create sequences for training
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])  # Target is the next step
        return np.array(X), np.array(y)

    # Generate sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    # Model parameters
    input_size = X_train.shape[2]  # Number of features per timestep
    hidden_size = 256           # Can be adjusted based on experiments
    output_size = y_train.shape[1] # Number of target features
    num_layers = 8           # Number of RNN layers (adjustable)

    # Create the model
    model = RNN(input_size, hidden_size, output_size, num_layers)

    # Example training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    train_loss_history = []
    test_loss_history = []
    train_mse_history = []
    test_mse_history = []
    train_r2_history = []
    test_r2_history = []
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_mse = mean_squared_error(y_train.cpu().numpy(), outputs.cpu().detach().numpy())
        train_r2 = r2_score(y_train.cpu().numpy(), outputs.cpu().detach().numpy())

        # Calculate test MSE and R²
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_mse = mean_squared_error(y_test.cpu().numpy(), test_outputs.cpu().numpy())
            test_r2 = r2_score(y_test.cpu().numpy(), test_outputs.cpu().numpy())

        # Store metrics
        train_loss_history.append(loss.item())
        test_loss_history.append(test_mse)  # You can also use test_loss if you have that
        train_mse_history.append(train_mse)
        test_mse_history.append(test_mse)
        train_r2_history.append(train_r2)
        test_r2_history.append(test_r2)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
    
    # Save loss and accuracy histories to a DataFrame
    metrics_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_loss_history,
        'Test Loss': test_loss_history,
        'Train MSE': train_mse_history,
        'Test MSE': test_mse_history,
        'Train R2': train_r2_history,
        'Test R2': test_r2_history
    })

    # Save metrics to CSV
    metrics_df.to_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/rnn_training_metrics.csv', index=False)
    print("Training metrics saved to rnn_training_metrics.csv")

    # Predictions and targets
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()  # Predictions on test set
        targets = y_test.cpu().numpy()  # Ground truth targets

    # Create a DataFrame with predictions and targets
    results_df = pd.DataFrame({
        'Target': targets[:, 0],  # Adjust index if needed
        'Prediction': predictions[:, 0]  # Adjust index if needed
    })

    # Save predictions to CSV
    results_df.to_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/rnn_predictions.csv', index=False)
    print("Predictions saved to rnn_predictions.csv")

if __name__ == '__main__':
    main_rnn()