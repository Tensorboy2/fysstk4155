import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.models.RNN import RNN, train_rnn
from src.models.FFNN import FFNN, train_ffnn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# Paths to data:
h_o_data_path = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/harmonic_occilator_data/h_o_data.csv'
stock_data_path = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data/stock_data_time.csv'


def main_h_o_FFNN():
    '''
    Main function for doing feed forward analysis of the harmonic oscillator.
    '''
    data = pd.read_csv(h_o_data_path)

    input_seq_len = 1
    hidden_sizes = [64,128,64]
    target_seq_len = 1
    num_epochs = 400
    batch_size = 32

    X = torch.tensor(data['time'].values, dtype=torch.float32).view(-1,1)
    y = torch.tensor(data['angle'].values, dtype=torch.float32).view(-1,1)

    # train_size = 0.8

    train_dataset = TensorDataset(X[:150], y[:150])
    test_dataset = TensorDataset(X[150:], y[150:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ffnn_model = FFNN(input_seq_len,hidden_sizes,target_seq_len)
    metrics = train_ffnn(ffnn_model, train_loader, test_loader, num_epochs=num_epochs, batch_size=batch_size)

    plt.figure(figsize=(6, 6))
    plt.plot(metrics["gradient_norms"], label="Gradient norms")
    plt.xlabel("Index",fontsize=20)
    plt.ylabel("Gradient Norm",fontsize=20)
    plt.legend()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_ffnn_grad_norm.pdf')

    plt.figure(figsize=(6, 6))
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_ffnn_loss.pdf')

    plt.figure(figsize=(7, 6))
    plt.plot(metrics["train_r2"], label="Train R2")
    plt.plot(metrics["val_r2"], label="Validation R2")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("R2 Score",fontsize=20)
    plt.legend()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_ffnn_R2.pdf')

    # Extrapolation
    num_extrapolations = 20
    all_extrapolations = []

    for _ in range(num_extrapolations):
        predictions = []
        ffnn_model = FFNN(input_seq_len,hidden_sizes,target_seq_len)

        _ = train_ffnn(ffnn_model, train_loader, test_loader, num_epochs=400, batch_size=batch_size)
    
        ffnn_model.eval()
        with torch.no_grad():
            for inputs, labels in test_dataset:
                output = ffnn_model(inputs)
                output = output.detach().squeeze()
                predictions.append(output)
            all_extrapolations.append(predictions)
    all_extrapolations = np.array(all_extrapolations)
    mean_extrapolation = all_extrapolations.mean(axis=0)
    std_extrapolation = all_extrapolations.std(axis=0)
    time_indices = np.arange(150, 251)
    intervals = [1, 2, 3]
    colors = sns.color_palette("magma", len(intervals))
    plt.figure(figsize=(6, 6))
    for i, sigma in enumerate(intervals):
        lower_bound = mean_extrapolation - sigma * std_extrapolation
        upper_bound = mean_extrapolation + sigma * std_extrapolation
        plt.fill_between(
            time_indices, lower_bound, upper_bound, color=colors[i], alpha=0.4,
            label=f"{sigma} Std Dev"
        )
    data = {
        "train": data[data.columns[2]].values[:150],
        "test": data[data.columns[2]].values[150:]
    }   
    plt.plot(time_indices, mean_extrapolation, color="black", label="Mean Prediction", linewidth=2)
    plt.plot(np.arange(len(data["train"])), data["train"], c='b', label="Train Data", linestyle="--")
    plt.plot(
        np.arange(len(data["train"]), len(data["train"]) + len(data["test"])),
        data["test"], c='g', label="Test Data", linestyle="--"
    )
    plt.xlabel("Time/Index",fontsize=20)
    plt.ylabel("Prediction",fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_ffnn_extrap.pdf')



class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main_h_o_RNN():
    '''
    Main function for doing extrapolation of the harmonic oscillator using the RNN.
    '''
    print(f'Loading data from: '+ h_o_data_path)
    data = pd.read_csv(h_o_data_path)
    data_length = len(data)
    input_size = 1
    sequence_length = 16 
    batch_size = 32
    hidden_sizes = 64
    num_layers = 1
    target_seq_len = 1  
    num_epochs = 100
    learning_rate = 0.001
    train_size = 0.75

    x = data[data.columns[2]].values[:150]
    dataset = TimeSeriesDataset(x, sequence_length)
    train_len = int(len(dataset) * train_size)
    val_len = len(dataset) - train_len

    train_dataset, val_dataset = train_test_split(dataset, train_size= train_size,test_size=0.2 )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    rnn_model = RNN(input_size,hidden_sizes,target_seq_len,num_layers,sequence_length)

    metrics = train_rnn(rnn_model, train_loader, val_loader, num_epochs=num_epochs,learning_rate=learning_rate)


    plt.figure(figsize=(6, 6))
    plt.plot(metrics["gradient_norms"], label="Gradient norms")
    plt.xlabel("Index",fontsize=20)
    plt.ylabel("Gradient Norm",fontsize=20)
    plt.legend()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_rnn_grad_norm.pdf')

    plt.figure(figsize=(6, 6))
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_rnn_loss.pdf')

    plt.figure(figsize=(7, 6))
    plt.plot(metrics["train_r2"], label="Train R2")
    plt.plot(metrics["val_r2"], label="Validation R2")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("R2 Score",fontsize=20)
    plt.legend()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_rnn_R2.pdf')
    
    # Extrapolation 
    num_future = 100
    num_extrapolations = 20
    all_extrapolations = []

    for _ in range(num_extrapolations):
        predicted_extrapolation = []
        extrapolation_input = torch.tensor(x[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Last sequence
        rnn_model = RNN(1,hidden_sizes,target_seq_len,num_layers,sequence_length)
        _ = train_rnn(rnn_model, train_loader, val_loader, num_epochs=100,learning_rate=learning_rate)    
        rnn_model.eval()
        with torch.no_grad():
            for _ in range(num_future):
                output = rnn_model(extrapolation_input)
                predicted_extrapolation.append(output.item())
                extrapolation_input = torch.cat((extrapolation_input[:, 1:, :], output.unsqueeze(-1)), dim=1)
            all_extrapolations.append(predicted_extrapolation)
    all_extrapolations = np.array(all_extrapolations)  
    mean_extrapolation = all_extrapolations.mean(axis=0)
    std_extrapolation = all_extrapolations.std(axis=0)
    time_indices = np.arange(151, 251)
    intervals = [1, 2, 3]
    colors = sns.color_palette("magma", len(intervals))
    plt.figure(figsize=(6, 6))
    for i, sigma in enumerate(intervals):
        lower_bound = mean_extrapolation - sigma * std_extrapolation
        upper_bound = mean_extrapolation + sigma * std_extrapolation
        plt.fill_between(
            time_indices, lower_bound, upper_bound, color=colors[i], alpha=0.4,
            label=f"{sigma} Std Dev"
        )
    data = {
        "train": data[data.columns[2]].values[:150],
        "test": data[data.columns[2]].values[150:]
    }   
    plt.plot(time_indices, mean_extrapolation, color="black", label="Mean Prediction", linewidth=2)
    plt.plot(np.arange(len(data["train"])), data["train"], c='b', label="Train Data", linestyle="--")
    plt.plot(
        np.arange(len(data["train"]), len(data["train"]) + len(data["test"])),
        data["test"], c='g', label="Test Data", linestyle="--"
    )
    plt.xlabel("Time/Index",fontsize=20)
    plt.ylabel("Prediction",fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/ho_rnn_extrap.pdf')
    

    

def main__stock_FFNN():
    '''
    Main function for extrapolation of the FFNN on the stock data.
    '''
    data = pd.read_csv(stock_data_path)
    X = torch.tensor(np.arange(len(data[data.columns[0]])), dtype=torch.float32).view(-1,1)

    for k in range(1,6):
        stock_name = data.columns[k]
        y = torch.tensor(data[data.columns[k]].values, dtype=torch.float32).view(-1,1)

        input_seq_len = 1  
        hidden_sizes = [64,128,64]
        target_seq_len = 1  
        num_epochs = 400
        batch_size = 32
        train_size = 0.8
        train_dataset = TensorDataset(X[:200], y[:200])
        test_dataset = TensorDataset(X[200:], y[200:])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        ffnn_model = FFNN(input_seq_len,hidden_sizes,target_seq_len)
        metrics = train_ffnn(ffnn_model, train_loader, test_loader, num_epochs=num_epochs, batch_size=batch_size)

        plt.figure(figsize=(6, 6))
        plt.plot(metrics["gradient_norms"], label="Gradient norms")
        plt.xlabel("Index",fontsize=20)
        plt.ylabel("Gradient Norm",fontsize=20)
        plt.legend()
        plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+'_ffnn_grad_norm.pdf')

        plt.figure(figsize=(6, 6))
        plt.plot(metrics["train_loss"], label="Train Loss")
        plt.plot(metrics["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch",fontsize=20)
        plt.ylabel("Loss",fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+'_ffnn_loss.pdf')

        plt.figure(figsize=(7, 6))
        plt.plot(metrics["train_r2"], label="Train R2")
        plt.plot(metrics["val_r2"], label="Validation R2")
        plt.xlabel("Epoch",fontsize=20)
        plt.ylabel("R2 Score",fontsize=20)
        plt.legend()
        plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+'_ffnn_R2.pdf')

        # Extrapolation
        # num_future = 100
        num_extrapolations = 20 
        all_extrapolations = []

        for _ in range(num_extrapolations):
            predictions = []
            ffnn_model = FFNN(input_seq_len,hidden_sizes,target_seq_len)

            _ = train_ffnn(ffnn_model, train_loader, test_loader, num_epochs=400, batch_size=batch_size)
        
            ffnn_model.eval()
            with torch.no_grad():
                for inputs, labels in test_dataset:
                    output = ffnn_model(inputs)
                    output = output.detach().squeeze()
                    predictions.append(output)
                all_extrapolations.append(predictions)
        all_extrapolations = np.array(all_extrapolations)
        mean_extrapolation = all_extrapolations.mean(axis=0)
        std_extrapolation = all_extrapolations.std(axis=0)
        time_indices = np.arange(200, 251)
        intervals = [1, 2, 3]
        colors = sns.color_palette("magma", len(intervals))
        plt.figure(figsize=(6, 6))
        for i, sigma in enumerate(intervals):
            lower_bound = mean_extrapolation - sigma * std_extrapolation
            upper_bound = mean_extrapolation + sigma * std_extrapolation
            plt.fill_between(
                time_indices, lower_bound, upper_bound, color=colors[i], alpha=0.4,
                label=f"{sigma} Std Dev"
            )
        plot_data = {
            "train": data[data.columns[k]].values[:200],
            "test": data[data.columns[k]].values[200:]
        }   
        plt.plot(time_indices, mean_extrapolation, color="black", label="Mean Prediction", linewidth=2)
        plt.plot(np.arange(len(plot_data["train"])), plot_data["train"], c='b', label="Train Data", linestyle="--")
        plt.plot(
            np.arange(len(plot_data["train"]), len(plot_data["train"]) + len(plot_data["test"])),
            plot_data["test"], c='g', label="Test Data", linestyle="--"
        )
        plt.xlabel("Time/Index",fontsize=20)
        plt.ylabel("Prediction",fontsize=20)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+f'_ffnn_extrap.pdf')


def main__stock_RNN():
    '''
    Main function for extrapolating using the RNN on the stck data.
    '''
    # Load dataset
    data = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data/stock_data_time.csv')
    for k in range(1,6):
        x = data[data.columns[k]].values[:200]
        stock_name = data.columns[k]

        sequence_length = 64
        batch_size = 32
        hidden_size = 64
        num_layers = 4
        target_seq_len = 1
        num_epochs = 200
        learning_rate = 0.001
        train_size = 0.8

        dataset = TimeSeriesDataset(x, sequence_length)
        train_len = int(len(dataset) * train_size)
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = train_test_split(dataset, train_size=train_size, test_size=0.2)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        rnn_model = RNN(input_size=1, hidden_size=hidden_size, output_size=target_seq_len, 
                        num_layers=num_layers,sequence_length=sequence_length)

        metrics = train_rnn(rnn_model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)
        plt.figure(figsize=(6, 6))
        plt.plot(metrics["gradient_norms"], label="Gradient norms")
        plt.xlabel("Index",fontsize=20)
        plt.ylabel("Gradient Norm",fontsize=20)
        plt.legend()
        plt.savefig(f'/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+f'_rnn_grad_norm.pdf')


        plt.figure(figsize=(6, 6))
        plt.plot(metrics["train_loss"], label="Train Loss")
        plt.plot(metrics["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch",fontsize=20)
        plt.ylabel("Loss",fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+f'_rnn_loss.pdf')

        plt.figure(figsize=(7, 6))
        plt.plot(metrics["train_r2"], label="Train R2")
        plt.plot(metrics["val_r2"], label="Validation R2")
        plt.xlabel("Epoch",fontsize=20)
        plt.ylabel("R2 Score",fontsize=20)
        plt.legend()
        plt.savefig(f'/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+f'_rnn_R2.pdf')

        # Extrapolation (predict future data beyond training)
        num_future = 50  
        num_extrapolations = 20 
        all_extrapolations = []

        for _ in range(num_extrapolations):
            predicted_extrapolation = []
            extrapolation_input = torch.tensor(x[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Last sequence
            rnn_model = RNN(1,hidden_size,target_seq_len,num_layers,sequence_length)
            _ = train_rnn(rnn_model, train_loader, val_loader, num_epochs=100,learning_rate=learning_rate)    
            rnn_model.eval()
            with torch.no_grad():
                for _ in range(num_future):
                    output = rnn_model(extrapolation_input)
                    predicted_extrapolation.append(output.item())
                    extrapolation_input = torch.cat((extrapolation_input[:, 1:, :], output.unsqueeze(-1)), dim=1)
                all_extrapolations.append(predicted_extrapolation)
        all_extrapolations = np.array(all_extrapolations)  
        mean_extrapolation = all_extrapolations.mean(axis=0)
        std_extrapolation = all_extrapolations.std(axis=0)
        time_indices = np.arange(201, 251)
        intervals = [1, 2, 3]
        colors = sns.color_palette("magma", len(intervals))
        plt.figure(figsize=(6, 6))
        for i, sigma in enumerate(intervals):
            lower_bound = mean_extrapolation - sigma * std_extrapolation
            upper_bound = mean_extrapolation + sigma * std_extrapolation
            plt.fill_between(
                time_indices, lower_bound, upper_bound, color=colors[i], alpha=0.4,
                label=f"{sigma} Std Dev"
            )
        plot_data = {
            "train": data[data.columns[k]].values[:200],
            "test": data[data.columns[k]].values[200:]
        }   
        plt.plot(time_indices, mean_extrapolation, color="black", label="Mean Prediction", linewidth=2)
        plt.plot(np.arange(len(plot_data["train"])), plot_data["train"], c='b', label="Train Data", linestyle="--")
        plt.plot(
            np.arange(len(plot_data["train"]), len(plot_data["train"]) + len(plot_data["test"])),
            plot_data["test"], c='g', label="Test Data", linestyle="--"
        )
        plt.xlabel("Time/Index",fontsize=20)
        plt.ylabel("Prediction",fontsize=20)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/'+stock_name+f'_rnn_extrap.pdf')



if __name__ == '__main__':
    main_h_o_FFNN()
    main__stock_FFNN()
    main_h_o_RNN()
    main__stock_RNN()