import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use LSTM or GRU instead of vanilla RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer(s)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)  # LSTM returns output and (hidden state, cell state)

        # Take the output at the last timestep
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Pass through the fully connected layer
        out = self.fc(out)  # Shape: (batch_size, output_size)
        return out
