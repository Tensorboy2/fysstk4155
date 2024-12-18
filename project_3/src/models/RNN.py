import torch
from sklearn.metrics import r2_score
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers =1,sequence_length = 15):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_size = sequence_length
        self.rnn = torch.nn.RNN(input_size = input_size, hidden_size = hidden_size,
                                 num_layers=num_layers, batch_first=True, dropout=0, nonlinearity='relu')
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h_0)
        out = out[:, -1, :]
        out = self.fc(out)  
        return out

def train_rnn(model, 
              train_loader, 
              val_loader, 
              num_epochs=100, 
              learning_rate = 0.001,
              device="cuda" if torch.cuda.is_available() else "cpu"
            ):
    '''
    Training function for RNN.
    '''
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_r2": [],
        "val_r2": [],
        "learning_rates": [],
        "gradient_norms": []
    }

    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        all_targets = []
        all_predictions = []

        for x, y in train_loader:
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            metrics["gradient_norms"].append(total_norm)
        
            optimizer.step()
            epoch_train_loss += loss.item()

            all_targets.append(y.detach().cpu().squeeze())
            all_predictions.append(output.detach().cpu().squeeze())


        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        train_r2 = r2_score(all_targets, all_predictions)
        
        epoch_train_loss /= len(train_loader)
        metrics["train_loss"].append(epoch_train_loss)
        metrics["train_r2"].append(train_r2)
        
        current_lr = optimizer.param_groups[0]["lr"]
        metrics["learning_rates"].append(current_lr)

        model.eval()
        val_loss = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.unsqueeze(-1)
                y = y.unsqueeze(-1)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                all_targets.append(y.detach().cpu().squeeze())
                all_predictions.append(output.detach().cpu().squeeze())
                
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        val_r2 = r2_score(all_targets, all_predictions)
        
        val_loss /= len(val_loader)
        metrics["val_loss"].append(val_loss)
        metrics["val_r2"].append(val_r2)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train R²: {train_r2:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f} | "
              f"Grad Norm: {total_norm:.4f} | ")
        
    return metrics

