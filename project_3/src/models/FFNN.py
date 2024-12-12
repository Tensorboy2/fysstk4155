import torch
import torch.optim as optim
from sklearn.metrics import r2_score


class FFNN(torch.nn.Module):
    def __init__(self, input_seq_len, hidden_sizes, target_seq_len):
        super(FFNN, self).__init__()
        in_features = input_seq_len

        layers = []
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(in_features, hidden_size))
            layers.append(torch.nn.Tanh())
            in_features = hidden_size  

        layers.append(torch.nn.Linear(in_features, target_seq_len))
        
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
    
        return self.network(x)
    
def train_ffnn(model, 
               train_loader, 
               test_loader, 
               num_epochs=100, 
               batch_size=32, 
               learning_rate=0.001):
    '''
    Training function for FFNN.
    '''

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

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

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            metrics["gradient_norms"].append(total_norm)
        
            optimizer.step()
            epoch_train_loss += loss.item()

            all_targets.append(labels.detach().cpu().squeeze())
            all_predictions.append(outputs.detach().cpu().squeeze())

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
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_targets.append(labels.detach().cpu().squeeze())
                all_predictions.append(outputs.detach().cpu().squeeze())

        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        val_r2 = r2_score(all_targets, all_predictions)
        
        val_loss /= len(test_loader)
        metrics["val_loss"].append(val_loss)
        metrics["val_r2"].append(val_r2)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train R²: {train_r2:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f} | "
              f"Grad Norm: {total_norm:.4f} | ")
    
    return metrics