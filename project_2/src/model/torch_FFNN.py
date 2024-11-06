'''
This module includes a FFNN that will be written using pytorch and tested against out own FFNN.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class torch_ffnn(nn.Module):
    '''
    Feed Forward Neural Network written using torch.
    '''
    def __init__(self,input_size, hidden_sizes, output_size, classify=False):
        super(torch_ffnn, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        if classify:
            self.layers.append(nn.Sigmoid())

    def forward(self,x):
        '''
        The feed forward method
        '''
        for layer in self.layers:
            x = layer(x)
        return x

    def train_NN(self, criterion, optimizer, x_train, y_train, epochs):
        '''
        Training function for torch FFNN.
        '''
        self.train()
        loss_history = []
        
        for epoch in range(epochs):
            running_loss = 0.0

            for x, y in zip(x_train,y_train):
                optimizer.zero_grad()
                
                outputs = self.forward(x)
                # print(outputs,y)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            loss_history.append(running_loss/len(y_train))
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(y_train):.8f}")
        return np.array(loss_history)


# if __name__ == '__main__':
    # Setting parameters
    # input_size_ = torch.randint(10,100,(1,))
    # hidden_size_ = torch.randint(10,100,(1,))
    # output_size_ = torch.randint(10,100,(1,))
    # num_hidden_layers_ = torch.randint(10,100,(1,))
    # epochs = 50
    # learing_rate = 0.1
    # batch_size = 30

    # # Model instance
    # model = torch_ffnn(input_size=input_size_,
    #                    hidden_size=hidden_size_,
    #                    num_hidden_layers=num_hidden_layers_,
    #                    output_size=output_size_)

    # # cost function:
    # criterion = nn.MSELoss()

    # # Learning strategy:
    # optimizer = optim.SGD(model.parameters(), lr=learing_rate, momentum=0.9)

    # # Random example data:
    # train_data = torch.randn(10, input_size_)
    # train_targets = torch.randn(10, output_size_)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=torch.utils.data.TensorDataset(train_data, train_targets),
    #     batch_size=batch_size,
    #     shuffle=True
    # )

    # # Training model:
    # train(model, criterion, optimizer, train_loader, epochs)
