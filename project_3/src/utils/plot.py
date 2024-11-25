import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="darkgrid", palette="muted")
def plot_stock_prices(data,tickers):
    data_melted = data.melt(id_vars=['Date'], 
                                   value_vars=tickers, 
                                   var_name='Stock', 
                                   value_name='Price')

    # Plot the stock prices over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data_melted, x='Date', y='Price', hue='Stock')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(title='Stock Ticker', title_fontsize=12, fontsize=10, loc='upper left')
    dates = data['Date']
    plt.xticks(dates[::10], rotation=90)
    plt.tight_layout()
    plt.show()


def plot_predictions(data):
    
    # Plot actual vs. predicted
    plt.figure(figsize=(12, 6))
    plt.plot(data['Target'], label='Actual', color='blue')
    plt.plot(data['Prediction'], label='Predicted', color='red', linestyle='--')

    # Format plot
    plt.title('RNN Predictions vs Actual Targets')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_training_progress(data):
    loss_df = data.melt(id_vars="Epoch", value_vars=["Train Loss", "Test Loss"], 
                        var_name="Loss Type", value_name="Loss")
    mse_df = data.melt(id_vars="Epoch", value_vars=["Train MSE", "Test MSE"], 
                        var_name="MSE Type", value_name="MSE")
    r2_df = data.melt(id_vars="Epoch", value_vars=["Train R2", "Test R2"], 
                      var_name="R2 Type", value_name="R2")

    # Create subplots: one for loss, one for MSE, and one for R²
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 10))

    # Plot Loss
    sns.lineplot(data=loss_df, x="Epoch", y="Loss", hue="Loss Type", marker="o", ax=ax1)
    ax1.set_title("Training and Test Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(title="Loss Type")
    ax1.grid(True)

    # Plot MSE
    sns.lineplot(data=mse_df, x="Epoch", y="MSE", hue="MSE Type", marker="o", ax=ax2)
    ax2.set_title("Training and Test MSE Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.legend(title="MSE Type")
    ax2.grid(True)

    # Plot R²
    sns.lineplot(data=r2_df, x="Epoch", y="R2", hue="R2 Type", marker="o", ax=ax3)
    ax3.set_title("Training and Test R2 Over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("R2")
    ax3.legend(title="R2 Type")
    ax3.grid(True)

    # Adjust layout to make space between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

if __name__ == '__main__':
    # data_pth = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data_time.csv'
    # tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    # data = pd.read_csv(data_pth)
    # plot_stock_prices(data, tickers)

    # data = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/rnn_predictions.csv')
    # plot_predictions(data)

    train_data = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/rnn_training_metrics.csv')
    plot_training_progress(train_data)
