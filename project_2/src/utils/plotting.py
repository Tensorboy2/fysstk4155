import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heat_plot(data, filename, title="Heatmap", cmap="viridis", colorbar=True, annot=False):
    """
    Saves a heatmap of the given data with customizable style.
    
    Parameters:
        data (array-like): Data for the heatmap.
        filename (str): Filename to save the heatmap.
        title (str): Title of the plot.
        cmap (str): Color map for the heatmap.
        colorbar (bool): Show color bar.
        annot (bool): Annotate cells with data values.
    """
    plt.figure(figsize=(20, 12))
    
    # Create the heatmap
    sns.heatmap(data, cmap=cmap, cbar=colorbar, annot=annot, fmt=".4f")
    
    # plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_path+filename}.png", dpi=300)
    plt.close()

def loss_plot(data, filename, title="Loss Over Epochs", xlabel="Epochs", ylabel="Loss"):
    """
    Plots the loss over epochs for each unique combination of learning rate and L2 penalty
    in separate subplots and saves the figure as a PNG file.

    Parameters:
        data (pd.DataFrame): DataFrame containing loss data with columns for "Learning Rate",
                             "L2 Penalty", "Epoch", "Loss", and optionally others like "Optimizer" or "Mini_batch".
        filename (str): Filename to save the plot.
        title (str): Title of the overall plot.
        xlabel (str): X-axis label for each subplot.
        ylabel (str): Y-axis label for each subplot.
    """
    data['Optimizer_Config'] = data['Optimizer'] + " | " + data['Momentum'] + " | " + data['Mini_batch']
    
    unique_configs = data[['Learning Rate', 'L2 Penalty']].drop_duplicates()
    
    n_configs = unique_configs.shape[0]
    n_cols = 3  
    n_rows = (n_configs + n_cols - 1) // n_cols  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharey=True)
    axes = axes.flatten()  

    for idx, (lr, l2) in enumerate(unique_configs.values):
        config_data = data[(data['Learning Rate'] == lr) & (data['L2 Penalty'] == l2)]
        
        sns.lineplot(
            data=config_data,
            x="Epoch",
            y="Loss",
            hue="Optimizer_Config",
            # style="Mini_batch",
            markers=True,
            ax=axes[idx]
        )
        
        axes[idx].set_title(f"LR: {lr}, L2: {l2}", fontsize=14, fontweight='bold')
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(ylabel)
    
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(f"{plot_path+filename}.png", dpi=300)
    plt.close()

plot_path = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_2/src/utils/'

def activation_plot(data, filename, title="Loss Over Epochs", xlabel="Epochs", ylabel="Loss"):
    plt.figure(figsize=(10, 8))
    sns.lineplot(
            data=data,
            x="Epoch",
            y="Loss",
            hue="Activation_function",
            markers=True
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.title(title, fontsize=16, fontweight='bold')    
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(f"{plot_path+filename}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # data = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_2/src/utils/different_activation_functions.csv')
    # activation_plot(data,'Different_activation_functions')
    # print("Plots saved as 'Different_activation_functions.png'")


    # data = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/results_regression.csv')
    # NN = data[data['Model']=='Neural_Network']
    # loss_plot(NN, filename="NN_loss_plot", title="NN Training Loss Over Epochs", xlabel="Epochs", ylabel="Loss Value")
    # R = data[data['Model']=='Regression']
    # loss_plot(R, filename="R_loss_plot", title="Regression Training Loss Over Epochs", xlabel="Epochs", ylabel="Loss Value")

    heat = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/results_NN_classification.csv')
    heat = heat[heat['Optimizer']=='SGD']
    heat = heat.sort_values(by=['Learning Rate', 'L2 Penalty'])
    NN = heat[heat['Model']=='Neural_Network']
    NN_heatmap_data = NN.pivot(index="L2 Penalty", columns="Learning Rate", values="Accuracy")
    heat_plot(NN_heatmap_data, filename="NN_heatmap", title="NN Heatmap", cmap="rocket", annot=True)

    # heat = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_2/src/utils/results_R_classification_long.csv')
    # heat = heat[heat['Optimizer']=='SGD']
    # heat = heat.sort_values(by=['Learning Rate', 'L2 Penalty'])
    # R = heat[heat['Model']=='Logistic_regression']
    # R_heatmap_data = R.pivot(index="L2 Penalty", columns="Learning Rate", values="Accuracy")
    # heat_plot(R_heatmap_data, filename="R_heatmap", title="R Heatmap", cmap="rocket", annot=True)

    print("Plots saved as 'NN_loss_plot.png' and 'R_loss_plot.png'")
