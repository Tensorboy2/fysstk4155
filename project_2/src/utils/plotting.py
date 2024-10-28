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
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap=cmap, annot=annot, cbar=colorbar)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()

def loss_plot(data, filename, title="Loss Over Epochs", xlabel="Epochs", ylabel="Loss", color="b"):
    """
    Plots the loss over epochs and saves it as png.
    
    Parameters:
        data (array-like): Loss values per epoch.
        filename (str): Filename to save the plot.
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        color (str): Line color.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, color=color, linewidth=2.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    heatmap_data = np.random.rand(10, 10)
    epochs = 50
    loss_data = pd.Series(np.random.exponential(scale=0.5, size=epochs).cumsum(), name="Loss")
    heat_plot(heatmap_data, filename="example_heatmap", title="Random Heatmap", cmap="coolwarm", annot=True)
    loss_plot(loss_data, filename="example_loss_plot", title="Training Loss Over Epochs", xlabel="Epochs", ylabel="Loss Value", color="purple")
    print("Plots saved as 'example_heatmap.png' and 'example_loss_plot.png'")
