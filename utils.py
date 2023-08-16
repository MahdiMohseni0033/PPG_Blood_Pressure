import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(tmp,losses):
    tmp = np.array(tmp)
    losses = np.array(losses)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot learning rate in the first subplot
    ax1.plot(tmp, color='red')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate')

    # Plot loss in the second subplot
    ax2.plot(losses, color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')

    plt.tight_layout()
    plt.show()