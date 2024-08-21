import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import load_config, load_losses

def plot_loss(training_loss, validation_loss=None, epochs=None, title="Training/Validation Loss", xlabel="Epoch", ylabel="Loss"):
    """
    Plots the training and validation loss using Matplotlib and Seaborn.

    Parameters:
    - training_loss: List or array of training loss values.
    - validation_loss: (Optional) List or array of validation loss values.
    - epochs: (Optional) Number of epochs. If not provided, the length of training_loss will be used.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    if epochs is None:
        epochs = len(training_loss)
    
    fig, ax = plt.subplots()
    
    ax.plot(range(1, epochs + 1), training_loss, label='Training Loss', color='blue', linewidth=2.5)
    
    if validation_loss is not None:
        ax.plot(range(1, epochs + 1), validation_loss, label='Validation Loss', color='orange', linewidth=2.5)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticks(range(1, epochs + 1))
    ax.set_xticklabels(range(1, epochs + 1))
    ax.legend()
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    return fig, ax


def plot_metrics(train_dict, val_dict=None, epochs=None, title="Metric Losses Train/Val", check_drop_first_epoch=True):
    """
    Plots a dictionary of training and validation metrics using Matplotlib and Seaborn in subplots.

    :param train_dict: Dictionary of training metrics where keys are metric names and values are lists of metric values per epoch.
    :param val_dict: (Optional) Dictionary of validation metrics where keys are metric names and values are lists of metric values per epoch.
    :param epochs: (Optional) Number of epochs. If not provided, the length of the metric lists will be used.
    :param check_drop_first_epoch:  If set to true, will check if the absolute value of the first epoch is larger than 500% of the next closes epoch's value, and drop it if so
    :returns: A tuple of Matplotlib figure and axes objects.
    :raises KeyError: If the keys in train_dict and val_dict do not match for a particular metric.
    """
    # Combine keys from both train and validation dictionaries
    all_keys = set(train_dict.keys())
    if val_dict is not None:
        all_keys.update(val_dict.keys())

    # Determine the number of rows (subplots)
    n_rows = len(all_keys)

    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    fig.suptitle(title)
    
    if n_rows == 1:
        axes = [axes]  # Ensure axes is a list even for a single subplot

    if epochs is None:
        # Assuming all lists have the same length
        epochs = len(next(iter(train_dict.values())))

    for i, key in enumerate(sorted(all_keys)):
        ax = axes[i]
        ax.set_title(key)
        ax.set_ylabel(key)

        start = 1
        splice = 0
        if key in train_dict and len(train_dict[key]) > 4 and check_drop_first_epoch:
            first_val = train_dict[key][0]
            next_closest = min(train_dict[key][1:], key=lambda x:abs(x-first_val))
            if abs(next_closest)*5 < abs(first_val):
                start = 2
                splice = 1

        if key in train_dict:
            ax.plot(range(start, epochs + 1), train_dict[key][splice:], label='Training ' + key, color='blue', linewidth=2.5)

        if val_dict is not None and key in val_dict:
            ax.plot(range(start, epochs + 1), val_dict[key][splice:], label='Validation ' + key, color='orange', linewidth=2.5)

        if i == 0:
            ax.legend()

    # Set x-label only on the last subplot
    axes[-1].set_xlabel("Epoch")

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig, axes

def main(config):
    train_losses = load_losses(config["train_losses"])
    val_losses = load_losses(config["val_losses"])
    train_loss = train_losses["loss"]
    del train_losses["loss"]
    val_loss = val_losses["loss"]
    del val_losses["loss"]

    plot_loss(train_loss, val_loss)
    plt.savefig(os.path.join(config["plot_dir"], "training_loss.png"), bbox_inches='tight')

    plot_metrics(train_losses, val_losses)
    plt.savefig(os.path.join(config["plot_dir"], "training_metric_losses.png"), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune the DETR model")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    config = load_config(args.dir)
    main(config)
