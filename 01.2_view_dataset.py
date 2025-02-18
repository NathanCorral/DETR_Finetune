import os
import json
import torch
import argparse
import numpy as np

from datasets import load_dataset

from transformers import DetrImageProcessor
from torch.nn import Identity
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from detr_utils import prepare_dataset, prepare_dataloader
from plot_utils import show_samples_batch, show_objs
from utils import load_config, set_random_seed, make_folder


def plot_class_distribution(dataset, dataset_name, id2label):
    """
    Print and plot the class distribution of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to analyze.
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    labels = np.zeros(len(id2label), dtype=int)
    for example in dataset:
        for label in example['labels']['class_labels']:
            labels[label] += 1

    # Plot the class distribution
    # Large figuresize to fit label axes.  Width x Height in inches
    # fig, ax = plt.subplots(figsize=(10.,12.))
    fig, ax = plt.subplots()
    ax.bar(id2label.values(), labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Examples")
    ax.set_title(f"Class Distribution for {dataset_name} Data")
    ax.xaxis.set_ticks(np.arange(len(id2label), dtype=int))
    ax.set_xticklabels(id2label.values(), rotation=90)
    return fig, ax

def dataset_explorer(dataset, id2label, save_dir, n=3):
    """
    Explore the n images and the objects in the dataset
    """
    # Note that splicing the samples e.g. samples["image"] wont apply the train transform because
    #   the transform examples only have the "image" column and none of the others i.e. ["image_id", "targets", ..]
    # This method may not work with an itterable dataset, which doesnt have a train transform
    dataset_without_transform = dataset.with_transform(lambda x: x) # Remove transform
    samples = dataset_without_transform.shuffle().select(range(n)) # selecting random images

    for i, sample in enumerate(samples):
        im = np.array(sample["image"])
        titles = [category for category in sample["objects"]["category"]]
        titles = [f'Image: {sample["image_id"]}'] + titles
        show_objs(im, sample["objects"]["bbox"], titles=titles)
        plt.savefig(os.path.join(save_dir, f'dataset_im_{sample["image_id"]}'), bbox_inches='tight')


def save_im_augs(dataset, save_dir, n=3):
    """
    Display augmented images with their originals side-by-side.
    """
    idxs = np.random.randint(len(dataset), size=(n))
    dataset_without_transform = dataset.with_transform(lambda x: x) 

    fig, axes = plt.subplots(nrows=n, ncols=2)
    for i, idx in enumerate(idxs):
        orig_img = dataset_without_transform[int(idx)]['image']
        trans_img = dataset[int(idx)]['pixel_values'].permute(1, 2, 0).numpy()

        axes[i][0].imshow(orig_img)
        axes[i][0].axis('off')
        axes[i][1].imshow(trans_img)
        axes[i][1].axis('off')
        
        if i == 0:
            axes[i][0].set_title('Original Image(s)')
            axes[i][1].set_title('Transformed Image(s)')
        
    plt.savefig(os.path.join(save_dir, f'im_augs.png'), bbox_inches='tight')


def main(config, num_samples=8):
    set_random_seed(7)
    save_dir = config["save_model_dir"]

    datasets = load_dataset(config['ds_name'], name=config["ds_name_arg"], cache_dir=config["dataset_cache_dir"])
    splits = ["train", "test"]

    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    mean, std = processor.image_mean, processor.image_std

    train_dataset, id2label = prepare_dataset(datasets["train"], processor, "train")
    test_dataset, _ = prepare_dataset(datasets["test"], processor, "test")
    datasets = [train_dataset, test_dataset]
    print("Dataset columns:  ",  train_dataset.column_names)

    train_dataloader = prepare_dataloader(train_dataset, processor, batch_size=config['train_batch_size'], shuffle=True)
    test_dataloader = prepare_dataloader(test_dataset, processor, batch_size=config['val_batch_size'], shuffle=False)
    dataloaders = [train_dataloader, test_dataloader]

    # Cycle through some examples, saving plots
    for split, dataset, dataloader in zip(splits, datasets, dataloaders):
        plot_dir = os.path.join(save_dir, f'data_{split}/')
        make_folder(plot_dir)
        print(f"Creating and Saving plots in {plot_dir}")

        # Plot class distribution
        # plot_class_distribution(dataset, split, id2label)
        # plt.savefig(os.path.join(plot_dir, "class_distribution.png"), bbox_inches='tight')

        if split == "train":
            save_im_augs(dataset, save_dir=plot_dir, n=3)

        # Plot objects in the dataset
        dataset_explorer(dataset, id2label, save_dir=plot_dir, n=num_samples)
        plt.close("all")

        # Save figures of a batch
        for i, batch in enumerate(dataloader):
            show_samples_batch(batch, id2label, mean, std)
            plt.savefig(os.path.join(plot_dir, f'batch_{i}.png'), bbox_inches='tight')

            if i >= num_samples:
                break
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the DETR pretrained model")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    config = load_config(args.dir)
    main(config)
