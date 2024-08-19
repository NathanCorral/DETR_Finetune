import os
import json
import argparse
import numpy as np
from tqdm import tqdm

# Torch
import torch
from torch.utils.data import DataLoader

# Hugging Face
import datasets as hf_datasets # datasets.arrow_dataset.Dataset, IterableDataset types
from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, get_cosine_schedule_with_warmup

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# Local
from plot_utils import draw_bbox_xywh, draw_bbox_centerxywh_rel, show_samples, denormalize_image, show_samples_batch, param_count, show_objs
from utils import make_folder, set_random_seed

def get_default_config():
    """ Create the default configuration, (optionally) overwritten by the .json passed in thru argparser. """
    config = {
        'debug_prints': False,
        'ds_name': 'keremberke/german-traffic-sign-detection',
        # 'ds_name': 'detection-datasets/coco',
        'model_name': 'facebook/detr-resnet-50',
        'train_batch_size': 32,
        'val_batch_size': 64,
        # Optomizer
        'weight_decay':.02,
        'lr': 2e-4,
        # DETR Specific 
        'threshold': 0.7, # postprocessing
        # Scheduler
           # multiply by the length of the dataloader to determine warmup steps
        'warmup_steps_ratio': 0.5,
        # Other
        'device': "cuda:0",
        'epochs': 30,
        # Make sure to include /
        'save_model_dir': "./streetsign_recognizer_0/",

        # Finetune v.s. Transfer Learning tests:
        'freeze_backbone': False,
        'freeze_encoder': False,

        # plotting
        # 'figure_dpi': 100,
    }
    return config

def load_config(config_path):
    """
    Load the configuration settings from a JSON file.

    :param config_path: Path to the configuration file
    :type config_path: str
    :returns: Configuration dictionary
    :rtype: dict
    :raises FileNotFoundError: If the config file is not found
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    """
    Save the configuration settings to a JSON file.

    :param config: Configuration dictionary
    :type config: dict
    :param config_path: Path to the file where config will be saved
    :type config_path: str
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def initialize_model(config, num_labels=None):
    """
    Initialize the DETR model with the given configuration.

    num_labels param see:
        https://github.com/huggingface/transformers/blob/8820fe8b8c4b9da94cf1e4761876f85c562e0efe/src/transformers/configuration_utils.py#L201
    ignore_mismatched_sizes see:
        https://github.com/huggingface/transformers/blob/8820fe8b8c4b9da94cf1e4761876f85c562e0efe/src/transformers/modeling_utils.py#L2986

    :param config: Configuration dictionary
    :type config: dict
    :param num_labels: Pass the number of labels to the model initializer (optional)
    :type config: int
    :returns: Initialized model
    :rtype: transformers.DetrForObjectDetection
    """
    if num_labels:
        model = DetrForObjectDetection.from_pretrained(
            config['model_name'], 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True
        ).to(config['device'])
    else:
        model = DetrForObjectDetection.from_pretrained(
            config['model_name'], 
        ).to(config['device'])

    if config['freeze_backbone']:
        for param in model.model.backbone.parameters():
            param.requires_grad = False

    if config['freeze_encoder']:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    
    return model

def prepare_dataset(datasets, processor, split):
    """
    Prepare and transform the datasets

    :param dataset: Hugging Face loaded dataset dict
    :type config: datasets.dataset_dict.DatasetDict
    :param processor: The processor to transform datasets
    :type processor: transformers.DetrImageProcessor
    :param split: The split to select from the dataset dict
    :type processor: str
    :returns: Transformed datasets, for usage with DETR
    :rtype: datasets.arrow_dataset.Dataset
    """
    def add_targets_col(example):
        """
        Prepare the annotations to be in COCO format.
        """
        annotations = {}
        annotations["image_id"] = example["image_id"]
        annotations_list = []
        for label, bbox_xywh, area, _ in zip(
            example['objects']['category'], 
            example['objects']['bbox'], 
            example['objects']['area'], 
            example['objects']['id']):
            new_ann = {
                "image_id": example["image_id"],
                "category_id": label,
                "isCrowd": 0,
                "area": area,
                "bbox": list(bbox_xywh),
            }
            annotations_list.append(new_ann)
        annotations["annotations"] = annotations_list
        example["targets"] = annotations
        return example

    def train_transform(examples):
        """
        Transform for an entire batch.
        """
        if not "image" in examples or not "targets" in examples:
            return examples # spliced batch
        images = [np.array(im) for im in examples["image"]]
        return processor(images=images, annotations=examples["targets"], return_tensors="pt")

    def val_transform(examples):
        """
        Transform for an entire batch.
        """
        if not "image" in examples or not "targets" in examples:
            return examples # spliced batch
        images = [np.array(im) for im in examples["image"]]
        return processor(images=images, annotations=examples["targets"], return_tensors="pt")

    if split == "train":
        transform = train_transform
    else:
        transform = val_transform


    # https://github.com/huggingface/datasets/issues/4983#issuecomment-1490444711
    split_dataset = datasets[split]
    if isinstance(split_dataset, hf_datasets.arrow_dataset.Dataset):
        dataset = split_dataset.map(add_targets_col).with_transform(transform)
    elif isinstance(split_dataset, hf_datasets.iterable_dataset.IterableDataset):
        dataset = split_dataset.map(add_targets_col).map(transform)
    else:
        print(f"Unknown dataset type {type(split_dataset)}, trying default...")
        dataset = split_dataset.map(add_targets_col).with_transform(transform)

    return dataset


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train the model for one epoch.

    :param model: The model to be trained
    :type model: torch.nn.Module
    :param dataloader: DataLoader for the training data
    :type dataloader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for updating model parameters
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Learning rate scheduler
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param device: Device to run the training on
    :type device: torch.device
    :returns: The average training loss for the epoch
    :rtype: float
    """
    model.train()
    loss_sum = 0.
    loss_dict_sum = {}
    for i, batch in enumerate(pbar := tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        loss_sum += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = v.item() if k not in loss_dict_sum.keys() else loss_dict_sum[k] + v.item()

        if i % 50 == 0:
            pbar.set_description(f'training_loss: {loss_sum/(i+1):.3f}')

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    loss_dict = {k: v/len(dataloader) for k, v in loss_dict_sum.items()}
    return loss_sum / len(dataloader), loss_dict

def validate_one_epoch(model, dataloader, device):
    """
    Validate the model for one epoch.

    :param model: The model to be validated
    :type model: torch.nn.Module
    :param dataloader: DataLoader for the validation data
    :type dataloader: torch.utils.data.DataLoader
    :param device: Device to run the validation on
    :type device: torch.device
    :returns: The average validation loss for the epoch
    :rtype: float
    """
    model.eval()
    val_loss_sum = 0.
    val_loss_dict_sum = {}
    with torch.no_grad():
        for i, batch in enumerate(pbar := tqdm(dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            loss_dict = outputs.loss_dict

            val_loss_sum += loss.item()
            for k, v in loss_dict.items():
                val_loss_dict_sum[k] = 0. if k not in val_loss_dict_sum.keys() else val_loss_dict_sum[k] + v.item()

            if i % 50 == 0:
                pbar.set_description(f'validation_loss: {val_loss_sum/(i+1):.3f}')

    val_loss_dict = {k: v/len(dataloader) for k, v in val_loss_dict_sum.items()}
    return val_loss_sum / len(dataloader), val_loss_dict

def save_model(model, path):
    """
    Save the trained model to the specified path.

    :param model: The trained model
    :type model: torch.nn.Module
    :param path: Directory path to save the model
    :type path: str
    """
    model.save_pretrained(path, from_pt=True)

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
    
    sns.set(style="whitegrid")

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

def plot_metrics(train_dict, val_dict=None, epochs=None, title="Metric Losses Train/Val"):
    """
    Plots a dictionary of training and validation metrics using Matplotlib and Seaborn in subplots.

    :param train_dict: Dictionary of training metrics where keys are metric names and values are lists of metric values per epoch.
    :param val_dict: (Optional) Dictionary of validation metrics where keys are metric names and values are lists of metric values per epoch.
    :param epochs: (Optional) Number of epochs. If not provided, the length of the metric lists will be used.
    :returns: A tuple of Matplotlib figure and axes objects.
    :raises KeyError: If the keys in train_dict and val_dict do not match for a particular metric.
    """

    sns.set(style="whitegrid")

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

        if key in train_dict:
            ax.plot(range(1, epochs + 1), train_dict[key], label='Training ' + key, color='blue', linewidth=2.5)

        if val_dict is not None and key in val_dict:
            ax.plot(range(1, epochs + 1), val_dict[key], label='Validation ' + key, color='orange', linewidth=2.5)

        if i == 0:
            ax.legend()

    # Set x-label only on the last subplot
    axes[-1].set_xlabel("Epoch")

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig, axes


def dataset_explorer(dataset, id2label, save_dir, n=3, with_transform=False):
    """
    Explore the n images and the objects in the dataset
    """
    # Note that splicing the samples e.g. samples["image"] wont apply the train transform because
    #   the transform examples only have the "image" column and none of the others i.e. ["image_id", "targets", ..]
    samples = dataset.shuffle().select(range(n)) # selecting random images
    print("Sample columns:  ",  samples.column_names)

    images = [np.array(im) for im in samples["image"]]
    objects = samples["objects"]
    image_ids = samples["image_id"]

    for i, im in enumerate(images):
        print(f"Displaying objects in image:  {image_ids[i]}")
        print(f"Objects in image:  {objects[i]}")
        print(f"Image shape:  {im.shape}")
        titles = [id2label[category] for category in objects[i]["category"]]
        titles = [f'Image: {image_ids[i]}'] + titles
        show_objs(im, objects[i]["bbox"], titles=titles)
        plt.savefig(os.path.join(save_dir, f'dataset_im_{image_ids[i]}'), bbox_inches='tight')


def main(config):
    """
    Main training loop.

    :param config: Configuration dictionary
    :type config: dict
    """
    # set_random_seed(86)
    set_random_seed(5)
    # Model data pre/post processor
    processor = DetrImageProcessor.from_pretrained(config['model_name'])

    # Prepare datasets
    datasets = load_dataset(config['ds_name'], name="full")
    train_dataset = prepare_dataset(datasets, processor, "train")
    val_dataset = prepare_dataset(datasets, processor, "validation")

    # Get pretrained dataset metadata
    id2label = {i: name for i, name in enumerate(train_dataset.features["objects"].feature["category"].names)}

    # Initialize model and processor
    model = initialize_model(config, num_labels=len(id2label))
    print("Model:  ",  model)
    total, trainable, frac = param_count(model)
    print(f"{total = :,} | {trainable = :,} | {frac:.2f}%")

    # Get pretrained model metadata
    mean, std = processor.image_mean, processor.image_std

    # Prepare dataloaders
    def collate_fn(batch):
        """
        Custom collate function to handle padding and tensor transformations.

        :param batch: A batch of data samples
        :type batch: list
        :returns: A collated batch of pixel values and labels
        :rtype: dict
        """
        pixel_values = [ex["pixel_values"] for ex in batch]
        labels = [ex["labels"] for ex in batch]
        ret = processor.pad(pixel_values, annotations=labels, return_tensors="pt")
        for elem in ret["labels"]:
            elem["boxes"] = elem["boxes"].float()
        return ret
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config['train_batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=config['val_batch_size'], shuffle=False)

    if config['debug_prints']:
        dataset_explorer(train_dataset, save_dir=config["save_model_dir"], id2label=id2label)

        # Save an example of the training batch
        batch = next(iter(train_dataloader))
        show_samples_batch(batch, id2label, mean, std)
        plt.savefig(os.path.join(config["save_model_dir"], "train_batch_sample.png"))

        # Display the distribution over class labels in the train/val sets
        print("Creating Figures Displaying Dataset Class Distribution...")
        plot_class_distribution(train_dataset, f'{config["ds_name"]} Train', id2label)
        plt.savefig(os.path.join(config["save_model_dir"], "train_class_distribution.png"), bbox_inches='tight')
        plot_class_distribution(val_dataset, f'{config["ds_name"]} Validation', id2label)
        plt.savefig(os.path.join(config["save_model_dir"], "val_class_distribution.png"), bbox_inches='tight')

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['warmup_steps_ratio'] * len(train_dataloader)),
        num_training_steps=len(train_dataloader)*config['epochs'],
    )

    # Training loop
    train_losses, val_losses = [], []
    train_loss_items, val_loss_items = {}, {}
    for epoch in range(1, config['epochs'] + 1):
        print(f'Epoch: {epoch}')
        train_loss, train_loss_dict = train_one_epoch(model, train_dataloader, optimizer, scheduler, config['device'])
        # Print Epoch Loss
        print(f'Training Loss: {train_loss:.3f}')
        for k,v in train_loss_dict.items(): 
            print(f'\ttrain_{k}:   {v:.3f}')
        # Save losses for plotting later
        train_losses.append(train_loss)
        for key, value in train_loss_dict.items():
            if key not in train_loss_items.keys():
                train_loss_items[key] = [value]
            else:
                train_loss_items[key].append(value)
        

        val_loss, val_loss_dict = validate_one_epoch(model, val_dataloader, config['device'])
        print(f'Validation Loss: {val_loss:.3f}')
        for k,v in val_loss_dict.items(): 
            print(f'\tval_{k}:   {v:.3f}')
        val_losses.append(val_loss)
        for key, value in val_loss_dict.items():
            if key not in val_loss_items.keys():
                val_loss_items[key] = [value]
            else:
                val_loss_items[key].append(value)

        # Save ckpt
        if epoch % 10 == 0:
            save_model(model, os.path.join(config['save_model_dir'], f'epoch_{epoch}'))



    # Save the model
    print("Saving Model...")
    save_model(model, os.path.join(config['save_model_dir'], f'epoch_{epoch}'))
    save_model(model, config['save_model_dir'])

    # Save loss data and plot the results
    np.save(os.path.join(config["save_model_dir"], "train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(config["save_model_dir"], "val_loss.npy"), np.array(val_losses))
    plot_loss(train_losses, val_losses, epochs=config['epochs'], xlabel=f"Epoch\nSteps/Epoch: {len(train_dataloader)}")
    plt.savefig(os.path.join(config["save_model_dir"], "training_loss.png"), bbox_inches='tight')

    # Save and plot the loss_dict arrays
    for key_train, key_val in zip(train_loss_items.keys(), val_loss_items.keys()):
        np.save(os.path.join(config["save_model_dir"], f"train_{key_train}.npy"), np.array(train_loss_items[key_train]))
        np.save(os.path.join(config["save_model_dir"], f"val_{key_val}.npy"), np.array(val_loss_items[key_val]))
    plot_metrics(train_loss_items, val_loss_items, epochs=config['epochs'])
    plt.savefig(os.path.join(config["save_model_dir"], "train_metrics.png"), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR for object detection")
    parser.add_argument('--config', type=str, default="use_default_config.json", help="Path to the configuration file")
    args = parser.parse_args()

    if args.config == "use_default_config.json":
        config = get_default_config()
    else:
        config = load_config(args.config)
    
    make_folder(config["save_model_dir"])
    save_config(config, os.path.join(config["save_model_dir"], "config.json"))
    main(config)
