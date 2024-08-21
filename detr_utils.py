import numpy as np
from tqdm import tqdm

import datasets as hf_datasets # datasets.arrow_dataset.Dataset, IterableDataset types

import torch
from torch.utils.data import DataLoader

def prepare_dataset(dataset, processor, split):
    """
    Prepare and transform the datasets

    :param dataset: Hugging Face loaded dataset dict
    :type config: datasets.arrow_dataset.Dataset, 
                datasets.iterable_dataset.IterableDataset, 
                or datasets.dataset_dict.DatasetDict
    :param processor: The processor to transform datasets
    :type processor: transformers.DetrImageProcessor
    :param split: The split to select from the dataset dict
    :type processor: str
    :returns: Transformed datasets, for usage with DETR
    :rtype: datasets.arrow_dataset.Dataset
    """
    if isinstance(dataset, hf_datasets.dataset_dict.DatasetDict):
        dataset = dataset[split]

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
    if isinstance(dataset, hf_datasets.arrow_dataset.Dataset):
        dataset = dataset.map(add_targets_col).with_transform(transform)
    elif isinstance(dataset, hf_datasets.iterable_dataset.IterableDataset):
        dataset = dataset.map(add_targets_col).map(transform)
    else:
        print(f"Unknown dataset type {type(dataset)}, trying default...")
        dataset = dataset.map(add_targets_col).with_transform(transform)


    id2label = {i: name for i, name in enumerate(dataset.features["objects"].feature["category"].names)}

    return dataset, id2label

def prepare_dataloader(dataset, processor, **kwargs):
    def collate_fn(batch):
        pixel_values = [ex["pixel_values"] for ex in batch]
        labels = [ex["labels"] for ex in batch]
        ret = processor.pad(pixel_values, annotations=labels, return_tensors="pt")
        for elem in ret["labels"]:
            elem["boxes"] = elem["boxes"].float()
        return ret

    dataloader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)

    return dataloader

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
