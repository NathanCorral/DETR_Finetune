import os
import argparse
import numpy as np

from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, get_cosine_schedule_with_warmup

import torch

from detr_utils import prepare_dataset, prepare_dataloader, train_one_epoch, validate_one_epoch
from utils import load_config, set_random_seed, param_count, make_folder, save_losses

def initialize_model(config, num_labels):
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
    model = DetrForObjectDetection.from_pretrained(
        config['model_name'], 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True
    ).to(config['device'])

    if config['freeze_backbone']:
        for param in model.model.backbone.parameters():
            param.requires_grad = False

    if config['freeze_encoder']:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    
    return model

def main(config):
    set_random_seed(5)
    save_dir = config["save_model_dir"]

    datasets = load_dataset(config['ds_name'], name="full", cache_dir=config["dataset_cache_dir"])

    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    mean, std = processor.image_mean, processor.image_std

    train_dataset, id2label = prepare_dataset(datasets["train"], processor, "train")
    val_dataset, _ = prepare_dataset(datasets["validation"], processor, "validation")

    train_dataloader = prepare_dataloader(train_dataset, processor, batch_size=config['train_batch_size'], shuffle=True)
    val_dataloader = prepare_dataloader(val_dataset, processor, batch_size=config['val_batch_size'], shuffle=False)

    model = initialize_model(config, num_labels=len(id2label))
    total, trainable, frac = param_count(model)
    print(f"{total = :,} | {trainable = :,} | {frac:.2f}%")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['warmup_steps_ratio'] * len(train_dataloader)),
        num_training_steps=len(train_dataloader)*config['epochs'],
    )

    ckpt_dir = os.path.join(save_dir, f'ckpts/')
    make_folder(ckpt_dir)

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

    # Save the model and training params
    model.save_pretrained(ckpt_dir, from_pt=True)

    # Save the training loss/outputs
    train_loss_items["loss"] = train_losses
    val_loss_items["loss"] = val_losses
    save_losses(train_loss_items, config["train_losses"])
    save_losses(val_loss_items, config["val_losses"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune the DETR model")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    config = load_config(args.dir)
    main(config)
