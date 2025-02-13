import os
import argparse
import numpy as np

from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, get_cosine_schedule_with_warmup

import torch
from torch.utils.tensorboard import SummaryWriter

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

def print_learning_rate(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Learning rate for param group {i}: {param_group['lr']}")

def save_train_ckpt(ckpt_dir, epoch, steps,
                model, optimizer, scheduler, 
                train_losses, train_loss_items, 
                val_losses, val_loss_items):
    checkpoint = {
        'epoch': epoch,
        'steps': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        # 'scaler_state_dict': scaler.state_dict(), # if using mixed precision
        'train_losses': train_losses,
        'train_loss_items': train_loss_items,
        'val_losses': val_losses,
        'val_loss_items': val_loss_items,
    }
    filename = f'train_ckpt_{epoch:03d}.pth'
    save_path = os.path.join(ckpt_dir, filename)
    torch.save(checkpoint, save_path)
    print(f'Model Saved under:  {save_path}')

def get_most_recent_checkpoint(ckpt_dir):
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth') and "train" in f]
    
    if not checkpoint_files:
        return None
    
    # Sort files by epoch number, assuming filenames are in the form 'train_ckpt_###.pth'
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    most_recent_checkpoint = checkpoint_files[-1]  # The latest checkpoint
    return os.path.join(ckpt_dir, most_recent_checkpoint)

def restore(ckpt_dir, model, optimizer, scheduler, device):
    ckpt_path = get_most_recent_checkpoint(ckpt_dir)
    if not ckpt_path:
        raise RuntimeError(f'Tried to continue training but no ckpt found in {ckpt_dir}')
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f'Continueing training from:  {ckpt_path}')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # scaler.load_state_dict(checkpoint['scaler_state_dict'], map_location=device)

    # Step the scheduler to catch up learing rate.  If not loading state dict
    # for _ in range(start_epoch):
    #     scheduler.step()

    # start_epoch = checkpoint['epoch']
    # cur_steps = checkpoint['steps']

    # train_losses = checkpoint['train_losses']
    # train_loss_items = checkpoint['train_loss_items']
    # val_losses = checkpoint['val_losses']
    # val_loss_items = checkpoint['val_loss_items']
    return checkpoint

def main(config, continue_training=False, seed=5):
    set_random_seed(seed)
    save_dir = config["save_model_dir"]
    device = config["device"]

    datasets = load_dataset(config['ds_name'], name=config["ds_name_arg"], cache_dir=config["dataset_cache_dir"])

    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    mean, std = processor.image_mean, processor.image_std

    train_dataset, id2label = prepare_dataset(datasets["train"], processor, "train")
    # val_dataset, _ = prepare_dataset(datasets["validation"], processor, "validation")

    train_dataloader = prepare_dataloader(train_dataset, processor, batch_size=config['train_batch_size'], shuffle=True)
    # val_dataloader = prepare_dataloader(val_dataset, processor, batch_size=config['val_batch_size'], shuffle=False)
    val_dataloader = None # No validation dataset in this database 

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
    writer = SummaryWriter(config["tensorboard"])

    # Reload if we are continueing training
    if continue_training:
        checkpoint = restore(ckpt_dir, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch']
        cur_steps = checkpoint['steps']
        train_losses = checkpoint['train_losses']
        train_loss_items = checkpoint['train_loss_items']
        val_losses = checkpoint['val_losses']
        val_loss_items = checkpoint['val_loss_items']
    else:
        start_epoch, cur_steps = 0, 0
        train_losses, val_losses = [], []
        train_loss_items, val_loss_items = {}, {}

    for epoch in range(start_epoch+1, config['epochs']+1):
        print(f'Epoch: {epoch}')
        print_learning_rate(optimizer)
        train_loss, train_loss_dict = train_one_epoch(model, train_dataloader, optimizer, scheduler, config['device'], writer, epoch)
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
        
        if val_dataloader:
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

        cur_steps += len(train_dataloader)
        save_train_ckpt(ckpt_dir, epoch, cur_steps,
            model, optimizer, scheduler,
            train_losses, train_loss_items,
            val_losses, val_loss_items)

    # Save just the model
    model.save_pretrained(ckpt_dir, from_pt=True)

    # Save the final training loss/outputs
    train_loss_items["loss"] = train_losses
    val_loss_items["loss"] = val_losses
    save_losses(train_loss_items, config["train_losses"])
    if val_dataloader:
        save_losses(val_loss_items, config["val_losses"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune the DETR model")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    parser.add_argument('--continue', dest="continue_training", action="store_true", help="Continue training from the dir")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    config = load_config(args.dir)
    main(config, continue_training=args.continue_training)
