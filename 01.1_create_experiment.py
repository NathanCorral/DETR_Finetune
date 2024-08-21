import os
import json
import argparse

from utils import make_folder

def get_default_config(save_dir):
    config = {
        'debug_prints': False,
        'ds_name': 'keremberke/german-traffic-sign-detection',
        'ds_name_arg': "full",
        'model_name': 'facebook/detr-resnet-50',
        'train_batch_size': 12,
        'val_batch_size': 24,
        'weight_decay': .02,
        'lr': 2e-5,
        'threshold': 0.7,
        'warmup_steps_ratio': 0.25,
        'device': "cuda:0",
        'epochs': 30,
        'save_model_dir': save_dir,
        'freeze_backbone': False,
        'freeze_encoder': False,

        # Directory for saving dataset
        'dataset_cache_dir': os.path.join(os.getenv("HOME"), "data", "hugging_face"),
        # Directories in save_dir/ for organization
        'model_ckpts': os.path.join(save_dir, f'ckpts/'),
        'train_losses': os.path.join(save_dir, f'train_losses/'),
        'val_losses': os.path.join(save_dir, f'val_losses/'),
        'test_losses': os.path.join(save_dir, f'test_losses/'),
        'plot_dir': os.path.join(save_dir, f'plots/'),
    }
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main(save_dir):
    config = get_default_config(save_dir)
    make_folder(save_dir)
    make_folder(config["dataset_cache_dir"])
    make_folder(config["plot_dir"])
    save_config(config, os.path.join(save_dir, "cust_train_config.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create experiment configuration")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    main(args.dir)
