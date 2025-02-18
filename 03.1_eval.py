import os
import json
import math
import random
import argparse
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection

import torch
from torch.utils.data import DataLoader

from detr_utils import prepare_dataset, prepare_dataloader, postprocess
from plot_utils import denormalize_image, show_samples_batch, draw_bbox_centerxywh_rel, convert_bbox_to_mask
from utils import load_config, set_random_seed, param_count, make_folder

import numpy as np
import torch
from evaluate import load


# def create_pred_mask(results_all, id2labels, colors, ):
#     """
#     Convert the predicted bounding boxes into a mask
#     """


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


def evaluate_bounding_boxes(predictions, ground_truth, image_size, num_classes):
    """Evaluate bounding boxes using mean_iou metric."""
    mean_iou = load("mean_iou")
    pred_masks, gt_masks = [], []

    for pred, gt in zip(predictions, ground_truth):
        pred_mask = np.zeros(image_size, dtype=np.int32)
        gt_mask = np.zeros(image_size, dtype=np.int32)

        for bbox, label in zip(pred['bboxes'], pred['labels']):
            pred_mask = np.maximum(pred_mask, convert_bbox_to_mask(bbox, image_size) * (label + 1))

        for bbox, label in zip(gt['bboxes'], gt['labels']):
            gt_mask = np.maximum(gt_mask, convert_bbox_to_mask(bbox, image_size) * (label + 1))

        pred_masks.append(pred_mask)
        gt_masks.append(gt_mask)

    results = mean_iou.compute(predictions=pred_masks, references=gt_masks, num_labels=num_classes + 1, ignore_index=0)
    return results


def display_multi_threshold_old(config, batch, outputs, thresholds):
    """
    Create a multi-axis plot for each image in the batch displaying results using multiple thresholds
    """
    mean, std = config["mean"], config["std"]
    id2label = config["id2label"]
    batch_size = len(batch)-1

    labels_to_color_gt = {
        "helmet": 'g',
        "head": 'b',
        "person": 'cyan',
        "other": 'purple',

    }
    labels_to_color_pred = {
        "helmet": 'y',
        "head": 'r',
        "person": 'orange',
        "other": 'magenta',
    }

    # Create custom legend handles for ground truth and predictions, only to be displayed on first ax
    gt_handles = [Line2D([0], [0], color=color, lw=4, label=label) for label, color in labels_to_color_gt.items()]
    pred_handles = [Line2D([0], [0], color=color, lw=4, label=label) for label, color in labels_to_color_pred.items()]

    # Postprocess the model outputs using specific batch
    batched_results = {}
    for threshold in thresholds:
        batched_results[threshold] = postprocess(outputs, threshold)

    # Create a figure for each image batch
    for i in range(batch_size):
        save_png = os.path.join(config["plot_dir"], "test_thresh_examples", f'image_{batch["labels"][i]["image_id"].item()}.png')
        make_folder(save_png)

        # batchresi = {thresh: batch_results[thresh][i] for thresh in batch_results.keys()}
        num_plots = len(thresholds)+1
        if num_plots == 1:
            fig, axes = plt.subplots()
            axes = [axes]
        elif num_plots > 3:
            # Divide into two columns.  rows, cols
            fig, axes = plt.subplots(int(math.ceil(num_plots/2.)), 2)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots((num_plots))

        # GT data and image from each batch
        im = batch["pixel_values"][i]
        im = denormalize_image(im.permute(1, 2, 0).numpy(), mean, std)
        # mask = batch["pixel_mask"][i] # unused
        labels = batch["labels"][i]["class_labels"].numpy()
        bboxes_centerxywh_rel = batch["labels"][i]["boxes"]

        for ax_i, thresh in enumerate(thresholds):
            ax = axes[ax_i]
            ax.imshow(im)

            # Show the ground truth
            for label, bbox_centerxywh_rel in zip(labels, bboxes_centerxywh_rel):
                label_str = id2label[label]
                c = labels_to_color_gt[label_str] if label_str in labels_to_color_gt else 'g'
                _, _ = draw_bbox_centerxywh_rel(ax, bbox_centerxywh_rel, edgecolor=c, linewidth=2)

            # Show predictions
            results = batched_results[thresh][i]
            for label, bbox_centerxywh_rel in zip(results["labels"], results["bboxes_centerxywh_rel"]):
                label_str = id2label[label]
                # print(bbox_centerxywh_rel)
                c = labels_to_color_pred[label_str] if label_str in labels_to_color_gt else 'r'
                _, _ = draw_bbox_centerxywh_rel(ax, bbox_centerxywh_rel, edgecolor=c, linewidth=1)

            # ax.set_title(f"Threshold:  {thresh}\n#GTs:  {len(labels)}\n#Preds:  {len(results['labels'])}")
            ax.set_title(f"Threshold:  {thresh}")

        # Add the legends to the last (blank) plot
        ax = axes[-1]
        gt_legend = ax.legend(handles=gt_handles, title="Ground Truth", loc='upper left')
        ax.add_artist(gt_legend)  # This adds the first legend to the axes
        pred_legend = ax.legend(handles=pred_handles, title="Predictions", loc='upper right')
        ax.axis("off")
        # print("Showing:  ", save_png)
        plt.savefig(save_png, bbox_inches='tight')
        # plt.savefig(save_png, dpi=300)

def display_multi_threshold(config, batch, outputs, thresholds):
    """
    Create a multi-axis plot for each image in the batch displaying results using multiple thresholds
    """
    mean, std = config["mean"], config["std"]
    id2label = config["id2label"]
    batch_size = len(batch)-1

    labels_to_color_gt = {
        "helmet": 'g',
        "head": 'b',
        "person": 'cyan',
        "other": 'purple',
    }
    labels_to_color_pred = {
        "helmet": 'orange',
        "head": 'r',
        "person": 'orange',
        "other": 'magenta',
    }

    # Create custom legend handles for ground truth and predictions
    gt_handles = [Line2D([0], [0], color=color, lw=4, label=label) for label, color in labels_to_color_gt.items()]
    pred_handles = [Line2D([0], [0], color=color, lw=4, label=label) for label, color in labels_to_color_pred.items()]

    # Postprocess the model outputs using specific batch
    batched_results = {}
    for threshold in thresholds:
        batched_results[threshold] = postprocess(outputs, threshold)

    for i in range(batch_size):
        save_png = os.path.join(config["plot_dir"], "test_thresh_examples", f'image_{batch["labels"][i]["image_id"].item()}.png')
        make_folder(save_png)

        num_plots = len(thresholds)+1
        if num_plots == 1:
            fig, axes = plt.subplots()
            axes = [axes]
        elif num_plots > 3:
            fig, axes = plt.subplots(int(math.ceil(num_plots/2.)), 2)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots((num_plots))

        # GT data and image from each batch
        im = batch["pixel_values"][i]
        im = denormalize_image(im.permute(1, 2, 0).numpy(), mean, std)
        labels = batch["labels"][i]["class_labels"].numpy()
        bboxes_centerxywh_rel = batch["labels"][i]["boxes"]

        for ax_i, thresh in enumerate(thresholds):
            ax = axes[ax_i]
            ax.imshow(im)

            # Show the ground truth
            for label, bbox_centerxywh_rel in zip(labels, bboxes_centerxywh_rel):
                label_str = id2label[label]
                c = labels_to_color_gt[label_str] if label_str in labels_to_color_gt else 'g'
                _, _ = draw_bbox_centerxywh_rel(ax, bbox_centerxywh_rel, edgecolor=c, linewidth=2)

            # Show predictions as masks
            results = batched_results[thresh][i]
            combined_mask = np.zeros(im.shape[:3], dtype=np.float32)
            for label, bbox_centerxywh_rel in zip(results["labels"], results["bboxes_centerxywh_rel"]):
                label_str = id2label[label]
                c = labels_to_color_pred[label_str] if label_str in labels_to_color_pred else 'r'
                mask = convert_bbox_to_mask(bbox_centerxywh_rel, im.shape[:3])
                combined_mask += mask * plt.cm.colors.to_rgba(c)[:3]

            # Normalize the combined mask
            combined_mask = np.clip(combined_mask, 0, 1)
            
            # Overlay the mask on the image
            ax.imshow(combined_mask, alpha=0.5)

            ax.set_title(f"Threshold: {thresh}")

        # Add the legends to the last (blank) plot
        ax = axes[-1]
        gt_legend = ax.legend(handles=gt_handles, title="Ground Truth", loc='upper left')
        ax.add_artist(gt_legend)
        pred_legend = ax.legend(handles=pred_handles, title="Predictions", loc='upper right')
        ax.axis("off")

        plt.savefig(save_png, bbox_inches='tight')
        plt.close("all")



def test_loop(config, model, dataloader, thresholds, display_prob=0.05):
    device = config["device"]
    id2label = config["id2label"]
    threshold = config["threshold"]
    model.eval()
    distribution_labels = np.zeros(len(id2label), dtype=int)

    for i, batch in enumerate(pbar := tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        # labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        # print(batch['labels'])

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            results = postprocess(outputs, threshold)
            for result in results:
                for label in result["labels"]:
                    distribution_labels[label] += 1

        if random.random() <= display_prob:
            display_multi_threshold(config, batch, outputs, thresholds)

        # if i > 8:
        #     break



    save_png = os.path.join(config["plot_dir"], f"distribution_thresh_{threshold}.png")
    print(f"Creating figure:  {save_png}")
    print(distribution_labels)
    # Plot the label distribution
    fig, ax = plt.subplots()
    ax.bar(id2label.values(), distribution_labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Examples")
    ax.set_title(f"Class Distribution of Model Predictions \nTest set.  Thresh: {threshold}")
    ax.xaxis.set_ticks(np.arange(len(id2label), dtype=int))
    ax.set_xticklabels(id2label.values(), rotation=90)
    plt.savefig(save_png, bbox_inches='tight')

def main(config):
    set_random_seed(5)
    device = config["device"]
    # threshold = config["threshold"]
    # threshold = 0.017
    thresholds = [0.7, 0.5, 0.3]
    # thresholds = [0.2, 0.5, 0.3]
    # thresholds = [0.3]
    save_dir = config["save_model_dir"]

    dataset = load_dataset(config['ds_name'], name=config["ds_name_arg"], split="test", cache_dir=config["dataset_cache_dir"])

    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    # mean, std = processor.image_mean, processor.image_std

    test_dataset, id2label = prepare_dataset(dataset, processor, "test")
    test_dataloader = prepare_dataloader(test_dataset, processor, batch_size=config['val_batch_size'], shuffle=False)
    model = DetrForObjectDetection.from_pretrained(config["model_ckpts"]).to(device)

    config["mean"], config["std"] = processor.image_mean, processor.image_std
    config["id2label"] = id2label
    print(id2label)
    test_loop(config, model, test_dataloader, thresholds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune the DETR model")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    config = load_config(args.dir)
    main(config)
