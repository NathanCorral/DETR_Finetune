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
from torch.nn import functional as F

from detr_utils import prepare_dataset, prepare_dataloader
from plot_utils import denormalize_image, show_samples_batch, draw_bbox_centerxywh_rel
from utils import load_config, set_random_seed, param_count, make_folder


def postprocess(model_out, threshold):
    # Almost replication of:
    # results = processor.post_process_object_detection(model_out, target_sizes=orig_target_sizes, threshold=threshold)
    batch_size = len(model_out.logits)

    probs = F.softmax(model_out.logits, dim=-1)
    # After softmask, drop the last (no object) label
    probs = probs[:,:,:-1]
    mask = probs > threshold
    # Get the indices where the probabilities exceed the threshold
    # The 'indices' variable will be a tuple of three tensors (batch_idx, pred_idx, obj_idx)
    # batch_idx: The batch index
    # pred_idx: The prediction index
    # obj_idx: The object index where the probability exceeds the threshold
    indices = torch.where(mask)
    indices = [element.detach().to("cpu").numpy().astype(int) for element in indices]
    # selected_probs = probs[mask]
    # selected_indices = indices[2]  # obj_idx where prob > threshold
    # print("Probabilities greater than threshold:", selected_probs)
    # print("Object indices:", selected_indices)

    # turn into a list of dictionaries (one item for each example in the batch)
    # orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

    results_all = [{"scores": [], "labels": [], "bboxes_centerxywh_rel": []} for _ in range(batch_size)]
    for batch_idx, prediction_id, obj_id in zip(indices[0], indices[1], indices[2]):
        score = probs[batch_idx, prediction_id, obj_id].item()
        bbox = model_out.pred_boxes[batch_idx, prediction_id].detach().to("cpu").numpy()
        label = obj_id

        results_all[batch_idx]["scores"].append(score)
        results_all[batch_idx]["labels"].append(label)
        results_all[batch_idx]["bboxes_centerxywh_rel"].append(bbox)

    return results_all

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


def test_loop(config, model, dataloader, thresholds, display_prob=0.05):
    device = config["device"]
    model.eval()

    for i, batch in enumerate(pbar := tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        # labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        # print(batch['labels'])

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        if True or random.random() >= display_prob:
            display_multi_threshold(config, batch, outputs, thresholds)

        if i > 8:
            break

def main(config):
    set_random_seed(5)
    device = config["device"]
    # threshold = config["threshold"]
    # threshold = 0.017
    thresholds = [0.7, 0.5, 0.3, 0.1]
    thresholds = [0.7, 0.5, 0.3]
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
