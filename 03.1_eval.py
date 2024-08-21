import os
import json
import torch
import argparse
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from detr_utils import prepare_dataset, prepare_dataloader

from detr_utils import prepare_dataset
from plot_utils import show_samples_batch
from utils import load_config, set_random_seed, param_count



def test_loop(model, test_dataloader):
    pass

def main(config):
    set_random_seed(5)
    device = config["device"]
    threshold = config["threshold"]
    # threshold = 0.017
    save_dir = config["save_model_dir"]

    dataset = load_dataset(config['ds_name'], name="full", split="test", cache_dir=config["dataset_cache_dir"])

    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    mean, std = processor.image_mean, processor.image_std

    test_dataset, id2label = prepare_dataset(dataset, processor, "test")
    test_dataloader = prepare_dataloader(test_dataset, processor, batch_size=config['val_batch_size'], shuffle=False)
    model = DetrForObjectDetection.from_pretrained(config["model_ckpts"]).to(device)



    batch = next(iter(test_dataloader))
    batch = next(iter(test_dataloader))

    show_samples_batch(batch, id2label, mean, std)
    plt.savefig(os.path.join(config["save_model_dir"], "test_batch_sample.png"))


    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # Forward Pass:
    with torch.no_grad():
        output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)


    # print("output:  ",  output.keys())
    # print("output:  ",  output.logits.shape)
    # print("output:  ",  output.pred_boxes.shape)

    # prediction = output.logits[0][0]
    # print(F.softmax(prediction))
    # exit(0)


    probs = F.softmax(output.logits, dim=-1)
    # After softmask, drop the last (no object) label
    probs = probs[:,:,:-1]
    mask = probs > threshold
    # Step 3: Get the indices where the probabilities exceed the threshold
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
    print()
    print()
    print(indices)
    print()
    print()
    # turn into a list of dictionaries (one item for each example in the batch)
    # orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    # results = processor.post_process_object_detection(output, target_sizes=orig_target_sizes, threshold=threshold)

    # print(results)
    # print(results)


    batch_objs = [[] for _ in range(len(batch))]
    batch_obj_bbox = [[] for _ in range(len(batch))]
    batch_obj_scores = [[] for _ in range(len(batch))]
    for batch_idx, prediction_id, obj_id in zip(indices[0], indices[1], indices[2]):
        batch_objs[batch_idx].append(id2label[obj_id])
        batch_obj_bbox[batch_idx].append(output.pred_boxes[batch_idx, prediction_id].detach().to("cpu").numpy())
        batch_obj_scores[batch_idx].append(probs[batch_idx, prediction_id, obj_id].item())

    print("Batch objects:  ",  batch_objs)
    print("Batch objects Bounding Boxes:  ",  batch_obj_bbox)
    print("Batch objects Scores:  ",  batch_obj_scores)

    for im, labels, pred_obj, pred_bbox, pred_score in zip(batch["pixel_values"], 
                                            batch['labels'], 
                                            batch_objs,
                                            batch_obj_bbox, 
                                            batch_obj_scores):
        print("Image shape:  ",  im.shape)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune the DETR model")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    config = load_config(args.dir)
    main(config)
