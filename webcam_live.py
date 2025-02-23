import argparse
from tqdm import tqdm
import cv2
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

from typing import Optional

import torch

from plot_utils import normalize_image, convert_bbox_to_mask
from utils import load_config, set_random_seed
from transformers import DetrImageProcessor, DetrForObjectDetection
from detr_utils import prepare_dataset, postprocess

from datasets import load_dataset


def run_inference_live(config, model, save_recording: Optional[str] = None):
    print("Running Live Inference...")
    device = config["device"]
    model.eval()
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    fps = None
    alpha = 0.4
    id2label = config["id2label"]
    labels_to_color_pred = {
        "helmet": 'green',
        "head": 'magenta',
    }

    with torch.no_grad():
        while True:
            ret, im = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            pixel_values = normalize_image(im, config["mean"], config["std"])
            pixel_values = torch.tensor(np.array([pixel_values])).permute(0, 3, 1, 2).to(device)
            pixel_mask = torch.tensor(np.array([np.ones(im.shape[:2])])).to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            result = postprocess(outputs, config["threshold"])[0] # First and only image results
            combined_mask = np.zeros(im.shape[:3], dtype=np.float32)
            # Create Image mask of bounding boxes
            combined_mask = np.zeros(im.shape[:3], dtype=np.uint8)
            print("Num Detections:  ",  len(result["scores"]))
            for label, bbox_centerxywh_rel in zip(result["labels"], 
                                                        result["bboxes_centerxywh_rel"]):
                mask = convert_bbox_to_mask(bbox_centerxywh_rel, im.shape[:3])
                color = labels_to_color_pred[id2label[label]]
                mask = mask * 255 * plt.cm.colors.to_rgba(color)[:3]
                combined_mask += mask.astype(np.uint8)

                
            combined_mask = np.clip(combined_mask, 0, 255)
            masked_im = cv2.addWeighted(combined_mask, alpha, im, 1 - alpha, 0)
            cv2.imshow('Image with Mask Overlay', masked_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            



def main(config):
    set_random_seed(5)
    device = config["device"]

    loaded_images_array = np.load('captured_images.npy')

    # Load the pre-processor just for the mean/std
    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    config["mean"], config["std"] = processor.image_mean, processor.image_std

    # Load the test dataset just to get the id2lable ....
    dataset = load_dataset(config['ds_name'], name=config["ds_name_arg"], 
                                split="test", cache_dir=config["dataset_cache_dir"])
    test_dataset, id2label = prepare_dataset(dataset, processor, "test")
    config["id2label"] = id2label

    # Reload the model
    model = DetrForObjectDetection.from_pretrained(config["model_ckpts"]).to(device)

    run_inference_live(config, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on live camera feed")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    config = load_config(args.dir)
    main(config)
