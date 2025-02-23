import cv2
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

from typing import Optional, Tuple

import torch

from plot_utils import normalize_image, convert_bbox_to_mask
from utils import load_config, set_random_seed
from transformers import DetrImageProcessor, DetrForObjectDetection
from detr_utils import prepare_dataset, postprocess

from datasets import load_dataset

# import os
# import json
# import math
# import random
import argparse
from tqdm import tqdm


# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D


# import torch
# from torch.utils.data import DataLoader
# from torch.nn import functional as F



# from plot_utils import denormalize_image, show_samples_batch, draw_bbox_centerxywh_rel
# from utils import load_config, set_random_seed, param_count, make_folder
# 
# import numpy as np
# import torch
# from evaluate import load


# Step 5: Load the NumPy array from the file
# loaded_images_array = np.load('captured_images.npy')
# print("Images loaded from 'captured_images.npy'")

# # Optional: Display the loaded images
# for img in loaded_images_array:
#     cv2.imshow('Loaded Image', img)
#     if cv2.waitKey(50) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

def make_mask(im, results):

    labels_to_color_pred = {
        0: 'green',
        1: 'magenta',
    }

    im_h, im_w = im.shape[:2]

    im_masked = copy.deepcopy(im)
    for bbox_centerxywh_rel in bboxes_centerxywh_rel:
        center_x, center_y, w, h = tuple(bbox_centerxywh_rel)

        y = int((center_y - h/2.)*im_h)
        x = int((center_x - w/2.)*im_w)

        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)



    return im_masked

def run_inference(config, model, video, save_recording: Optional[str] = None):
    print("Running Inference...")
    device = config["device"]
    model.eval()

    # Preporcess the images into an input the model expects
    batches = []
    print("Config keys:  ", config.keys())

    # Normalize images and set up batch
    for i, im in enumerate(video):
        pixel_values = normalize_image(im, config["mean"], config["std"])
        pixel_mask = np.ones(im.shape[:2])
        batches.append({"image": [im],
                    "pixel_values": torch.tensor(np.array([pixel_values])).permute(0, 3, 1, 2), 
                    "pixel_mask": torch.tensor(np.array([pixel_mask]))})

        if i % 100 == 0:
            print("Image:  ", i)
            print("Image shape:  ", im.shape)
            print("Image max:  ", im.max())
            print("Image man:  ", im.min())
            print("Image type:  ", im.dtype)
            print("Postprocess Image shape:  ", batches[-1]["pixel_values"].shape)
            print("Postprocess Image max:  ", batches[-1]["pixel_values"].max())
            print("Postprocess Image man:  ", batches[-1]["pixel_values"].min())
            print("Postprocess Image type:  ", batches[-1]["pixel_values"].dtype)

        if i > 300:
            break
        # cv2.imshow('Image with Mask Overlay', im)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    fps = None
    alpha = 0.4
    id2label = config["id2label"]
    labels_to_color_pred = {
        "helmet": 'green',
        "head": 'magenta',
    }


    with torch.no_grad():
        for pbar, batch in enumerate(pbar := tqdm(batches)):
            ims = batch["image"]
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            results = postprocess(outputs, config["threshold"])
            combined_mask = np.zeros(im.shape[:3], dtype=np.float32)
            for im, result in zip(ims, results):
                # Create Image mask of bounding boxes
                combined_mask = np.zeros(im.shape[:3], dtype=np.uint8)
                for label, bbox_centerxywh_rel in zip(result["labels"], 
                                                            result["bboxes_centerxywh_rel"]):
                    mask = convert_bbox_to_mask(bbox_centerxywh_rel, im.shape[:3])
                    color = labels_to_color_pred[id2label[label]]
                    mask = mask * 255 * plt.cm.colors.to_rgba(color)[:3]
                    combined_mask += mask.astype(np.uint8)

                
                combined_mask = np.clip(combined_mask, 0, 255)
                masked_im = cv2.addWeighted(combined_mask, alpha, im, 1 - alpha, 0)
                cv2.imshow('Image with Mask Overlay', masked_im)
                cv2.waitKey(1)

def load_video(video_filepath : str) -> Optional[np.ndarray]:
    if video_filepath.endswith(".npy"):
        return np.load(video_filepath)
    elif video_filepath.endswith(".mp4"):
        cap = cv2.VideoCapture(video_filepath)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return np.array(frames)
    else:
        print(f"Uknown file type:  {video_filepath}")
        return None

def run_mp4(config, model, video_file, save_file):
    device = config["device"]
    model.eval()

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' for AVI files
    out = cv2.VideoWriter(save_file, fourcc, fps, (frame_width, frame_height), isColor=True)


    # Plotting control
    alpha = 0.4
    id2label = config["id2label"]
    labels_to_color_pred = {
        "helmet": 'green',
        "head": 'magenta',
    }
    batch_size = config["val_batch_size"]

    # Enable entire frame
    pixel_mask = torch.ones((batch_size, frame_height, frame_width)).to(device)
    with torch.no_grad():
        for pbar, i in enumerate(pbar := tqdm(range(0, frame_count, batch_size))):
            if i > 500:
                break
            # print(f"Running frame:  {i}")

            loop_fail = False
            batch = {"ims": [], "pixel_values_norm": [], "pixel_mask": pixel_mask}
            for _ in range(batch_size):
                ret, im = cap.read()
                if not ret:
                    loop_fail = True
                    break

                pixel_values = torch.tensor(normalize_image(im, config["mean"], config["std"]))
                # pixel_values = [pixel_values].permute(0, 3, 1, 2).to(device)

                batch["ims"].append(im)
                batch["pixel_values_norm"].append(pixel_values)
            if loop_fail:
                print("loop failed")
                if len(batch["pixel_values_norm"]) == 0:
                    break
                pixel_mask = torch.ones((len(batch["pixel_values_norm"]), frame_height, frame_width)).to(device)

            batch["pixel_values"] = torch.stack(batch["pixel_values_norm"]).permute(0, 3, 1, 2).to(device)
            # print("Batch shape:  ",  batch["pixel_values"].shape)
            outputs = model(pixel_values=batch["pixel_values"], pixel_mask=pixel_mask)
            results = postprocess(outputs, config["threshold"])

            for im, result in zip(batch["ims"], results):
                # Create Image mask of bounding boxes
                combined_mask = np.zeros(im.shape[:3], dtype=np.uint8)
                for label, bbox_centerxywh_rel in zip(result["labels"], 
                                                            result["bboxes_centerxywh_rel"]):
                    mask = convert_bbox_to_mask(bbox_centerxywh_rel, im.shape[:3])
                    color = labels_to_color_pred[id2label[label]]
                    mask = mask * 255 * plt.cm.colors.to_rgba(color)[:3]
                    combined_mask += mask.astype(np.uint8)
                    
                    
                combined_mask = np.clip(combined_mask, 0, 255)
                masked_im = cv2.addWeighted(combined_mask, alpha, im, 1 - alpha, 0)
                # cv2.imshow('Image with Mask Overlay', masked_im)
                # cv2.waitKey(1)
                out.write(masked_im)


    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    print(f"Processed video saved to {save_file}")


def main(config, video_file, save_file):
    set_random_seed(5)
    device = config["device"]

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

    run_mp4(config, model, video_file, save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on live camera feed")
    parser.add_argument('dir', type=str, help="Target folder to save the configuration, models, and outputs")
    parser.add_argument('video', type=str, help=".mp4 file of the video data")
    parser.add_argument('out', type=str, help=".mp4 to write results")
    args = parser.parse_args()

    if not args.dir.endswith("/"):
        args.dir += "/"

    if not args.video.endswith(".mp4"):
        print(".mp4 file required")
        exit(1)

    if not args.out.endswith(".mp4"):
        print("Must write to .mp4 file")
        exit(1)



    config = load_config(args.dir)
    main(config, args.video, args.out)
