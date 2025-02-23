import numpy as np
from tqdm import tqdm

import datasets as hf_datasets # datasets.arrow_dataset.Dataset, IterableDataset types

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

import albumentations



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

    id2label = {0:'head', 1:'helmet', 2:'person', 3:'others'}
    label2id = {v: k for k, v in id2label.items()}

    transform = albumentations.Compose(
        [
            # albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )

    def formatted_anns(image_id, category, area, bbox):
        # The image_processor (https://github.com/huggingface/transformers/blob/9485289f374d4df7e8aa0ca917dc131dcf64ebaf/src/transformers/models/detr/image_processing_detr.py)
        #   expects annotations to be in the format:
        #       'image_id': int, 'annotations': List[Dict]
        #   where each dictionary is a COCO object annoation
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)
        return annotations

    def train_transform_aug(examples):
        """
        Transform for an entire batch.
        """
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))
            out = transform(image=image, bboxes=objects["bbox"], category=objects["id"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return processor(images=images, annotations=targets, return_tensors="pt")

    def val_transform(examples):
        """
        Transform for an entire batch.
        """
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))
            area.append(objects["area"])
            images.append(image)
            bboxes.append(objects["bbox"])
            categories.append(objects["id"])

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]
        return processor(images=images, annotations=targets, return_tensors="pt")

    if split == "train":
        use_transform = train_transform_aug
    else:
        use_transform = val_transform

    # https://github.com/huggingface/datasets/issues/4983#issuecomment-1490444711
    if isinstance(dataset, hf_datasets.arrow_dataset.Dataset):
        dataset = dataset.with_transform(use_transform)
    elif isinstance(dataset, hf_datasets.iterable_dataset.IterableDataset):
        dataset = dataset.map(use_transform)
    else:
        print(f"Unknown dataset type {type(dataset)}, trying default...")
        dataset = dataset.with_transform(use_transform)

    return dataset, id2label

def prepare_dataloader(dataset, processor, **kwargs):
    # print("Type processor:  ",  type(processor))
    def collate_fn(batch):
        pixel_values = [ex["pixel_values"] for ex in batch]
        labels = [ex["labels"] for ex in batch]
        # print("Labels:  ",  labels)
        # print("Type processor:  ",  type(processor))
        ret = processor.pad(pixel_values, annotations=labels, return_tensors="pt")
        for elem in ret["labels"]:
            elem["boxes"] = elem["boxes"].float()
        return ret

    dataloader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)

    return dataloader

def train_one_epoch(model, dataloader, optimizer, scheduler, device, writer, epoch):
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
            # Log the training loss every 50 iterations
            writer.add_scalar('Training/Loss', loss.item(), epoch * len(dataloader) + i)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log learning rate
        writer.add_scalar('Training/LearningRate', scheduler.get_last_lr()[0], epoch * len(dataloader) + i)

    loss_dict = {k: v/len(dataloader) for k, v in loss_dict_sum.items()}
    
    model.train()
    # Log average loss and individual loss components for the epoch
    writer.add_scalar('Training/AverageLoss', loss_sum / len(dataloader), epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(f'Training/{k}', v, epoch)

    return loss_sum / len(dataloader), loss_dict
    # for i, batch in enumerate(pbar := tqdm(dataloader)):
    #     pixel_values = batch["pixel_values"].to(device)
    #     pixel_mask = batch["pixel_mask"].to(device)
    #     labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

    #     outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    #     loss = outputs.loss
    #     loss_dict = outputs.loss_dict

    #     loss_sum += loss.item()
    #     for k, v in loss_dict.items():
    #         loss_dict_sum[k] = v.item() if k not in loss_dict_sum.keys() else loss_dict_sum[k] + v.item()

    #     if i % 50 == 0:
    #         pbar.set_description(f'training_loss: {loss_sum/(i+1):.3f}')

    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()
    #     optimizer.zero_grad()

    # loss_dict = {k: v/len(dataloader) for k, v in loss_dict_sum.items()}
    # return loss_sum / len(dataloader), loss_dict

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
