import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_bbox_xywh(ax, bbox_xywh, label, score=None, id2label=None, edgecolor='g', linewidth=2):
    """
    Draw bounding box on the image given in XYWH format (top-left x, top-left y, width, height).

    :param ax: The axis on which to draw the bounding box
    :type ax: matplotlib.axes.Axes
    :param bbox_xywh: The bounding box in XYWH format
    :type bbox_xywh: tuple
    :param label: The label associated with the bounding box
    :type label: int
    :param score: The confidence score of the detection
    :type score: float, optional
    :param id2label: A dictionary mapping label ids to label names
    :type id2label: dict, optional
    :param edgecolor: The color of the bounding box edge
    :type edgecolor: str, optional
    :param linewidth: The width of the bounding box edge
    :type linewidth: int, optional
    :returns: The rectangle patch and text annotation
    :rtype: tuple
    """
    x, y, w, h = tuple(bbox_xywh)
    rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
    
    label_text = f'{id2label[int(label)] if id2label else label}'
    if score:
        label_text += f': {score:.2f}'
    
    text = ax.text(x, y - 10, label_text, fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.2))
    ax.add_patch(rect)

    return rect, text

def draw_bbox_centerxywh_rel(ax, bbox_centerxywh_rel, label, score=None, id2label=None, edgecolor='g', linewidth=2):
    """
    Draw bounding box on the image given in relative center XYWH format.

    :param ax: The axis on which to draw the bounding box
    :type ax: matplotlib.axes.Axes
    :param bbox_centerxywh_rel: The bounding box in relative center XYWH format
    :type bbox_centerxywh_rel: tuple
    :param label: The label associated with the bounding box
    :type label: int
    :param score: The confidence score of the detection
    :type score: float, optional
    :param id2label: A dictionary mapping label ids to label names
    :type id2label: dict, optional
    :param edgecolor: The color of the bounding box edge
    :type edgecolor: str, optional
    :param linewidth: The width of the bounding box edge
    :type linewidth: int, optional
    :returns: The rectangle patch and text annotation
    :rtype: tuple
    """
    center_x, center_y, w, h = tuple(bbox_centerxywh_rel)
    ax_ims = ax.get_images()
    if len(ax_ims) == 0:
        print(f'Unable to add relative sized bounding boxes to axs without an image')
        return

    im_h, im_w = ax_ims[0].get_size()

    center_x *= im_w
    w *= im_w
    center_y *= im_h
    h *= im_h

    y = center_y - h//2
    x = center_x - w//2

    rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
    
    label_text = f'{id2label[int(label)] if id2label else label}'
    if score:
        label_text += f': {score:.2f}'
    
    text = ax.text(x, y - 10, label_text, fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.2))
    ax.add_patch(rect)

    return rect, text

def show_samples(ds, rows, cols, id2label):
    """
    Display a grid of sample images from the dataset with bounding boxes.

    :param ds: The dataset to display samples from
    :type ds: datasets.Dataset
    :param rows: Number of rows in the grid
    :type rows: int
    :param cols: Number of columns in the grid
    :type cols: int
    :param id2label: A dictionary mapping label ids to label names
    :type id2label: dict
    """
    samples = ds.shuffle().select(np.arange(rows*cols))  # selecting random images
    fig, axs = plt.subplots(rows, cols)
    axs_patches = []
    axs_texts = []
    axs_ims = []

    for i in range(rows*cols):
        row, col = i // rows, i % rows
        img = samples[i]['image']
        labels = samples[i]['objects']['category']
        boxes_xywh = samples[i]['objects']['bbox']
        ax = axs[row, col]

        ax_im = ax.imshow(img)

        patches, texts = [], []
        for label, bbox_xywh in zip(labels, boxes_xywh):
            patch, text = draw_bbox_xywh(ax, bbox_xywh, label, id2label=id2label)
            patches.append(patch)
            texts.append(text)

        axs_ims.append(ax_im)
        axs_texts.append(texts)
        axs_patches.append(patches)

        ax.axis('off')

def denormalize_image(im, mean, std):
    """
    Denormalize an image using the provided mean and standard deviation.

    :param im: The image to denormalize
    :type im: np.array
    :param mean: The mean used during normalization
    :type mean: list
    :param std: The standard deviation used during normalization
    :type std: list
    :returns: The denormalized image
    :rtype: np.array
    """
    im = std * im + mean
    im = np.clip(im, 0, 1)
    return im

def show_samples_batch(batch, id2label, mean, std, num_cols=2, max_num_rows=3):
    """
    Display a grid of sample images from a batch with bounding boxes.

    :param batch: A batch of data samples
    :type batch: dict
    :param id2label: A dictionary mapping label ids to label names
    :type id2label: dict
    :param mean: The mean used during normalization
    :type mean: list
    :param std: The standard deviation used during normalization
    :type std: list
    :param num_cols: Number of columns in the grid
    :type num_cols: int, optional
    :param max_num_rows: Maximum number of rows in the grid
    :type max_num_rows: int, optional
    """
    bs = len(batch["pixel_values"])

    num_rows = min(max_num_rows, bs//num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)

    axs_patches = []
    axs_texts = []
    axs_ims = []

    for i in range(num_rows*num_cols):
        if num_rows == 1:
            col = i
            ax = axs[col]
        else:
            row, col = i // num_rows, i % num_rows
            ax = axs[row, col]

        img = batch["pixel_values"][i].permute(1, 2, 0).numpy()
        img = denormalize_image(img, mean, std)
        ax_im = ax.imshow(img)

        axs_ims.append(ax_im)
        if not "labels" in batch.keys():
            continue

        bboxes_centerxywh_rel = batch['labels'][i]['boxes']
        labels = batch['labels'][i]['class_labels']

        patches, texts = [], []
        for label, bbox_centerxywh in zip(labels, bboxes_centerxywh_rel):
            patch, text = draw_bbox_centerxywh_rel(ax, bbox_centerxywh, label, id2label=id2label, edgecolor='b', linewidth=1)
            patches.append(patch)
            texts.append(text)

        axs_texts.append(texts)
        axs_patches.append(patches)

        ax.axis('off')

def param_count(model):
    """
    Calculate the number of trainable and total parameters in the model.

    :param model: The model to count parameters for
    :type model: torch.nn.Module
    :returns: Total number of parameters, trainable parameters, and the fraction that is trainable
    :rtype: tuple
    """
    params = [(p.numel(), p.requires_grad) for p in model.parameters()]
    trainable = sum([count for count, trainable in params if trainable])
    total = sum([count for count, _ in params])
    frac = (trainable / total) * 100
    return total, trainable, frac
