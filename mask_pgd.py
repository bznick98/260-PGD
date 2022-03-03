import os
import torch
import torchvision
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pprint import pprint
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms


def load_model():
    # load a model pre-trained on COCO-2017 with backbone=Resnet-50, neck=FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def infer_single(model, image):
    """
    @params:
        model: pytorch model
        image: a torch.Tensor with shape NxCxHxW
    @return:
        result: a single image inference result, a dict with field 'boxes', 'labels' and 'scores'
    """
    return model(image)[0]

def draw_bbox(img, detections):
    """
    @params:
        img: a PIL image with shape (CxHxW)
        detections: detection result from model inference (should be a dict)
    @return:
        None
    @side effects:
        show predicted bounding boxes drawn on the image
    """
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)
    # 
    img_w = img.width
    img_h = img.height
    img_size = img_w
    pad_x = max(img_w - img_h, 0) * (img_size / max(img.size))
    pad_y = max(img_h - img_w, 0) * (img_size / max(img.size))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    # choose a color 
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    if detections is not None:
        unique_labels = detections['labels'].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        # browse detections and draw bounding boxes
        boxes = detections['boxes'].detach()
        labels = detections['labels'].detach()
        scores = detections['scores'].detach()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box

            box_h = ((y2 - y1) / unpad_h) * img.size[0]
            box_w = ((x2 - x1) / unpad_w) * img.size[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.size[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.size[1]
            color = bbox_colors[int(np.where(
                unique_labels == int(label))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=int(label), 
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # load pre-trained model
    model = load_model()
    
    # load image file paths
    image_dir = "images" # TODO: make it argparse
    files = os.listdir(image_dir)
    files.sort()

    for filename in files:
        image_path = os.path.join(image_dir, filename)
        # load single image
        image = Image.open(image_path)

        # transform from PIL Image (HxWxC) into Tensor (NxCxHxW)
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image = torch.Tensor(image) / 255

        # infer single image
        res = infer_single(model, image)

        # visualize detection result
        # tranform image from (NxCxHxW) to (HxWxC) for plotting
        image_PIL = transforms.ToPILImage()(image[0]).convert('RGB')
        draw_bbox(image_PIL, res)

        # TODO: start attack iterations



        break

