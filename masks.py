from __future__ import annotations
import cv2
import numpy as np
import os
import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as F
from xml.dom import minidom
from torchvision.ops import _box_convert
from albumentations.augmentations.bbox_utils import normalize_bbox

label_fields=['rebar', 'spall', 'crack']

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

import albumentations as A
import cv2

# Add transforms in the code later
transform = A.Compose([
    # A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def img2bbox(file):
    img = cv2.imread(file,0)
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    print(M)
    x,y,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return x,y,w,h

class SegmentationToDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.idx_to_class = {
                            0: "rebar",
                            1: "spall",
                            2: "crack",
                        }
    def __len__(self):
        self.filelength = len(self.imgs)
        return self.filelength

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = read_image(img_path)
        file = self.imgs[idx]
        filename = os.path.splitext(file)[0]
        # mask_path = os.path.join(self.root, "masks", self.masks[idx])
        spallmask_file = os.path.join(self.root, "masks", filename + 'spall' + '.jpg')
        rebarmask_file = os.path.join(self.root, "masks",filename + 'rebar' + '.jpg')
        crackmask_file = os.path.join(self.root, "masks", filename + 'crack' + '.jpg')    
        mask1 = torch.zeros_like(img)
        mask2 = torch.zeros_like(img)
        mask3 = torch.zeros_like(img)
        if os.path.exists(rebarmask_file):
            mask1 = read_image(rebarmask_file)
        if os.path.exists(spallmask_file):
            mask2 = read_image(spallmask_file)
        if os.path.exists(crackmask_file):
            mask3 = read_image(crackmask_file)
        mask = mask1 + mask2*2 + mask3*3
        assert img.shape == mask.shape, f"Shape mismatch between mask and image"
        # if self.transforms is not None:
        #     transformed = self.transforms(image = img.numpy(),  mask=mask.numpy())
        #     transformed_image = transformed['image']
        #     transformed_mask = transformed['mask']
        # assert transformed_image.shape == transformed_mask.shape, f"Shape mismatch between mask and image when transforming"
        img = F.convert_image_dtype(img, dtype=torch.float)
        mask = F.convert_image_dtype(mask, dtype=torch.float)
        c, h, w = img.shape
        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)

        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of boolean masks.
        masks = mask == obj_ids[:, None, None]

        boxes = masks_to_boxes(masks)
        for i, box in enumerate(boxes):
            boxes[i] = torch.tensor(normalize_bbox(box,img.shape[1], img.shape[2]))

        # there is only one class
        labels = torch.ones((mask.shape[0],), dtype=torch.int64)

        target = {}
        # target["boxes"] = normalize_bbox(boxes, h, w)
        target["boxes"] = boxes
        target["labels"] = labels

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        if self.transforms is not None:
            transformed = self.transforms(image=img.permute(1,2,0).numpy(), bboxes=target["boxes"], category_ids=labels)
            img = torch.tensor(transformed['image']).permute(2,0,1)
            target["boxes"] = transformed['bboxes']
            # target["boxes"] = _box_convert._box_xyxy_to_xywh(transformed['bboxes'])
        # for i, box in enumerate(target["boxes"]):
        # print(target["boxes"])
        target["boxes"] = _box_convert._box_xyxy_to_xywh(torch.tensor(target["boxes"]))

            # target["labels"] = transformed['class_labels']

        return img, target

import matplotlib.pyplot as plt

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# https://github.com/ZHANGKEON/DIS-YOLO/blob/master/pre_process.py