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
    A.LongestMaxSize(max_size=1333),
    # A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
]) #, bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))

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

class SegmentationToDetectionDataset_CV2(torch.utils.data.Dataset):
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
        boxes = []
        labels = []
        if os.path.exists(rebarmask_file):
            mask1 = cv2.imread(rebarmask_file,0)
            ret,thresh = cv2.threshold(mask1,127,255,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                boxes.append(torch.tensor([x,y,w,h]))
                labels.append(torch.tensor(0))
        if os.path.exists(spallmask_file):
            mask2 = cv2.imread(spallmask_file,0)
            ret,thresh = cv2.threshold(mask2,127,255,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                boxes.append(torch.tensor([x,y,w,h]))
                labels.append(torch.tensor(1))
        if os.path.exists(crackmask_file):
            mask3 = cv2.imread(crackmask_file,0)
            ret,thresh = cv2.threshold(mask3,127,255,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                boxes.append(torch.tensor([x,y,w,h]))
                labels.append(torch.tensor(2))
        img = F.convert_image_dtype(img, dtype=torch.float)
        # mask = F.convert_image_dtype(mask, dtype=torch.float)
        c, h, w = img.shape

        for i, box in enumerate(boxes):
            box = _box_convert._box_xywh_to_xyxy(box)
            x1,y1,x2,y2 = normalize_bbox(box,img.shape[1], img.shape[2]) #`(x_min, y_min, x_max, y_max)`.
            boxes[i] = torch.tensor([x1,y1,x2,y2])
            
        # print("boxes before A", boxes)
        # there is only one class
        # labels = torch.ones((mask.shape[0],), dtype=torch.int64)

        target = {}
        # target["boxes"] = normalize_bbox(boxes, h, w)
        # print("boxes before transform", boxes)
        target["boxes"] = torch.stack(boxes)
        target["labels"] = torch.stack(labels)

        if self.transforms is not None:
            transformed = self.transforms(image=img.numpy(), bboxes=target["boxes"].numpy(), category_ids=labels)
            img = torch.tensor(transformed['image']) #.permute(2,0,1)
            target["boxes"] = torch.tensor(transformed['bboxes'])
            # target["boxes"] = _box_convert._box_xyxy_to_xywh(transformed['bboxes'])
        for i, box in enumerate(target["boxes"]):
            x1,y1,x2,y2 = box
            target["boxes"][i] = torch.tensor([x1,y1,x2,y2])
        # print("boxes after transform", boxes)
        # target["boxes"] = torch.stack(target["boxes"])
        # print("boxes after A", target["boxes"])
        return img, target

def find_box(mask, boxes, labels, label):
    imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        boxes.append(torch.tensor([x,y,w,h]))

        labels.append(torch.tensor(label))
class SegmentationToDetectionDataset_Seg(torch.utils.data.Dataset):
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
        image = cv2.imread(img_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        file = self.imgs[idx]
        filename = os.path.splitext(file)[0]
        # mask_path = os.path.join(self.root, "masks", self.masks[idx])
        figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(2, 2))
        spallmask_file = os.path.join(self.root, "masks", filename + 'spall' + '.jpg')
        rebarmask_file = os.path.join(self.root, "masks",filename + 'rebar' + '.jpg')
        crackmask_file = os.path.join(self.root, "masks", filename + 'crack' + '.jpg')    
        mask1 = np.zeros_like(img)
        mask2 = np.zeros_like(img)
        mask3 = np.zeros_like(img)
        boxes = []
        labels = []
        if os.path.exists(rebarmask_file):
            mask1 = cv2.imread(rebarmask_file)
            # print("mask1 shape",mask1.shape)
        if os.path.exists(spallmask_file):
            mask2 = cv2.imread(spallmask_file)
            # print("mask2 shape",mask2.shape)
        if os.path.exists(crackmask_file):
            mask3 = cv2.imread(crackmask_file)
            # print("mask3 shape",mask3.shape)
        masks = [mask1, mask2, mask3]
        ax[0, 0].imshow(img)
        # ax[0, 1].imshow(mask1)
        # ax[0, 2].imshow(mask2)
        # ax[0, 3].imshow(mask3)
        # print("img shape",img.shape)
        # print(type(masks))
        if self.transforms is not None:
            # transformed = transform(image=img, mask=np.array(masks))
            transformed_image = transform(image=image)['image']
            # transformed_image = torch.tensor(transformed['image'])
            mask1 = transform(image=mask1)['image']
            mask2 = transform(image=mask2)['image']
            mask3 = transform(image=mask3)['image']
            # transformed_mask = transformed['mask']
            # mask1, mask2, mask3 = transformed_mask
        ax[0, 1].imshow(transformed_image)
        # ax[1, 1].imshow(mask1)
        # ax[1, 2].imshow(mask2)
        # ax[1, 3].imshow(mask3)
        # print("masks shape post processing",masks.shape)
        find_box(mask1, boxes, labels, label =0)
        find_box(mask2, boxes, labels, label =1)
        find_box(mask3, boxes, labels, label =2)
        img = F.convert_image_dtype(torch.tensor(transformed_image), dtype=torch.float)
        # mask = F.convert_image_dtype(mask, dtype=torch.float)
        row, col, chan = img.shape
        for i, box in enumerate(boxes):
            box = _box_convert._box_xywh_to_xyxy(box)
            x1,y1,x2,y2 = normalize_bbox(box,row, col) #`(x_min, y_min, x_max, y_max)`.
            boxes[i] = torch.tensor([x1,y1,x2,y2])
        target = {}
        target["boxes"] = torch.stack(boxes)
        target["labels"] = torch.stack(labels)
        target["orig_size"] = torch.as_tensor([int(row), int(col)])
        target["size"] = torch.as_tensor([int(row), int(col)])

        for i, box in enumerate(target["boxes"]):
            x1,y1,x2,y2 = box
            target["boxes"][i] = torch.tensor([x1,y1,x2,y2])
        return img.permute(2,0,1), target

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

import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/best_model.pth')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')