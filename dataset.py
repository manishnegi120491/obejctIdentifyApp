import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class CocoPersonDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {}
        
        # Group annotations by image_id
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Get list of image IDs that have person annotations
        self.image_ids = list(self.annotations.keys())
        print(f"Found {len(self.image_ids)} images with person annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.annotations[img_id]
        
        # Process bounding boxes
        boxes = []
        labels = []
        
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            xmax, ymax = xmin + w, ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Person class
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        # Apply transforms to image only
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target
