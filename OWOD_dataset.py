import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class OWODDataset(Dataset):
    def __init__(self, img_dir, annotation_file, known_classes, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            annotation_file (string): Path to the COCO format json file.
            known_classes (list): List of category IDs that are considered "known" in this task.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.known_classes = known_classes
        
        # Map original COCO classes to continuous IDs (1, 2, ..., N)
        # Faster R-CNN expects background to be 0 and classes from 1 to num_classes
        self.category_id_to_continuous_id = {
            cat_id: i + 1 for i, cat_id in enumerate(self.known_classes)
        }
        
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
            
        self.images = coco_data['images']
        
        # Group annotations by image_id
        self.img_id_to_anns = {img['id']: [] for img in self.images}
        for ann in coco_data['annotations']:
            # Load only annotations of known classes
            if ann['category_id'] in self.category_id_to_continuous_id:
                self.img_id_to_anns[ann['image_id']].append(ann)
                
        print(f"Dataset loaded. Total images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image safely and close the file descriptor immediately
        with Image.open(img_path) as img_file:
            img = img_file.convert("RGB")
        
        # Load annotations
        anns = self.img_id_to_anns[img_info['id']]
        
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO bbox format: [x_min, y_min, width, height]
            x_min, y_min, w, h = ann['bbox']
            
            # Faster R-CNN format: [x_min, y_min, x_max, y_max]
            x_max = x_min + w
            y_max = y_min + h
            
            # Avoid degenerate boxes
            if w > 0 and h > 0:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.category_id_to_continuous_id[ann['category_id']])
                
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_info['id']])
        
        # Transform image to tensor
        img_tensor = TF.to_tensor(img)
        
        if self.transform is not None:
            # Ensure transforms support bounding box logic if used
            img_tensor, target = self.transform(img_tensor, target)
            
        return img_tensor, target
