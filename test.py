import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

from OWOD_dataset import OWODDataset
from OWOD_detector import OWODFasterRCNN

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Starting Test phase on: {device}")

    # ==========================================
    # 2. Test Dataset Setup
    # ==========================================
    # We assume 'task1_uu_test.json' has been generated in /kaggle/working/
    test_json_path = "/kaggle/working/task1_uu_test.json"
    
    # Task 1 classes (same as train.py)
    TASK_1_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    # For testing, image folders point to the original val2017 of COCO
    test_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017", 
        annotation_file=test_json_path, 
        known_classes=TASK_1_CLASSES,
        transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Initialize the COCO object for evaluation
    coco_gt = COCO(test_json_path)

    # ==========================================
    # 3. Model Setup and Weight Loading
    # ==========================================
    model = OWODFasterRCNN(num_known_classes=20, use_spatial_cnn=True).to(device)
    
    print("Loading weights from best_model.pth...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    else:
        print("WARNING: best_model.pth not found. Untrained weights will be used!")

    # Disable ETM and URM for inference, we will only calculate base losses
    model.use_etm = False
    model.use_urm = False

    total_loss_sum = 0.0

    coco_predictions = []

    print("Starting inference and loss calculation...")
    # Test loop
    for images, targets in tqdm(test_loader, desc="Test Iteration"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # No DINOv2 features required
        val_dino_features = None
        
        # 1. TRAIN PASS (to calculate normal losses)
        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets, val_dino_features)
            
            # Sum of losses (only RPN and base RoI)
            batch_total = sum(loss for loss in loss_dict.values()).item()
            total_loss_sum += batch_total
            
        # 2. EVAL PASS (for actual predictions on pycocotools)
        model.eval()
        with torch.no_grad():
            # In eval mode, Faster R-CNN ignores targets and returns predictions
            predictions = model(images, dino_features_list=val_dino_features)
            
            for i, pred in enumerate(predictions):
                image_id = targets[i]['image_id'].item()
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    # COCO Eval expects box format [x, y, w, h] instead of [x1, y1, x2, y2]
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # Convert the continuous ID back to the original COCO ID
                    if label <= len(TASK_1_CLASSES):
                        coco_label_id = TASK_1_CLASSES[label - 1]
                    else:
                        coco_label_id = -1 # Dummy unknown class (999 or other, pycocotools evaluates only knowns for mAP)
                    
                    coco_predictions.append({
                        "image_id": image_id,
                        "category_id": int(coco_label_id),
                        "bbox": [float(x_min), float(y_min), float(w), float(h)],
                        "score": float(score)
                    })

    # ==========================================
    # 4. Loss Report
    # ==========================================
    num_batches = len(test_loader)
    print("\n" + "="*40)
    print("LOSS REPORT ON TEST SET:")
    print(f"Average Total Loss (RPN + RoI): {total_loss_sum/num_batches:.4f}")
    print("="*40 + "\n")

    # ==========================================
    # 5. mAP Evaluation with pycocotools
    # ==========================================
    if len(coco_predictions) > 0:
        print("Calculating mAP with COCOeval...")
        with open("test_predictions.json", "w") as f:
            json.dump(coco_predictions, f)
            
        coco_dt = coco_gt.loadRes("test_predictions.json")
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # We only evaluate known classes (Task 1) for the mAP calculation
        coco_eval.params.catIds = TASK_1_CLASSES
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    else:
        print("No predictions generated by the model.")

if __name__ == "__main__":
    main()
