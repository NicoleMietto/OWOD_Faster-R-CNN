import torch
import torchvision
from torchvision.transforms import functional as F
# IMPORTA L'ARCHITETTURA VECCHIA (11 classi invece di 12)
from OWOD_detector import OWODFasterRCNN
import json
import os
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image

def evaluate_map(checkpoint_path, val_json_path, image_dir, device, use_spatial_cnn=False):
    print(f"\n--- CALCOLO mAP UFFICIALE (MODELLO VECCHIO): {checkpoint_path} ---")
    
    # USA L'ARCHITETTURA VECCHIA
    model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=use_spatial_cnn)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    with open(val_json_path, 'r') as f:
        data = json.load(f)
        
    known_classes = {1, 3, 5, 17, 27, 44, 52, 62, 72, 84}
    
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    valid_images = [img for img in data['images'] if len(img_to_anns[img['id']]) > 0]
    
    images_to_eval = valid_images
    
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=False)
    
    print(f"Calcolo mAP in corso su {len(images_to_eval)} immagini di validazione...")
    
    for img_info in tqdm(images_to_eval):
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        anns = img_to_anns[img_info['id']]
        if len(anns) == 0:
            continue
            
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                detections = model([image_tensor])[0]
            
        pred_boxes = detections['boxes'].cpu()
        pred_labels = detections['labels'].cpu()
        pred_scores = detections['scores'].cpu()
        
        # Filtro (Senza la classe 11/21/81 degli sconosciuti, dato che non c'era)
        top_k = min(100, len(pred_boxes))
        pred_boxes = pred_boxes[:top_k]
        pred_labels = pred_labels[:top_k]
        pred_scores = pred_scores[:top_k]
        
        preds = [
            dict(
                boxes=pred_boxes,
                scores=pred_scores,
                labels=pred_labels,
            )
        ]
        
        gt_boxes_list = []
        gt_labels_list = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if ann['category_id'] in known_classes:
                gt_boxes_list.append([x, y, x+w, y+h])
                gt_labels_list.append(ann['category_id'])
                
        if len(gt_boxes_list) > 0:
            target = [
                dict(
                    boxes=torch.tensor(gt_boxes_list, dtype=torch.float32),
                    labels=torch.tensor(gt_labels_list, dtype=torch.int64),
                )
            ]
            
            metric.update(preds, target)

    print("\nCalcolo finale mAP in corso...")
    result = metric.compute()
    
    print("="*40)
    print(" RISULTATI mAP (COCO Standard) ")
    print("="*40)
    print(f"mAP (IoU=0.50:0.95): {result['map'].item() * 100:.2f}%")
    print(f"mAP@50 (IoU=0.50):   {result['map_50'].item() * 100:.2f}%")
    print(f"mAP@75 (IoU=0.75):   {result['map_75'].item() * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to epoch_X.pth")
    parser.add_argument("--use_spatial_cnn", action="store_true", help="Set if model was trained with CNN")
    args = parser.parse_args()
    
    val_json = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_val2017.json"
    img_dir = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    evaluate_map(args.checkpoint, val_json, img_dir, device, args.use_spatial_cnn)
