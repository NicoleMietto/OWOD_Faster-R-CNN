import torch
import torchvision
from torchvision.transforms import functional as F
from OWOD_detector import OWODFasterRCNN
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from torchvision.ops import box_iou

def evaluate_model(checkpoint_path, val_json_path, image_dir, device, use_spatial_cnn=False):
    print(f"\n--- VALUTAZIONE UFFICIALE: {checkpoint_path} ---")
    
    # 1. Carica Modello
    model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=use_spatial_cnn)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # 2. Carica Annotazioni COCO originali
    with open(val_json_path, 'r') as f:
        data = json.load(f)
        
    known_classes = {1, 3, 5, 17, 27, 44, 52, 62, 72, 84}
    
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Variabili per le Metriche
    total_gt_known = 0
    total_gt_unknown = 0
    hits_known = 0
    hits_unknown = 0
    
    all_embeddings = []
    all_labels = []

    print("Calcolo Metriche in corso su TUTTE le 5000 immagini di validazione (Test Finale!)...")
    
    images_to_eval = data['images'] # Valuta su tutto il dataset!
    
    for img_info in tqdm(images_to_eval):
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        anns = img_to_anns[img_info['id']]
        if len(anns) == 0:
            continue
            
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        # --- PARTE 1: Unknown Recall e Known Recall (Inference Standard) ---
        with torch.no_grad():
            detections = model([image_tensor])[0]
            
        pred_boxes = detections['boxes'].cpu().numpy()
        pred_labels = detections['labels'].cpu().numpy()
        pred_scores = detections['scores'].cpu().numpy()
        
        # Separiamo predizioni
        mask_unk_pred = (pred_labels == 81) | (pred_labels == 11) | (pred_labels == 21)
        unk_pred_boxes = pred_boxes[mask_unk_pred & (pred_scores > 0.3)]
        known_pred_boxes = pred_boxes[(~mask_unk_pred) & (pred_scores > 0.3)]
        known_pred_labels = pred_labels[(~mask_unk_pred) & (pred_scores > 0.3)]

        # --- PARTE 2: Recall@1 (Estrazione Embeddings sui Ground Truth) ---
        gt_boxes_for_emb = []
        gt_labels_for_emb = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 10 and h > 10: # ignora troppo piccoli
                gt_box = [x, y, x+w, y+h]
                gt_label = ann['category_id']
                
                gt_boxes_for_emb.append(gt_box)
                gt_labels_for_emb.append(gt_label)
                
                # Calcolo Recalls vettorializzato:
                is_known = gt_label in known_classes
                gt_tensor = torch.tensor([gt_box], dtype=torch.float32)
                
                if is_known:
                    total_gt_known += 1
                    if len(known_pred_boxes) > 0:
                        pred_tensor = torch.tensor(known_pred_boxes, dtype=torch.float32)
                        ious = box_iou(gt_tensor, pred_tensor)[0]
                        if (ious > 0.5).any():
                            hits_known += 1
                else:
                    total_gt_unknown += 1
                    if len(unk_pred_boxes) > 0:
                        pred_tensor = torch.tensor(unk_pred_boxes, dtype=torch.float32)
                        ious = box_iou(gt_tensor, pred_tensor)[0]
                        if (ious > 0.5).any():
                            hits_unknown += 1

        # Estraiamo gli embeddings dei box GT per il Recall@1
        if len(gt_boxes_for_emb) > 0:
            gt_boxes_tensor = torch.tensor(gt_boxes_for_emb, dtype=torch.float32).to(device)
            with torch.no_grad():
                # Creiamo un finto target per far sì che la transform di Faster RCNN
                # scali i nostri bounding box coerentemente con il resize dell'immagine!
                target = [{'boxes': gt_boxes_tensor, 'labels': torch.tensor(gt_labels_for_emb, device=device)}]
                images_list, targets_list = model.detector.transform([image_tensor], target)
                
                features = model.detector.backbone(images_list.tensors)
                # Usiamo le boxes trasformate (targets_list) e le vere dimensioni dell'immagine scalata
                box_features = model.detector.roi_heads.box_roi_pool(features, [t['boxes'] for t in targets_list], images_list.image_sizes)
                
                embeddings = model.embedding_head(box_features)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.extend(gt_labels_for_emb)

    # --- CALCOLO FINALE RECALL@1 ---
    X = np.concatenate(all_embeddings, axis=0)
    Y = np.array(all_labels)
    
    print("\nCalcolo Recall@1 sugli Embeddings...")
    hits_r1 = 0
    # Per ogni embedding, trova il più vicino usando distanza del coseno
    # Normalizziamo X
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Calcolo similarità (lento se troppi oggetti, max 5000)
    for i in tqdm(range(X_norm.shape[0])):
        sims = np.dot(X_norm, X_norm[i])
        sims[i] = -1 # ignora se stesso
        closest_idx = np.argmax(sims)
        if Y[closest_idx] == Y[i]:
            hits_r1 += 1
            
    recall_at_1 = (hits_r1 / len(Y)) * 100
    u_recall = (hits_unknown / total_gt_unknown) * 100 if total_gt_unknown > 0 else 0
    k_recall = (hits_known / total_gt_known) * 100 if total_gt_known > 0 else 0

    print("="*40)
    print(" RISULTATI FINALI (PAPER METRICS) ")
    print("="*40)
    print(f"Known Recall (Proxy per mAP): {k_recall:.2f}% ({hits_known}/{total_gt_known})")
    print(f"Unknown Recall (U-Recall):    {u_recall:.2f}% ({hits_unknown}/{total_gt_unknown})")
    print(f"Recall@1 (Embeddings):        {recall_at_1:.2f}% ({hits_r1}/{len(Y)})")
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
    
    evaluate_model(args.checkpoint, val_json, img_dir, device, args.use_spatial_cnn)
