import torch
import torchvision
from torchvision.transforms import functional as F
from OWOD_detector_fixed import OWODFasterRCNN_Fixed
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from torchvision.ops import box_iou

def visualize_best_unknowns(checkpoint_path, val_json_path, image_dir, output_dir, device, num_images_to_save=10, use_spatial_cnn=False):
    print(f"Caricamento modello da {checkpoint_path}...")
    model = OWODFasterRCNN_Fixed(num_known_classes=10, use_spatial_cnn=use_spatial_cnn)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    print("Caricamento annotazioni COCO...")
    with open(val_json_path, 'r') as f:
        data = json.load(f)
        
    known_classes = {1, 3, 5, 17, 27, 44, 52, 62, 72, 84}
    
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    os.makedirs(output_dir, exist_ok=True)
    
    # Filtra solo le immagini che contengono ALMENO un Ground Truth Sconosciuto
    valid_images = []
    for img in data['images']:
        anns = img_to_anns[img['id']]
        has_unknown_gt = any(ann['category_id'] not in known_classes for ann in anns)
        if has_unknown_gt:
            valid_images.append((img, anns))
            
    print(f"Trovate {len(valid_images)} immagini con Ground Truth sconosciuti.")
    
    best_results = []

    print("Ricerca delle migliori detection sconosciute...")
    # Analizziamo le immagini finché non ne troviamo abbastanza di spettacolari
    for img_info, anns in tqdm(valid_images):
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path): continue
            
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                detections = model([image_tensor])[0]
            
        pred_boxes = detections['boxes'].cpu().numpy()
        pred_labels = detections['labels'].cpu().numpy()
        pred_scores = detections['scores'].cpu().numpy()
        
        # Filtra Top 100
        top_k = min(100, len(pred_boxes))
        pred_boxes = pred_boxes[:top_k]
        pred_labels = pred_labels[:top_k]
        pred_scores = pred_scores[:top_k]

        # Separiamo predizioni Known e Unknown
        mask_unk_pred = (pred_labels == 11) | (pred_labels == 81) | (pred_labels == 21)
        unk_pred_boxes = pred_boxes[mask_unk_pred]
        unk_pred_scores = pred_scores[mask_unk_pred]
        
        # Estraiamo i Ground Truth Known e Unknown per filtrare le predizioni
        gt_unk_boxes = []
        gt_known_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 10 and h > 10:
                if ann['category_id'] not in known_classes:
                    gt_unk_boxes.append([x, y, x+w, y+h])
                else:
                    gt_known_boxes.append([x, y, x+w, y+h])

        # 1. Filtra predizioni sconosciute che si sovrappongono ai GROUND TRUTH KNOWN
        if len(gt_known_boxes) > 0 and len(unk_pred_boxes) > 0:
            unk_tensor = torch.tensor(unk_pred_boxes, dtype=torch.float32)
            known_gt_tensor = torch.tensor(gt_known_boxes, dtype=torch.float32)
            ious = box_iou(unk_tensor, known_gt_tensor)
            max_ious, _ = ious.max(dim=1)
            
            # Scarta box sconosciuti che collidono con box noti reali (>0.3 di IoU)
            keep_mask = (max_ious < 0.3).numpy()
            unk_pred_boxes = unk_pred_boxes[keep_mask]
            unk_pred_scores = unk_pred_scores[keep_mask]

        # 2. Applica NMS (Non-Maximum Suppression) tra le sole predizioni sconosciute 
        # per evitare che "l'immagine esploda" con decine di box sovrapposti
        if len(unk_pred_boxes) > 0:
            unk_tensor = torch.tensor(unk_pred_boxes, dtype=torch.float32)
            unk_scores_tensor = torch.tensor(unk_pred_scores, dtype=torch.float32)
            # torchvision.ops.nms restituisce gli indici da tenere
            keep_indices = torchvision.ops.nms(unk_tensor, unk_scores_tensor, iou_threshold=0.3)
            unk_pred_boxes = unk_tensor[keep_indices].numpy()
            unk_pred_scores = unk_scores_tensor[keep_indices].numpy()
                    
        hits = 0
        valid_unk_preds = []
        max_iou_in_image = 0.0
        
        if len(gt_unk_boxes) > 0 and len(unk_pred_boxes) > 0:
            gt_tensor = torch.tensor(gt_unk_boxes, dtype=torch.float32)
            pred_tensor = torch.tensor(unk_pred_boxes, dtype=torch.float32)
            ious = box_iou(gt_tensor, pred_tensor)
            
            # Troviamo i migliori "Hit" per mostrare le detection sconosciute più precise
            max_ious, _ = ious.max(dim=0)
            for j, iou_val in enumerate(max_ious):
                if iou_val > 0.5:
                    hits += 1
                    valid_unk_preds.append((unk_pred_boxes[j], unk_pred_scores[j], iou_val.item()))
                    if iou_val.item() > max_iou_in_image:
                        max_iou_in_image = iou_val.item()
                    
        if hits > 0:
            best_results.append({
                'img_info': img_info,
                'img_path': img_path,
                'hits': hits,
                'max_iou': max_iou_in_image,
                'valid_unk_preds': valid_unk_preds,
                'all_preds': (pred_boxes, pred_labels, pred_scores),
                'gt_unk_boxes': gt_unk_boxes
            })
            
        # Ferma la ricerca appena troviamo un buon pool di immagini da cui estrarre le migliori
        if len(best_results) >= 150:
            print("\nTrovato un ampio campione di immagini candidate. Interrompo la ricerca!")
            break

    # Ordina per MASSIMO IOU trovato nell'immagine (vogliamo le immagini con le sovrapposizioni più evidenti)
    best_results.sort(key=lambda x: x['max_iou'], reverse=True)
    top_results = best_results[:num_images_to_save]
    
    print(f"\nGenerazione di {len(top_results)} immagini spettacolari nella cartella {output_dir}...")
    
    for idx, res in enumerate(top_results):
        img_info = res['img_info']
        image = Image.open(res['img_path']).convert("RGB")
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # 1. Disegna i Ground Truth Sconosciuti REALI in BLU tratteggiato
        for gt_box in res['gt_unk_boxes']:
            xmin, ymin, xmax, ymax = gt_box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, "True Unknown (Hidden GT)", color='blue', fontsize=12, weight='bold', backgroundcolor='white')

        # 2. Disegna SOLO le predizioni UNKNOWN (Classe 11) che hanno fatto "Hit" (sovrapposizione)
        for unk_pred in res['valid_unk_preds']:
            box, score, iou = unk_pred
            xmin, ymin, xmax, ymax = box
            
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=4, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin-20, f"Predicted UNKNOWN! (IoU: {iou:.2f} | Score: {score:.2f})", color='white', fontsize=12, weight='bold', backgroundcolor='red')

        plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"success_{idx+1}_img_{img_info['id']}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    print(f"Finito! Trovi le immagini in: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_fixed.pth")
    parser.add_argument("--use_spatial_cnn", action="store_true")
    args = parser.parse_args()
    
    val_json = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_val2017.json"
    img_dir = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
    output_dir = "/kaggle/working/unknown_visualizations"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    visualize_best_unknowns(args.checkpoint, val_json, img_dir, output_dir, device, num_images_to_save=10, use_spatial_cnn=args.use_spatial_cnn)
