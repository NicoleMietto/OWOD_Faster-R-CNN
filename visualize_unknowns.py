import torch
import torchvision
from torchvision.transforms import functional as F
from OWOD_detector_mlp_filtered import OWODFasterRCNN_Fixed
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from torchvision.ops import box_iou

def visualize_best_unknowns(checkpoint_path, val_json_path, image_dir, output_dir, device, num_images_to_save=10, use_spatial_cnn=False, target_image_id=None, is_baseline=False):
    print(f"Caricamento modello da {checkpoint_path}...")
    
    if is_baseline:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11) # 10 + background
    else:
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
        if target_image_id is not None and img['id'] != target_image_id:
            continue
        anns = img_to_anns[img['id']]
        has_unknown_gt = any(ann['category_id'] not in known_classes for ann in anns)
        if has_unknown_gt:
            valid_images.append((img, anns))
            
    print(f"Trovate {len(valid_images)} immagini con Ground Truth sconosciuti.")
    
    import random
    random.shuffle(valid_images)
    
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
        
        # Filtra Top 100 per partire da una base pulita
        top_k = min(100, len(pred_boxes))
        pred_boxes = pred_boxes[:top_k]
        pred_labels = pred_labels[:top_k]
        pred_scores = pred_scores[:top_k]

        # Separiamo predizioni Known e Unknown
        mask_unk_pred = (pred_labels == 11) | (pred_labels == 81) | (pred_labels == 21)
        
        unk_pred_boxes = pred_boxes[mask_unk_pred]
        unk_pred_scores = pred_scores[mask_unk_pred]
        
        known_pred_boxes = pred_boxes[~mask_unk_pred]
        known_pred_scores = pred_scores[~mask_unk_pred]
        known_pred_labels = pred_labels[~mask_unk_pred]

        # --- NMS SUI KNOWN E UNKNOWN SEPARATAMENTE ---
        # Così mostriamo le predizioni reali senza filtri "magici" legati al Ground Truth
        if len(known_pred_boxes) > 0:
            k_tensor = torch.tensor(known_pred_boxes, dtype=torch.float32)
            k_scores = torch.tensor(known_pred_scores, dtype=torch.float32)
            keep_k = torchvision.ops.nms(k_tensor, k_scores, iou_threshold=0.4)
            
            # Applichiamo NMS e filtriamo per soglia minima (es. > 0.3)
            keep_idx = keep_k.numpy()
            high_conf_mask = known_pred_scores[keep_idx] > 0.3
            keep_idx = keep_idx[high_conf_mask]
            
            # Prendiamo i top 10 KNOWN
            known_pred_boxes = known_pred_boxes[keep_idx][:10]
            known_pred_scores = known_pred_scores[keep_idx][:10]
            known_pred_labels = known_pred_labels[keep_idx][:10]

        if len(unk_pred_boxes) > 0:
            u_tensor = torch.tensor(unk_pred_boxes, dtype=torch.float32)
            u_scores = torch.tensor(unk_pred_scores, dtype=torch.float32)
            keep_u = torchvision.ops.nms(u_tensor, u_scores, iou_threshold=0.4)
            # Prendiamo i top 5 UNKNOWN
            unk_pred_boxes = unk_pred_boxes[keep_u.numpy()][:5]
            unk_pred_scores = unk_pred_scores[keep_u.numpy()][:5]
            
        # Estraiamo i Ground Truth Sconosciuti (solo per ordinare le immagini e disegnarli come riferimento)
        gt_unk_boxes = []
        for ann in anns:
            if ann['category_id'] not in known_classes:
                x, y, w, h = ann['bbox']
                if w > 10 and h > 10:
                    gt_unk_boxes.append([x, y, x+w, y+h])

        max_iou_in_image = 0.0
        if len(gt_unk_boxes) > 0 and len(unk_pred_boxes) > 0:
            gt_tensor = torch.tensor(gt_unk_boxes, dtype=torch.float32)
            pred_tensor = torch.tensor(unk_pred_boxes, dtype=torch.float32)
            ious = box_iou(gt_tensor, pred_tensor)
            
            max_ious, _ = ious.max(dim=0)
            if len(max_ious) > 0:
                max_iou_in_image = max_ious.max().item()

        # Salviamo TUTTO, a prescindere dal fatto che abbiano beccato il GT o meno (Confronto FAIR)
        best_results.append({
            'img_info': img_info,
            'img_path': img_path,
            'max_iou': max_iou_in_image,
            'unk_pred_boxes': unk_pred_boxes,
            'unk_pred_scores': unk_pred_scores,
            'known_pred_boxes': known_pred_boxes,
            'known_pred_scores': known_pred_scores,
            'known_pred_labels': known_pred_labels,
            'gt_unk_boxes': gt_unk_boxes
        })
            
        # Ferma la ricerca appena troviamo un buon pool di immagini da cui estrarre le migliori
        if len(best_results) >= 150:
            print("\nTrovato un ampio campione di immagini candidate. Interrompo la ricerca!")
            break

    # Ordina per MASSIMO IOU trovato nell'immagine (vogliamo le immagini con le sovrapposizioni più evidenti)
    # in modo da mostrarti le immagini "interessanti", ma al loro interno vedrai TUTTA la verità, anche gli errori.
    best_results.sort(key=lambda x: x['max_iou'], reverse=True)
    top_results = best_results[:num_images_to_save]
    
    print(f"\nGenerazione di {len(top_results)} immagini spettacolari nella cartella {output_dir}...")
    
    for idx, res in enumerate(top_results):
        img_info = res['img_info']
        image = Image.open(res['img_path']).convert("RGB")
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # 1. Disegna i Ground Truth Sconosciuti REALI in BLU tratteggiato
        # (L'abbiamo disattivato per rendere l'immagine meno caotica e più pulita per il paper)
        # for gt_box in res['gt_unk_boxes']:
        #     xmin, ymin, xmax, ymax = gt_box
        #     rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        #     ax.add_patch(rect)
        #     ax.text(xmin, ymin-5, "True Unknown (Hidden GT)", color='blue', fontsize=12, weight='bold', backgroundcolor='white')
        # 2. Disegna i box KNOWN in VERDE
        for box, score, label in zip(res['known_pred_boxes'], res['known_pred_scores'], res['known_pred_labels']):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            label_text = f"KN-{label} ({score:.2f})"
            ax.text(xmin, ymin-5, label_text, color='white', fontsize=12, weight='bold', backgroundcolor='green')

        # 3. Disegna i box UNKNOWN PREDETTI in ROSSO
        for box, score in zip(res['unk_pred_boxes'], res['unk_pred_scores']):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            label_text = f"UNK ({score:.2f})"
            ax.text(xmin, ymin-5, label_text, color='white', fontsize=12, weight='bold', backgroundcolor='red')

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
    parser.add_argument("--target_image_id", type=int, default=None, help="Esegui solo su questo specifico Image ID")
    parser.add_argument("--is_baseline", action="store_true", help="Usa il modello baseline a 11 classi (senza unknown esplicita)")
    args = parser.parse_args()
    
    val_json = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_val2017.json"
    img_dir = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
    output_dir = "/kaggle/working/unknown_visualizations"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    visualize_best_unknowns(args.checkpoint, val_json, img_dir, output_dir, device, num_images_to_save=10, use_spatial_cnn=args.use_spatial_cnn, target_image_id=args.target_image_id, is_baseline=args.is_baseline)
