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
    # Analizziamo un po' di immagini (es. prime 300) per trovare le migliori
    for img_info, anns in tqdm(valid_images[:300]):
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

        mask_unk_pred = (pred_labels == 11) | (pred_labels == 81) | (pred_labels == 21)
        unk_pred_boxes = pred_boxes[mask_unk_pred]
        unk_pred_scores = pred_scores[mask_unk_pred]
        
        # Estrai i Ground Truth UNKNOWN reali
        gt_unk_boxes = []
        for ann in anns:
            if ann['category_id'] not in known_classes:
                x, y, w, h = ann['bbox']
                if w > 10 and h > 10:
                    gt_unk_boxes.append([x, y, x+w, y+h])
                    
        # Calcola quanti Hit perfetti abbiamo fatto
        hits = 0
        valid_unk_preds = []
        
        if len(gt_unk_boxes) > 0 and len(unk_pred_boxes) > 0:
            gt_tensor = torch.tensor(gt_unk_boxes, dtype=torch.float32)
            pred_tensor = torch.tensor(unk_pred_boxes, dtype=torch.float32)
            ious = box_iou(gt_tensor, pred_tensor)
            
            # Per ogni predizione, vediamo se ha beccato un GT ignoto
            max_ious, _ = ious.max(dim=0)
            
            for j, iou_val in enumerate(max_ious):
                if iou_val > 0.5:
                    hits += 1
                    valid_unk_preds.append((unk_pred_boxes[j], unk_pred_scores[j]))
                    
        if hits > 0:
            best_results.append({
                'img_info': img_info,
                'img_path': img_path,
                'hits': hits,
                'valid_unk_preds': valid_unk_preds,
                'all_preds': (pred_boxes, pred_labels, pred_scores),
                'gt_unk_boxes': gt_unk_boxes
            })

    # Ordina per numero di hit trovati (immagini più spettacolari)
    best_results.sort(key=lambda x: x['hits'], reverse=True)
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
            ax.text(xmin, ymin-5, "True Unknown (COCO)", color='blue', fontsize=10, weight='bold')

        # 2. Disegna le predizioni della rete
        pred_boxes, pred_labels, pred_scores = res['all_preds']
        for i in range(len(pred_boxes)):
            if pred_scores[i] < 0.3: continue # Mostra solo quelli sicuri
            
            xmin, ymin, xmax, ymax = pred_boxes[i]
            is_unk = (pred_labels[i] == 11 or pred_labels[i] == 81 or pred_labels[i] == 21)
            
            color = 'red' if is_unk else 'green'
            label_name = 'UNKNOWN!' if is_unk else f'Known ({pred_labels[i]})'
            linewidth = 3 if is_unk else 1
            
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=linewidth, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, f"{label_name} {pred_scores[i]:.2f}", color=color, fontsize=10, weight='bold', backgroundcolor='white')

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
