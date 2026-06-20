import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from OWOD_detector import OWODFasterRCNN
import os
import json
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Avvio test visuale comparativo su {device}...")

    # 1. Modelli da testare
    checkpoints_to_test = {
        "Best Model (Epoche 1-7)": "/kaggle/working/best_model.pth",
        "Last Model (Epoca 12 - Open World)": "/kaggle/working/owod_model_epoch_12.pth"
    }
    
    # 2. Creiamo un Test Split "Al volo" dal VERO COCO validation set (che non è mai stato usato per train/val)
    coco_val_path = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_val2017.json"
    if not os.path.exists(coco_val_path):
        print(f"❌ Errore: {coco_val_path} non trovato!")
        return

    with open(coco_val_path, 'r') as f:
        data = json.load(f)
        
    # Le nostre 10 classi note
    known_classes = {1, 3, 5, 17, 27, 44, 52, 62, 72, 84}
    
    # Raggruppiamo le annotazioni per immagine
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Troviamo immagini che hanno ALMENO una classe nota (regola OWOD)
    images_to_test = []
    for img in data['images']:
        anns = img_to_anns[img['id']]
        if any(ann['category_id'] in known_classes for ann in anns):
            images_to_test.append(img)
            if len(images_to_test) >= 10:
                break
    img_dir = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
    
    # 3. Carichiamo entrambi i modelli IN MEMORIA
    models = {}
    for name, path in checkpoints_to_test.items():
        if os.path.exists(path):
            print(f"Caricamento {name} da {path}...")
            model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=True)
            checkpoint = torch.load(path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            model.to(device)
            model.eval() # MODALITÀ INFERENZA
            models[name] = model
        else:
            print(f"⚠️ ATTENZIONE: Modello {path} non trovato! Salto.")

    if len(models) == 0:
        print("❌ Nessun modello caricato. Interruzione.")
        return

    output_dir = "/kaggle/working/visual_test_comparisons"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Iteriamo sulle immagini
    print("\nInizio inferenza e generazione grafici...")
    for img_info in images_to_test:
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)
        image_to_draw = (image_tensor.cpu() * 255).to(torch.uint8)

        # Creiamo la figura Matplotlib per il Side-by-Side
        fig, axes = plt.subplots(1, len(models), figsize=(12 * len(models), 12))
        if len(models) == 1:
            axes = [axes]

        for idx, (model_name, model) in enumerate(models.items()):
            with torch.no_grad():
                detections = model([image_tensor]) 
            
            pred = detections[0]
            boxes = pred['boxes'].cpu()
            labels = pred['labels'].cpu()
            scores = pred['scores'].cpu()

            colors, text_labels, keep_boxes, keep_scores_list = [], [], [], []
            
            # Filtro base (Soglia 50% per vedere più roba)
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5: 
                    keep_boxes.append(box)
                    keep_scores_list.append(score)
                    
                    # 81 (o 21 a seconda del dataset) è l'etichetta fittizia per Unknown
                    if label.item() == 81 or label.item() == 21 or label.item() == 11:
                        colors.append("red") 
                        text_labels.append(f"UNKNOWN: {score:.2f}")
                    else:
                        colors.append("green") 
                        text_labels.append(f"Known {label.item()}: {score:.2f}")
            
            if len(keep_boxes) > 0:
                keep_boxes = torch.stack(keep_boxes)
                keep_scores_tensor = torch.stack(keep_scores_list)
                
                # Visual NMS
                final_keep = torchvision.ops.nms(keep_boxes, keep_scores_tensor, iou_threshold=0.3)
                
                keep_boxes = keep_boxes[final_keep]
                colors = [colors[i] for i in final_keep]
                text_labels = [text_labels[i] for i in final_keep]
                
                drawn_image = torchvision.utils.draw_bounding_boxes(
                    image_to_draw, keep_boxes, labels=text_labels, 
                    colors=colors, width=4
                )
            else:
                drawn_image = image_to_draw

            img_pil = F.to_pil_image(drawn_image)
            axes[idx].imshow(img_pil)
            axes[idx].set_title(model_name, fontsize=20, fontweight='bold')
            axes[idx].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"compare_{img_info['file_name']}")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    print(f"\n✅ Finito! Immagini comparative salvate nella cartella '{output_dir}'.")
    print("Vai a vederle e cerca i riquadri ROSSI nell'ultimo modello!")

if __name__ == "__main__":
    main()
