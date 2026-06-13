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
    print(f"Inizio fase di Test su: {device}")

    # (Non necessario per l'inferenza pura o per testare le predizioni)

    # ==========================================
    # 2. Setup Dataset Test
    # ==========================================
    # Assumiamo di aver generato 'task1_uu_test.json'
    test_json_path = "task1_uu_test.json"
    
    # Task 1 classes (come in train.py)
    TASK_1_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    # Nel test, le cartelle delle immagini puntano al val2017 originale di COCO
    test_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017", 
        annotation_file=test_json_path, 
        known_classes=TASK_1_CLASSES,
        transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Inizializziamo l'oggetto COCO per la valutazione
    coco_gt = COCO(test_json_path)

    # ==========================================
    # 3. Setup Modello e Caricamento Pesi
    # ==========================================
    model = OWODFasterRCNN(num_known_classes=20, use_spatial_cnn=True).to(device)
    
    print("Caricamento pesi da best_model.pth...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    else:
        print("ATTENZIONE: best_model.pth non trovato. Verranno usati i pesi non addestrati!")

    # Disabilitiamo ETM e URM per l'inferenza, calcoleremo solo le loss base
    model.use_etm = False
    model.use_urm = False

    total_loss_sum = 0.0

    coco_predictions = []

    print("Inizio inferenza e calcolo loss...")
    # Loop di test
    for images, targets in tqdm(test_loader, desc="Test Iteration"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Nessuna feature DINOv2 necessaria
        val_dino_features = None
        
        # 1. PASSO TRAIN (per calcolare le loss normali)
        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets, val_dino_features)
            
            # Somma delle loss (solo RPN e RoI base)
            batch_total = sum(loss for loss in loss_dict.values()).item()
            total_loss_sum += batch_total
            
        # 2. PASSO EVAL (per le predizioni effettive su pycocotools)
        model.eval()
        with torch.no_grad():
            # In eval mode, Faster R-CNN ignora i targets e restituisce le predizioni
            predictions = model(images, dino_features_list=val_dino_features)
            
            for i, pred in enumerate(predictions):
                image_id = targets[i]['image_id'].item()
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    # COCO Eval si aspetta box formato [x, y, w, h] anziché [x1, y1, x2, y2]
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # Convertiamo l'ID continuo della label indietro all'ID originale di COCO
                    if label <= len(TASK_1_CLASSES):
                        coco_label_id = TASK_1_CLASSES[label - 1]
                    else:
                        coco_label_id = -1 # Classe unknown fittizia (999 o altro, pycocotools valuta solo le known per mAP)
                    
                    coco_predictions.append({
                        "image_id": image_id,
                        "category_id": int(coco_label_id),
                        "bbox": [float(x_min), float(y_min), float(w), float(h)],
                        "score": float(score)
                    })

    # ==========================================
    # 4. Report delle Loss
    # ==========================================
    num_batches = len(test_loader)
    print("\n" + "="*40)
    print("REPORT DELLE LOSS SUL TEST SET:")
    print(f"Loss Totale Media (RPN + RoI): {total_loss_sum/num_batches:.4f}")
    print("="*40 + "\n")

    # ==========================================
    # 5. Valutazione mAP con pycocotools
    # ==========================================
    if len(coco_predictions) > 0:
        print("Calcolo mAP con COCOeval...")
        with open("test_predictions.json", "w") as f:
            json.dump(coco_predictions, f)
            
        coco_dt = coco_gt.loadRes("test_predictions.json")
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Valutiamo solo le classi note (Task 1) per il calcolo mAP
        coco_eval.params.catIds = TASK_1_CLASSES
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    else:
        print("Nessuna predizione generata dal modello.")

if __name__ == "__main__":
    main()
