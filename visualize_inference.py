import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from OWOD_detector import OWODFasterRCNN
import os
import json
#import config

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Avvio test visuale su {device}...")

    # 1. Inizializza e carica il modello addestrato sulle 10 classi
    model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=True)
    
    # Prova a caricare best_model.pth, altrimenti owod_model_last.pth
    model_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = "owod_model_last.pth"
        
    if not os.path.exists(model_path):
        print(f"❌ Errore: Nessun modello trovato in {model_path}.")
        return
        
    print(f"Caricamento pesi da: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    # Se il salvataggio contiene 'model_state_dict', carichiamo quello
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    
    # IMPORTANTE: Mettiamo in modalità EVALUATION. 
    # Questo disattiva la richiesta di targets, DINOv2 e SAM.
    model.eval() 

    # 2. Carichiamo le immagini dal json di TEST (immagini mai viste dalla rete!)
    json_path = "/kaggle/working/task1_10cls_uu_test.json"
    if not os.path.exists(json_path):
        print(f"❌ Errore: {json_path} non trovato. Lancia prima generate_10class_splits.py!")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    output_dir = os.path.join(config.OUTPUT_DIR, "visual_test_results")
    os.makedirs(output_dir, exist_ok=True)

    # Prendiamo 15 immagini a caso per il test visivo
    images_to_test = data['images'][:15]
    # Usa la cartella di validation originale di COCO poiché questo JSON deriva da val2017
    img_dir = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017"

    print("Inizio Inferenza...\n")
    for img_info in images_to_test:
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        # Carica immagine
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        # 3. INFERENZA PURA (Senza DINO né SAM)
        with torch.no_grad():
            # Passiamo solo l'immagine, niente targets o feature!
            detections = model([image_tensor]) 
        
        # Estrai risultati
        pred = detections[0]
        boxes = pred['boxes'].cpu()
        labels = pred['labels'].cpu()
        scores = pred['scores'].cpu()

        # 4. Disegna i riquadri
        image_to_draw = (image_tensor.cpu() * 255).to(torch.uint8)
        
        colors = []
        text_labels = []
        keep_boxes = []
        
        # Filtriamo per confidenza e separiamo i colori
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5: # Soglia di confidenza al 50%
                keep_boxes.append(box)
                
                # La nostra label fittizia per Unknown in OWOD_detector è 81
                if label.item() == 81:
                    colors.append("red") # Unknown in ROSSO
                    text_labels.append(f"UNK: {score:.2f}")
                else:
                    colors.append("green") # Known in VERDE
                    text_labels.append(f"KN {label.item()}: {score:.2f}")
        
        if len(keep_boxes) > 0:
            keep_boxes = torch.stack(keep_boxes)
            drawn_image = torchvision.utils.draw_bounding_boxes(
                image_to_draw, 
                keep_boxes, 
                labels=text_labels, 
                colors=colors, 
                width=3,
                font_size=12
            )
        else:
            drawn_image = image_to_draw

        # Salva immagine su disco
        save_path = os.path.join(output_dir, f"pred_{img_info['file_name']}")
        img_pil = F.to_pil_image(drawn_image)
        img_pil.save(save_path)
        print(f"✅ Salvata {save_path} con {len(keep_boxes)} oggetti rilevati.")

    print(f"\n🎉 Finito! Vai nella cartella '{output_dir}' per vedere visivamente i risultati.")

if __name__ == "__main__":
    main()
