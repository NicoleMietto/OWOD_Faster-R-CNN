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
    print(f"Avvio test visuale comparativo su {device}...")

    # 1. Modelli da testare
    checkpoints_to_test = {
        "best_model": "/kaggle/working/best_model.pth",
        "last_model": "/kaggle/working/owod_model_last.pth"  # O sostituisci con owod_model_epoch_12.pth
    }
    
    # 2. Carichiamo le immagini dal json di TEST
    json_path = "/kaggle/working/task1_10cls_uu_test.json"
    if not os.path.exists(json_path):
        print(f"❌ Errore: {json_path} non trovato. Provo a usare il json di validazione...")
        json_path = "/kaggle/working/task1_10cls_uu_val.json"
        if not os.path.exists(json_path):
             return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Prendiamo 15 immagini fisse per avere un confronto diretto 1-a-1
    images_to_test = data['images'][:15]
    img_dir = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
    
    # 3. Iteriamo sui vari pesi per testarli entrambi
    for model_name, model_path in checkpoints_to_test.items():
        if not os.path.exists(model_path):
            print(f"\n⚠️ Salto {model_name}: file {model_path} non trovato.")
            continue
            
        print(f"\n" + "="*50)
        print(f"TESTING MODELLO: {model_name}")
        print(f"Caricamento pesi da: {model_path}")
        print("="*50)

        # Inizializza un modello fresco
        model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=True)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model.to(device)
        model.eval() # MODALITÀ INFERENZA

        output_dir = f"/kaggle/working/visual_test_results/{model_name}"
        os.makedirs(output_dir, exist_ok=True)

        for img_info in images_to_test:
            img_path = os.path.join(img_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
                
            image = Image.open(img_path).convert("RGB")
            image_tensor = F.to_tensor(image).to(device)

            with torch.no_grad():
                detections = model([image_tensor]) 
            
            pred = detections[0]
            boxes = pred['boxes'].cpu()
            labels = pred['labels'].cpu()
            scores = pred['scores'].cpu()

            image_to_draw = (image_tensor.cpu() * 255).to(torch.uint8)
            colors, text_labels, keep_boxes, keep_scores_list = [], [], [], []
            
            # Filtro base (Soglia 60%)
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.6: 
                    keep_boxes.append(box)
                    keep_scores_list.append(score)
                    
                    if label.item() == 81:
                        colors.append("red") 
                        text_labels.append(f"UNK: {score:.2f}")
                    else:
                        colors.append("green") 
                        text_labels.append(f"KN {label.item()}: {score:.2f}")
            
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
                    colors=colors, width=3, font_size=12
                )
            else:
                drawn_image = image_to_draw

            save_path = os.path.join(output_dir, f"{img_info['file_name']}")
            img_pil = F.to_pil_image(drawn_image)
            img_pil.save(save_path)
            
        print(f"✅ Salvate immagini in '{output_dir}'.")

    print("\n🎉 Finito! Puoi confrontare le cartelle dentro 'visual_test_results/'.")

if __name__ == "__main__":
    main()
